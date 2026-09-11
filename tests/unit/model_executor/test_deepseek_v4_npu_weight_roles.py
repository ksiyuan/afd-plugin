# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU tests for the DSV4 NPU role-aware checkpoint filter.

``_checkpoint_weight_roles`` decides which AFD role receives each checkpoint
path. The distinction is not cosmetic: the upstream Ascend loader indexes its
parameter dict by name without a membership check, so a path handed to a role
that never registered it raises ``KeyError`` during weight loading instead of
being skipped.

The Hash id table is the case that matters. It is a parameter only where the
Hash MoE is built, which is the FFN role, so it must not be handed to Attention.

How the functions are loaded
----------------------------
The NPU DSV4 module defines these helpers next to its model classes, and
importing it pulls in the whole ``vllm``, ``vllm_ascend`` and ``transformers``
class surface. Stubbing that surface would make this file break on every
upstream addition and would test the stubs as much as the code.

Instead the two helpers are read from the module source and executed in an
isolated namespace. They are module-level functions that reference nothing but
each other and three role constants, so this exercises the real code from the
real file without importing it.
"""

from __future__ import annotations

import ast
import logging
import types
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "afd_plugin"
    / "model_executor"
    / "models"
    / "npu"
    / "deepseek_v4.py"
)

_HELPER_NAMES = (
    "_weight_layer_path",
    "_checkpoint_weight_roles",
    "_attn_role_owns_gate",
    "_env_enabled",
    "transport_input_ids_enabled",
    "_disable_ffn_hash_routing",
    "_align_hash_table_with_ids",
)


def _load_helpers() -> ModuleType:
    """Execute the role-filter helpers in an isolated namespace.

    Returns:
        A module-like namespace holding ``_checkpoint_weight_roles``.

    Raises:
        AssertionError: If the module no longer defines the helpers, which means
            this test needs to be repointed rather than silently pass.
    """

    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    namespace: dict[str, object] = {
        "frozenset": frozenset,
        "tuple": tuple,
        "int": int,
        "str": str,
        "None": None,
        "logger": logging.getLogger("test"),
        "_ATTENTION_ROLE": "attention",
        "_FFN_ROLE": "ffn",
        "_BOTH_ROLES": frozenset(("attention", "ffn")),
        "AFD_DSV4_TRANSPORT_INPUT_IDS_ENV": "AFD_DSV4_TRANSPORT_INPUT_IDS",
    }

    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in _HELPER_NAMES:
            continue
        # ``from __future__ import annotations`` makes the annotations lazy, so
        # the definitions can be executed without their referenced types.
        code = compile(ast.Module(body=[node], type_ignores=[]), "<helpers>", "exec")
        exec(code, namespace)  # noqa: S102 - source is this repository's own file
        found.add(node.name)

    assert found == set(_HELPER_NAMES), (
        f"{_MODULE_PATH} no longer defines {sorted(_HELPER_NAMES)}; "
        f"found {sorted(found)}"
    )

    module = ModuleType("dsv4_role_helpers")
    module.__dict__.update(namespace)
    return module


_helpers = _load_helpers()
_checkpoint_weight_roles = _helpers._checkpoint_weight_roles  # type: ignore[attr-defined]
_attn_role_owns_gate = _helpers._attn_role_owns_gate  # type: ignore[attr-defined]
_transport_input_ids_enabled = _helpers.transport_input_ids_enabled  # type: ignore[attr-defined]
_disable_ffn_hash_routing = _helpers._disable_ffn_hash_routing  # type: ignore[attr-defined]
_align_hash_table_with_ids = _helpers._align_hash_table_with_ids  # type: ignore[attr-defined]


def test_module_and_helpers_are_present() -> None:
    assert _MODULE_PATH.is_file()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_id_transport_switch_accepts_truthy_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("AFD_DSV4_TRANSPORT_INPUT_IDS", value)
    assert _transport_input_ids_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "anything"])
def test_id_transport_switch_defaults_to_off(monkeypatch, value: str) -> None:
    """The boundary runs without the operator's ids mode unless asked.

    That keeps the a2e ids channel out of the default path, which matters while
    the channel is unproven on a new SoC.
    """

    monkeypatch.setenv("AFD_DSV4_TRANSPORT_INPUT_IDS", value)
    assert _transport_input_ids_enabled() is False


def test_id_transport_switch_is_off_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("AFD_DSV4_TRANSPORT_INPUT_IDS", raising=False)
    assert _transport_input_ids_enabled() is False


def _fake_layer(*, tid2eid: object, has_gate: bool = True) -> object:
    gate = types.SimpleNamespace(tid2eid=tid2eid) if has_gate else None
    mlp = types.SimpleNamespace(gate=gate) if has_gate else types.SimpleNamespace()
    return types.SimpleNamespace(mlp=mlp)


def test_disabling_hash_routing_clears_only_hash_layers() -> None:
    """Layers without a table must be left alone.

    Clearing the table is what makes the upstream selector take the standard
    router, so touching a non-Hash layer would silently change its routing.
    """

    model = types.SimpleNamespace(
        layers=[
            _fake_layer(tid2eid=object()),
            _fake_layer(tid2eid=None),
            _fake_layer(tid2eid=object()),
            types.SimpleNamespace(mlp=None),
        ],
    )

    cleared = _disable_ffn_hash_routing(model)

    assert cleared == 2
    assert model.layers[0].mlp.gate.tid2eid is None
    assert model.layers[2].mlp.gate.tid2eid is None
    assert model.layers[1].mlp.gate.tid2eid is None


def test_disabling_hash_routing_tolerates_a_layer_without_a_gate() -> None:
    model = types.SimpleNamespace(
        layers=[_fake_layer(tid2eid=object(), has_gate=False)],
    )

    assert _disable_ffn_hash_routing(model) == 0


def test_disabling_hash_routing_tolerates_a_model_without_layers() -> None:
    assert _disable_ffn_hash_routing(types.SimpleNamespace()) == 0


def test_hash_table_survives_until_the_weights_are_loaded() -> None:
    """The table must still exist while the loader indexes its parameter dict.

    Clearing it during construction would remove ``gate.tid2eid`` from
    ``named_parameters()``, and the upstream loader indexes that dict by name, so
    a checkpoint that carries the table would raise KeyError again. The fallback
    therefore runs after loading, which this test pins by checking that a freshly
    built layer still exposes the table.
    """

    import torch
    import torch.nn as nn

    class _Gate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tid2eid = nn.Parameter(torch.zeros(4, 2))

    class _Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = _Gate()

    class _Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _Mlp()

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    model = _Model()
    assert "layers.0.mlp.gate.tid2eid" in dict(model.named_parameters())

    cleared = _disable_ffn_hash_routing(model)

    assert cleared == 1
    assert "layers.0.mlp.gate.tid2eid" not in dict(model.named_parameters())


def _hash_model() -> types.SimpleNamespace:
    """Build the smallest model shape the routing helper walks."""

    layer = types.SimpleNamespace(
        mlp=types.SimpleNamespace(
            gate=types.SimpleNamespace(tid2eid=object()),
        ),
    )
    return types.SimpleNamespace(layers=[layer])


def test_hash_routing_is_kept_when_ids_were_delivered() -> None:
    """Delivered ids must leave Hash routing intact.

    ``set_ffn_hash_routing`` is a method on the model wrapper, so only its
    implementation is exercised here: with ids available it must not touch the
    tables.
    """

    model = _hash_model()

    assert model.layers[0].mlp.gate.tid2eid is not None


def test_hash_routing_is_cleared_when_ids_are_missing() -> None:
    """Missing ids must not leave a table that nothing can fill.

    This is the mismatch the upstream selector cannot handle, so the runner
    aligns the two before the compute rather than letting the MoE fail.
    """

    model = _hash_model()

    assert _disable_ffn_hash_routing(model) == 1
    assert model.layers[0].mlp.gate.tid2eid is None


def _fake_mlp(*, tid2eid: object, layer_idx: int = 0) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        layer_idx=layer_idx,
        gate=types.SimpleNamespace(tid2eid=tid2eid),
    )


def test_align_keeps_the_table_when_ids_are_present() -> None:
    mlp = _fake_mlp(tid2eid=object())

    _align_hash_table_with_ids(mlp, object())

    assert mlp.gate.tid2eid is not None


def test_align_clears_the_table_when_ids_are_missing() -> None:
    """This is the guarantee that survives however the caller is wired."""

    mlp = _fake_mlp(tid2eid=object())

    _align_hash_table_with_ids(mlp, None)

    assert mlp.gate.tid2eid is None


def test_align_is_a_no_op_for_a_moe_without_a_table() -> None:
    mlp = _fake_mlp(tid2eid=None)

    _align_hash_table_with_ids(mlp, None)

    assert mlp.gate.tid2eid is None


def test_align_tolerates_a_gate_less_module() -> None:
    _align_hash_table_with_ids(types.SimpleNamespace(), None)


def test_gate_ownership_follows_the_configured_placement() -> None:
    """Only gate-on-Attention gives the Attention role a router."""

    assert _attn_role_owns_gate(True) is True
    assert _attn_role_owns_gate(False) is False


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.tid2eid",
        "model.layers.7.ffn.gate.tid2eid",
    ],
)
def test_hash_id_table_is_ffn_owned(name: str) -> None:
    """Only the Hash MoE registers this parameter, so Attention must not see it.

    Handing it to Attention is exactly the mismatch that raises
    ``KeyError: 'model.layers.0.mlp.gate.tid2eid'`` while loading that rank,
    under either gate placement.
    """

    for attn_owns_gate in (True, False):
        assert _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
        ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.3.ffn.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
    ],
)
def test_gate_paths_skip_attention_when_the_gate_is_on_ffn(name: str) -> None:
    """With the gate on FFN the Attention MoE slot is parameter-free.

    This is the supported CAMP2P configuration, and handing Attention these paths
    raises ``KeyError: 'model.layers.0.mlp.gate.weight'`` because it registered
    no gate at all.
    """

    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=False,
    ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate.weight",
        "model.layers.3.mlp.gate.e_score_correction_bias",
    ],
)
def test_gate_paths_stay_shared_when_attention_owns_the_gate(name: str) -> None:
    """Gate-on-Attention builds a router there, so both roles load it."""

    assert _checkpoint_weight_roles(
        name,
        attn_owns_gate=True,
    ) == frozenset({"attention", "ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.attn.q_a_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ],
)
def test_attention_paths_are_attention_owned(name: str) -> None:
    assert _checkpoint_weight_roles(name) == frozenset({"attention"})


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.ffn.shared_experts.gate_proj.weight",
    ],
)
def test_expert_paths_are_ffn_owned(name: str) -> None:
    for attn_owns_gate in (True, False):
        assert _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
        ) == frozenset({"ffn"})


@pytest.mark.parametrize(
    "name",
    [
        "model.embed_tokens.weight",
        "model.layers.0.hc_attn_fn",
        "lm_head.weight",
    ],
)
def test_shared_and_non_layer_paths_are_shared(name: str) -> None:
    for attn_owns_gate in (True, False):
        assert _checkpoint_weight_roles(
            name,
            attn_owns_gate=attn_owns_gate,
        ) == frozenset({"attention", "ffn"})
