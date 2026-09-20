# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import test_sync_camp2p_npu as entrypoint
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_SYNC_CAMP2P_A3_SCENARIO,
    DSV4_SYNC_CAMP2P_A5_SCENARIO,
    DSV4_SYNC_SHAPES,
)

# (attention DP, attention TP, FFN DP, FFN TP) each shape must produce.
EXPECTED_PARALLELISM = {
    DSV4_SYNC_CAMP2P_A5_SCENARIO: ("1", "2", "1", "2"),
    DSV4_SYNC_CAMP2P_A3_SCENARIO: ("1", "4", "1", "4"),
}


def _devices_for(scenario: str) -> str:
    count = DSV4_SYNC_SHAPES[scenario].device_count
    return ",".join(str(index) for index in range(count))


def _arguments(
    monkeypatch,
    tmp_path,
    *,
    scenario: str = DSV4_SYNC_CAMP2P_A5_SCENARIO,
    model: str = "/models/dsv4",
):
    monkeypatch.setenv("AFD_E2E_BACKEND", "npu")
    monkeypatch.setenv("AFD_E2E_DEVICES", _devices_for(scenario))
    monkeypatch.setenv("AFD_NPU_E2E_MODEL", model)
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_API_PORT", raising=False)
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_AFD_PORT", raising=False)
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_QUANTIZATION", raising=False)
    for name in (
        "AFD_NPU_DSV4_SYNC_E2E_MAX_MODEL_LEN",
        "AFD_NPU_DSV4_SYNC_E2E_MAX_NUM_BATCHED_TOKENS",
        "AFD_NPU_DSV4_SYNC_E2E_MAX_NUM_SEQS",
        "AFD_NPU_DSV4_SYNC_E2E_MEMORY_UTILIZATION",
    ):
        monkeypatch.delenv(name, raising=False)
    command = entrypoint.build_runner_command(scenario, tmp_path / "responses.json")
    monkeypatch.setattr(sys, "argv", ["runner", *command[3:]])
    return runner.parse_args()


def _model_with_config(root, quant_method: str | None) -> str:
    root.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {}
    if quant_method is not None:
        config["quantization_config"] = {"quant_method": quant_method}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(root)


@pytest.mark.parametrize("scenario", sorted(EXPECTED_PARALLELISM))
def test_dsv4_sync_fixed_deployment(monkeypatch, tmp_path, scenario):
    args = _arguments(monkeypatch, tmp_path, scenario=scenario)
    runner.configure_scenario(args)
    runner.validate_topology(
        args,
        runner.parse_csv(args.attention_devices),
        runner.parse_csv(args.ffn_devices),
    )
    # The synchronous transport must not borrow the async CAM teardown, which
    # defers an FFN SIGKILL for a pending CAM receive.
    assert not runner.uses_npu_async_process_cleanup(args)
    assert args.afd_async is False
    assert args.compute_gate_on_attention is False
    assert args.gsm8k_output_path is None
    expected = EXPECTED_PARALLELISM[scenario]
    for role, dp, tp in (
        ("attention", expected[0], expected[1]),
        ("ffn", expected[2], expected[3]),
    ):
        command = runner.build_vllm_command(args, role=role)
        assert command[command.index("--data-parallel-size") + 1] == dp
        assert command[command.index("--tensor-parallel-size") + 1] == tp
        assert command[command.index("--max-num-batched-tokens") + 1] == "1024"
        assert command[command.index("--max-model-len") + 1] == "8192"
        assert command[command.index("--max-num-seqs") + 1] == "16"
        assert command[command.index("--gpu-memory-utilization") + 1] == "0.7"
        assert command[command.index("--quantization") + 1] == "ascend"
        assert "--enforce-eager" in command
        # DeepSeek V4 shards by tensor parallel here: an expert-parallel world
        # greater than one selects MC2 on A5 and does not tile.
        assert "--enable-expert-parallel" not in command
        assert "--enable-dbo" not in command
        assert "--kv-transfer-config" not in command
        config = json.loads(command[command.index("--additional-config") + 1])
        assert config["enable_dsv4_shared_compressor_workspace"] is False
        assert config["enable_cpu_binding"] is True
        # Exact equality also pins the absent keys: CAMP2P rejects both the
        # asynchronous mode and gate-on-Attention.
        assert config["afd"] == {
            "role": role,
            "connector": "CAMP2pAFDConnector",
            "host": "192.0.2.1",
            "port": 6456,
            "num_attention_ranks": int(expected[0]) * int(expected[1]),
            "num_ffn_ranks": int(expected[2]) * int(expected[3]),
            "connector_extra_config": {
                "hccl_buffer_size": 2048,
                "quant_mode": 0,
            },
        }
        env = runner.build_env("0", args, role=role, e2e_run_id="test")
        assert env["VLLM_PLUGINS"] == "ascend,afd"
        # Attention TP>1 has a TP/SP token split, but this connector path must
        # not inherit the async case's FlashComm1 setting.
        assert "VLLM_ASCEND_ENABLE_FLASHCOMM1" not in env


@pytest.mark.parametrize("scenario", sorted(EXPECTED_PARALLELISM))
def test_dsv4_sync_main_uses_concurrent_requests_and_longer_cleanup(
    monkeypatch,
    tmp_path,
    scenario,
):
    args = _arguments(monkeypatch, tmp_path, scenario=scenario)
    cleanup_options = {}
    evaluations = []
    process = SimpleNamespace(pid=123, poll=lambda: None)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "start_process", lambda *_args: process)
    monkeypatch.setattr(
        runner,
        "stream_output",
        lambda *_args: SimpleNamespace(join=lambda **_kwargs: None),
    )
    monkeypatch.setattr(runner, "wait_for_openai_api", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "run_concurrent_completion_evaluation",
        lambda _args: evaluations.append("concurrent"),
    )
    monkeypatch.setattr(
        runner,
        "run_gsm8k_evaluation",
        lambda _args: pytest.fail("must not run GSM8K"),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes, **kwargs: cleanup_options.update(kwargs),
    )
    assert runner.main() == 0
    assert evaluations == ["concurrent"]
    assert cleanup_options["termination_timeout_s"] == 60
    assert cleanup_options["deferred_sigkill_pgids"] == ()


@pytest.mark.parametrize("scenario", sorted(EXPECTED_PARALLELISM))
def test_dsv4_sync_entrypoint_rejects_wrong_devices(monkeypatch, tmp_path, scenario):
    count = DSV4_SYNC_SHAPES[scenario].device_count
    wrong_counts = [
        ",".join(str(index) for index in range(count - 1)),
        ",".join(str(index) for index in range(count + 1)),
    ]
    for devices in [*wrong_counts, ",".join(["0"] * count)]:
        _arguments(monkeypatch, tmp_path, scenario=scenario)
        monkeypatch.setenv("AFD_E2E_DEVICES", devices)
        with pytest.raises(
            RuntimeError,
            match=rf"exactly {count} devices|devices must be unique",
        ):
            entrypoint.build_runner_command(scenario, tmp_path / "responses.json")


def test_dsv4_sync_scenarios_reject_each_others_device_count(monkeypatch, tmp_path):
    """The two shapes are host-specific; the other host's list must fail."""
    a5_count = DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A5_SCENARIO].device_count
    a3_count = DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A3_SCENARIO].device_count
    assert a5_count != a3_count
    _arguments(monkeypatch, tmp_path, scenario=DSV4_SYNC_CAMP2P_A5_SCENARIO)
    monkeypatch.setenv("AFD_E2E_DEVICES", _devices_for(DSV4_SYNC_CAMP2P_A3_SCENARIO))
    with pytest.raises(RuntimeError, match=rf"exactly {a5_count} devices"):
        entrypoint.build_runner_command(
            DSV4_SYNC_CAMP2P_A5_SCENARIO,
            tmp_path / "responses.json",
        )


@pytest.mark.parametrize(
    "field", ["common_vllm_arg", "attention_vllm_arg", "ffn_vllm_arg"]
)
def test_dsv4_sync_rejects_deployment_overrides(monkeypatch, tmp_path, field):
    args = _arguments(monkeypatch, tmp_path)
    setattr(args, field, ["--max-num-batched-tokens=1"])
    with pytest.raises(ValueError, match="extra vLLM arguments"):
        runner.configure_scenario(args)


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
@pytest.mark.parametrize("scenario", sorted(EXPECTED_PARALLELISM))
def test_dsv4_sync_rejects_non_npu_backends(
    monkeypatch,
    tmp_path,
    scenario,
    backend,
):
    args = _arguments(monkeypatch, tmp_path, scenario=scenario)
    args.device_backend = backend
    runner.configure_scenario(args)
    shape = DSV4_SYNC_SHAPES[scenario]
    with pytest.raises(ValueError, match="require NPU"):
        runner.validate_topology(
            args,
            [str(index) for index in range(shape.attention_ranks)],
            [
                str(index)
                for index in range(
                    shape.attention_ranks,
                    shape.attention_ranks + shape.ffn_ranks,
                )
            ],
        )


@pytest.mark.parametrize(
    ("name", "value", "flag"),
    [
        ("AFD_NPU_DSV4_SYNC_E2E_MAX_MODEL_LEN", "32768", "--max-model-len"),
        (
            "AFD_NPU_DSV4_SYNC_E2E_MAX_NUM_BATCHED_TOKENS",
            "2048",
            "--max-num-batched-tokens",
        ),
        ("AFD_NPU_DSV4_SYNC_E2E_MAX_NUM_SEQS", "12", "--max-num-seqs"),
        (
            "AFD_NPU_DSV4_SYNC_E2E_MEMORY_UTILIZATION",
            "0.85",
            "--gpu-memory-utilization",
        ),
    ],
)
def test_dsv4_sync_runtime_profile_env_overrides(
    monkeypatch,
    tmp_path,
    name,
    value,
    flag,
):
    """A host can retune the context and batch budget without a code change."""
    args = _arguments(monkeypatch, tmp_path)
    monkeypatch.setenv(name, value)
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role="attention")

    assert command[command.index(flag) + 1] == value


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("AFD_NPU_DSV4_SYNC_E2E_MAX_NUM_SEQS", "9"),
        ("AFD_NPU_DSV4_SYNC_E2E_MAX_NUM_SEQS", "nope"),
        ("AFD_NPU_DSV4_SYNC_E2E_MAX_MODEL_LEN", "0"),
        ("AFD_NPU_DSV4_SYNC_E2E_MEMORY_UTILIZATION", "1.5"),
    ],
)
def test_dsv4_sync_runtime_profile_rejects_bad_values(
    monkeypatch,
    tmp_path,
    name,
    value,
):
    args = _arguments(monkeypatch, tmp_path)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        runner.configure_scenario(args)


def test_dsv4_sync_omits_quantization_for_a_declaring_checkpoint(monkeypatch, tmp_path):
    """vLLM rejects `--quantization ascend` against the A5 FP8 checkpoint."""
    model = _model_with_config(tmp_path / "dsv4-fp8", "fp8")
    args = _arguments(monkeypatch, tmp_path, model=model)
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role="attention")

    assert "--quantization" not in command


def test_dsv4_sync_keeps_ascend_for_an_undeclared_checkpoint(monkeypatch, tmp_path):
    model = _model_with_config(tmp_path / "dsv4-w8a8", None)
    args = _arguments(monkeypatch, tmp_path, model=model)
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role="attention")

    assert command[command.index("--quantization") + 1] == "ascend"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("fp8", "fp8"), ("none", None), ("", None), (" ascend ", "ascend")],
)
def test_dsv4_sync_quantization_env_override(
    monkeypatch,
    tmp_path,
    value,
    expected,
):
    args = _arguments(monkeypatch, tmp_path)
    monkeypatch.setenv("AFD_NPU_DSV4_SYNC_E2E_QUANTIZATION", value)
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role="attention")

    if expected is None:
        assert "--quantization" not in command
    else:
        assert command[command.index("--quantization") + 1] == expected


def test_dsv4_sync_environment_needs_no_cam_package(monkeypatch):
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth-test")
    monkeypatch.delenv("CAM_CUST_OPAPI_LIB_PATH", raising=False)
    # A caller shell that exported the async CAM recipe's buffer size must not
    # change this case, which sizes its CAMP2P domains itself.
    monkeypatch.setenv("HCCL_BUFFSIZE", "4096")
    env = entrypoint.build_environment()
    assert env["GLOO_SOCKET_IFNAME"] == "eth-test"
    assert env["TP_SOCKET_IFNAME"] == "eth-test"
    assert env["AFD_FORCE_BALANCED_TOPK_IDS"] == "0"
    assert "CAM_CUST_OPAPI_LIB_PATH" not in env
    assert "HCCL_BUFFSIZE" not in env
