from __future__ import annotations

from types import SimpleNamespace

import pytest

from afd_plugin.v1.worker.cuda_graph import (
    FULL_DECODE_ONLY,
    UBATCH_PREPARE_POSITIONS,
    AFDGraphRunMode,
    cudagraph_mode_name,
    describe_prepare_call,
    graph_run_mode,
    make_ffn_graph_key,
    positional_parameter_names,
    resolve_prepare_arguments,
    split_capture_preparation_events,
    validate_cuda_graph_mode,
)


def _config(
    *,
    enforce_eager,
    cudagraph_mode=None,
    use_ubatching=False,
    num_ubatches=1,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        parallel_config=SimpleNamespace(
            use_ubatching=use_ubatching,
            num_ubatches=num_ubatches,
        ),
    )


def test_cuda_graph_policy_allows_eager():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=True, cudagraph_mode="FULL"),
        role="attention",
    )

    assert policy.enabled is False
    assert policy.allow_attention_full_decode_only is False


def test_cuda_graph_policy_allows_full_decode_only_for_attention():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=False, cudagraph_mode=FULL_DECODE_ONLY),
        role="attention",
    )

    assert policy.enabled is True
    assert policy.mode_name == FULL_DECODE_ONLY
    assert policy.allow_attention_full_decode_only is True
    assert policy.enable_ffn_graph_cache is False


def test_cuda_graph_policy_allows_full_decode_only_for_ffn():
    policy = validate_cuda_graph_mode(
        _config(enforce_eager=False, cudagraph_mode=FULL_DECODE_ONLY),
        role="ffn",
    )

    assert policy.enabled is True
    assert policy.allow_attention_full_decode_only is False
    assert policy.enable_ffn_graph_cache is True


@pytest.mark.parametrize(
    "mode",
    [None, "NONE", "FULL", "PIECEWISE", "FULL_AND_PIECEWISE"],
)
def test_cuda_graph_policy_rejects_non_full_decode_only_graph_modes(mode):
    with pytest.raises(RuntimeError, match="FULL_DECODE_ONLY"):
        validate_cuda_graph_mode(
            _config(enforce_eager=False, cudagraph_mode=mode),
            role="attention",
        )


def test_cuda_graph_policy_allows_two_way_ubatching_with_full_decode_only_graph():
    policy = validate_cuda_graph_mode(
        _config(
            enforce_eager=False,
            cudagraph_mode=FULL_DECODE_ONLY,
            use_ubatching=True,
            num_ubatches=2,
        ),
        role="attention",
    )

    assert policy.enabled is True
    assert policy.allow_cuda_graph_with_ubatching is True


def test_cuda_graph_policy_rejects_unsupported_ubatch_count_with_graph():
    with pytest.raises(RuntimeError, match="ubatching"):
        validate_cuda_graph_mode(
            _config(
                enforce_eager=False,
                cudagraph_mode=FULL_DECODE_ONLY,
                use_ubatching=True,
                num_ubatches=4,
            ),
            role="attention",
        )


def test_cudagraph_mode_name_handles_enum_like_values():
    mode = SimpleNamespace(name=FULL_DECODE_ONLY)

    assert cudagraph_mode_name(_config(enforce_eager=False, cudagraph_mode=mode)) == (
        FULL_DECODE_ONLY
    )


def test_make_ffn_graph_key_matches_original_shape():
    metadata_0 = SimpleNamespace(num_tokens_across_dp_cpu=[3, 5])
    metadata_1 = SimpleNamespace(num_tokens_across_dp_cpu=[7, 11])

    assert make_ffn_graph_key({1: metadata_1, 0: metadata_0}) == (
        (0, (3, 5)),
        (1, (7, 11)),
    )


def test_make_ffn_graph_key_can_aggregate_attention_counts_to_ffn_counts():
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[12] * 8)

    assert make_ffn_graph_key(
        {0: metadata},
        attention_size=8,
        ffn_size=4,
        fallback=24,
    ) == ((0, (24, 24, 24, 24)),)


# --- TP expansion tests ---


def test_make_ffn_graph_key_expands_dp1_tp2():
    """DP=1 has 1 entry in num_tokens_across_dp_cpu; with TP=2 the
    attention_size=2 so the single DP entry must be replicated."""
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[8])

    key = make_ffn_graph_key(
        {0: metadata},
        attention_size=2,
        ffn_size=2,
        fallback=32,
    )
    # OLD (buggy) behaviour returned (32, 32) via the fallback path.
    # Correct behaviour: replicate [8] -> (8, 8), then aggregate.
    assert key == ((0, (8, 8)),)


def test_make_ffn_graph_key_different_tokens_for_dp1_tp2():
    """Prefill (4 tokens) and decode (8 tokens) must produce different keys
    so that the FFN correctly distinguishes EAGER from REPLAY."""
    prefill_meta = SimpleNamespace(num_tokens_across_dp_cpu=[4])
    decode_meta = SimpleNamespace(num_tokens_across_dp_cpu=[8])

    prefill_key = make_ffn_graph_key(
        {0: prefill_meta},
        attention_size=2,
        ffn_size=2,
        fallback=32,
    )
    decode_key = make_ffn_graph_key(
        {0: decode_meta},
        attention_size=2,
        ffn_size=2,
        fallback=32,
    )

    assert prefill_key == ((0, (4, 4)),)
    assert decode_key == ((0, (8, 8)),)
    assert prefill_key != decode_key


def test_make_ffn_graph_key_expands_dp2_tp2():
    """DP=2, TP=2: two DP entries replicated to 4 AFD entries."""
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[4, 8])

    key = make_ffn_graph_key(
        {0: metadata},
        attention_size=4,
        ffn_size=4,
        fallback=32,
    )
    # [4, 8] -> replicate tp_size=2 -> [4, 4, 8, 8] -> aggregate group_size=1
    assert key == ((0, (4, 4, 8, 8)),)


def test_make_ffn_graph_key_dp1_tp1_unchanged():
    """TP=1 should not trigger expansion; behaviour unchanged from before."""
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=[8])

    key = make_ffn_graph_key(
        {0: metadata},
        attention_size=1,
        ffn_size=1,
        fallback=32,
    )
    assert key == ((0, (8,)),)


@pytest.mark.parametrize(
    ("is_graph_replaying", "graph_exists", "expected"),
    [
        (False, True, AFDGraphRunMode.EAGER),
        (True, True, AFDGraphRunMode.REPLAY),
        (True, False, AFDGraphRunMode.EAGER),
    ],
)
def test_graph_run_mode_requires_attention_replaying(
    is_graph_replaying,
    graph_exists,
    expected,
):
    assert (
        graph_run_mode(
            is_warmup=False,
            is_graph_capturing=False,
            is_graph_replaying=is_graph_replaying,
            graph_enabled=True,
            graph_exists=graph_exists,
        )
        is expected
    )


def _descriptor(num_tokens, num_reqs, num_ubatches=1):
    return SimpleNamespace(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_ubatches=num_ubatches,
    )


def test_capture_preparation_events_pair_warmup_and_capture_per_descriptor():
    descriptors = [
        _descriptor(8, 4),
        _descriptor(4, 4, num_ubatches=2),
    ]

    single_events, ubatch_events = split_capture_preparation_events(descriptors)

    assert [(d.num_tokens, warmup) for d, warmup in single_events] == [
        (8, True),
        (8, False),
    ]
    assert [(d.num_tokens, warmup) for d, warmup in ubatch_events] == [
        (4, True),
        (4, False),
    ]


def test_capture_preparation_events_keep_one_path_when_only_one_is_used():
    single_only = [_descriptor(8, 4)]
    assert split_capture_preparation_events(single_only) == (
        [(single_only[0], True), (single_only[0], False)],
        [],
    )

    ubatch_only = [_descriptor(4, 4, num_ubatches=2)]
    assert split_capture_preparation_events(ubatch_only) == (
        [],
        [(ubatch_only[0], True), (ubatch_only[0], False)],
    )


def _extra_parameter_prepare(
    input_batch,
    block_tables,
    slot_mappings,
    cg_mode=None,
    for_capture=False,
    extra=None,
):
    return extra


def _opaque_prepare(*args, **kwargs):
    """A wrapper whose signature introspection cannot expose parameter names."""
    return args


def test_resolve_prepare_arguments_reads_the_pinned_call_shape():
    batch = object()
    previous = _extra_parameter_prepare
    args = (batch, ("table",), "slots")
    kwargs = {"cg_mode": "FULL", "for_capture": True}

    resolved = resolve_prepare_arguments(
        previous, args, kwargs, UBATCH_PREPARE_POSITIONS
    )

    assert resolved == {
        "input_batch": batch,
        "cg_mode": "FULL",
        "for_capture": True,
    }
    assert positional_parameter_names(previous) == (
        "input_batch",
        "block_tables",
        "slot_mappings",
        "cg_mode",
        "for_capture",
        "extra",
    )


def test_resolve_prepare_arguments_tolerates_an_unexpected_extra_argument():
    batch = object()
    args = (batch, ("table",), "slots", "FULL", True, "surprise")

    resolved = resolve_prepare_arguments(
        _extra_parameter_prepare, args, {}, UBATCH_PREPARE_POSITIONS
    )

    assert resolved["input_batch"] is batch
    assert resolved["cg_mode"] == "FULL"
    assert resolved["for_capture"] is True
    assert "prepare(" in describe_prepare_call(_extra_parameter_prepare, args, {})


def test_resolve_prepare_arguments_falls_back_to_positions_for_opaque_targets():
    """An unintrospectable wrapper must not lose the fields AFD validates.

    The hardware failure this guards against reported a missing input_batch
    because the resolved signature exposed no parameter names.
    """
    batch = object()
    args = (batch, ("table",), "slots", "FULL", True)

    assert positional_parameter_names(_opaque_prepare) == ()
    assert resolve_prepare_arguments(
        _opaque_prepare, args, {}, UBATCH_PREPARE_POSITIONS
    ) == {"input_batch": batch, "cg_mode": "FULL", "for_capture": True}


def test_resolve_prepare_arguments_omits_names_it_cannot_resolve():
    resolved = resolve_prepare_arguments(
        _opaque_prepare, (object(),), {}, UBATCH_PREPARE_POSITIONS
    )

    assert tuple(resolved) == ("input_batch",)
