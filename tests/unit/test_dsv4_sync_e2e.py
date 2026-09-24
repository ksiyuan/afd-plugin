# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from tests.e2e import runner
from tests.e2e.models.deepseek_v4_flash import completions
from tests.e2e.models.deepseek_v4_flash import test_sync_camp2p_npu as entrypoint
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_SYNC_CAMP2P_A3_SCENARIO,
    DSV4_SYNC_CAMP2P_A5_SCENARIO,
    DSV4_SYNC_SHAPES,
)

# (attention DP, attention TP, FFN DP, FFN TP) each profile must produce. The
# A5 profile shards by data parallel with expert parallelism, the A3 profile by
# tensor parallel.
EXPECTED_PARALLELISM = {
    DSV4_SYNC_CAMP2P_A5_SCENARIO: ("2", "1", "2", "1"),
    DSV4_SYNC_CAMP2P_A3_SCENARIO: ("1", "4", "1", "4"),
}


def _flag_value(command: list[str], flag: str) -> str | None:
    """Return a flag's value, or None when the scenario omits the flag."""
    return command[command.index(flag) + 1] if flag in command else None


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
    monkeypatch.delenv("AFD_NPU_DSV4_SYNC_E2E_EAGER", raising=False)
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
    profile = DSV4_SYNC_SHAPES[scenario]
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
    assert args.cuda_graph_full_decode_only is profile.use_graph
    assert args.enable_dbo is profile.enable_dbo
    expected = EXPECTED_PARALLELISM[scenario]
    for role, dp, tp in (
        ("attention", expected[0], expected[1]),
        ("ffn", expected[2], expected[3]),
    ):
        command = runner.build_vllm_command(args, role=role)
        assert command[command.index("--data-parallel-size") + 1] == dp
        assert command[command.index("--tensor-parallel-size") + 1] == tp
        assert command[command.index("--max-model-len") + 1] == profile.max_model_len
        assert (
            _flag_value(command, "--max-num-batched-tokens")
            == profile.max_num_batched_tokens
        )
        assert (
            _flag_value(command, "--gpu-memory-utilization")
            == profile.memory_utilization
        )
        assert ("--enable-expert-parallel" in command) is (
            profile.enable_expert_parallel
        )
        assert ("--enable-dbo" in command) is profile.enable_dbo
        assert ("--enforce-eager" in command) is not profile.use_graph
        if profile.compilation_config is not None:
            # A verbatim profile passes its host script's compilation config and
            # nothing else from the case's graph deployment.
            assert (
                json.loads(
                    command[command.index("--compilation-config") + 1],
                )
                == profile.compilation_config
            )
            assert "--cudagraph-capture-sizes" not in command
            assert "--max-cudagraph-capture-size" not in command
        elif profile.use_graph:
            capture_size = str(profile.cudagraph_capture_size)
            assert command[command.index("--cudagraph-capture-sizes") + 1] == (
                capture_size
            )
            assert "--compilation-config" in command
            assert command[command.index("--max-num-seqs") + 1] == (
                profile.max_num_seqs or capture_size
            )
        if profile.verbatim_launch:
            # Only the script's own deployment flags, plus the tokenizer mode and
            # parsers the concurrent chat oracle needs.
            for absent in (
                "--api-server-count",
                "--seed",
                "--block-size",
                "--no-enable-prefix-caching",
                "--enable-chunked-prefill",
                "--data-parallel-address",
                "--no-disable-hybrid-kv-cache-manager",
            ):
                assert absent not in command, absent
        else:
            assert command[command.index("--api-server-count") + 1] == "1"
            assert command[command.index("--seed") + 1] == "1024"
            assert command[command.index("--block-size") + 1] == "128"
            assert "--no-enable-prefix-caching" in command
            assert "--enable-chunked-prefill" in command
        if profile.enable_dbo:
            assert command[command.index("--dbo-decode-token-threshold") + 1] == str(
                profile.dbo_decode_token_threshold,
            )
            assert command[command.index("--dbo-prefill-token-threshold") + 1] == str(
                profile.dbo_prefill_token_threshold,
            )
            assert ("--no-enable-chunked-prefill" in command) is (
                profile.dbo_disables_chunked_prefill
            )
        if profile.quantization_from_checkpoint:
            assert command[command.index("--quantization") + 1] == "ascend"
        else:
            assert "--quantization" not in command
        assert "--kv-transfer-config" not in command
        config = json.loads(command[command.index("--additional-config") + 1])
        # Every DSV4 case pins the model-path switches, because the pinned
        # runtime defaults the multistream DSA overlap to True and that RoPE
        # path fails to tile on A5.
        assert config["enable_dsv4_shared_compressor_workspace"] is False
        assert config["multistream_dsv4_dsa_overlap"] is False
        assert config["enable_dsa_cp"] is False
        assert config["enable_cpu_binding"] is True
        # Exact equality also pins the absent keys: CAMP2P rejects both the
        # asynchronous mode and gate-on-Attention.
        expected_afd = {
            "role": role,
            "connector": "CAMP2pAFDConnector",
            "host": "192.0.2.1",
            "port": 6456,
            "num_attention_ranks": int(expected[0]) * int(expected[1]),
            "num_ffn_ranks": int(expected[2]) * int(expected[3]),
        }
        if profile.connector_extra_config is not None:
            expected_afd["connector_extra_config"] = profile.connector_extra_config
        assert config["afd"] == expected_afd
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
    capsys,
):
    args = _arguments(monkeypatch, tmp_path, scenario=scenario)
    cleanup_options = {}
    evaluations = []
    dbo_checks = []
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
        "assert_dbo_live_split_coverage",
        lambda *check_args: dbo_checks.append(check_args),
    )
    monkeypatch.setattr(
        runner,
        "terminate_processes",
        lambda _processes, **kwargs: cleanup_options.update(kwargs),
    )
    assert runner.main() == 0
    assert evaluations == ["concurrent"]
    # The DBO coverage gate runs only for a profile that enables DBO and whose
    # runtime logs a two-ubatch split line; the A5 profile runs without DBO.
    profile = DSV4_SYNC_SHAPES[scenario]
    expected_checks = int(
        profile.enable_dbo and profile.dbo_split_evidence_available,
    )
    assert len(dbo_checks) == expected_checks
    notice = "[dbo-coverage]" in capsys.readouterr().out
    assert notice is (profile.enable_dbo and not profile.dbo_split_evidence_available)
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


@pytest.mark.parametrize("quant_method", ["fp8", "ascend", None])
def test_dsv4_sync_a5_never_passes_quantization(monkeypatch, tmp_path, quant_method):
    """The A5 launch script passes no `--quantization` at all."""
    model = _model_with_config(tmp_path / f"dsv4-{quant_method}", quant_method)
    args = _arguments(
        monkeypatch,
        tmp_path,
        scenario=DSV4_SYNC_CAMP2P_A5_SCENARIO,
        model=model,
    )
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role="attention")

    assert "--quantization" not in command


def test_dsv4_sync_a3_omits_quantization_for_a_declaring_checkpoint(
    monkeypatch,
    tmp_path,
):
    """vLLM rejects `--quantization ascend` against the A3 W8A8 checkpoint."""
    model = _model_with_config(tmp_path / "dsv4-fp8", "fp8")
    args = _arguments(
        monkeypatch,
        tmp_path,
        scenario=DSV4_SYNC_CAMP2P_A3_SCENARIO,
        model=model,
    )
    runner.configure_scenario(args)

    command = runner.build_vllm_command(args, role="attention")

    assert "--quantization" not in command


def test_dsv4_sync_a3_keeps_ascend_for_an_undeclared_checkpoint(
    monkeypatch,
    tmp_path,
):
    model = _model_with_config(tmp_path / "dsv4-w8a8", None)
    args = _arguments(
        monkeypatch,
        tmp_path,
        scenario=DSV4_SYNC_CAMP2P_A3_SCENARIO,
        model=model,
    )
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


def test_dsv4_sync_a3_environment_needs_no_cam_package(monkeypatch):
    monkeypatch.setenv("HCCL_IF_IP", "192.0.2.1")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth-test")
    monkeypatch.delenv("CAM_CUST_OPAPI_LIB_PATH", raising=False)
    # A caller shell that exported the async CAM recipe's buffer size must not
    # change this case, which sizes its CAMP2P domains itself.
    monkeypatch.setenv("HCCL_BUFFSIZE", "4096")
    env = entrypoint.build_environment(DSV4_SYNC_CAMP2P_A3_SCENARIO)
    assert env["GLOO_SOCKET_IFNAME"] == "eth-test"
    assert env["TP_SOCKET_IFNAME"] == "eth-test"
    assert env["AFD_FORCE_BALANCED_TOPK_IDS"] == "0"
    assert "CAM_CUST_OPAPI_LIB_PATH" not in env
    assert "HCCL_BUFFSIZE" not in env


def test_dsv4_sync_eager_override_drops_graph_capture(monkeypatch, tmp_path):
    """A host whose capture path trips a runtime op can fall back to eager."""
    args = _arguments(monkeypatch, tmp_path)
    monkeypatch.setenv("AFD_NPU_DSV4_SYNC_E2E_EAGER", "1")
    runner.configure_scenario(args)

    assert args.cuda_graph_full_decode_only is False
    command = runner.build_vllm_command(args, role="attention")

    assert "--enforce-eager" in command
    assert "--compilation-config" not in command


def test_dsv4_sync_a5_runs_without_dbo(monkeypatch, tmp_path):
    """The A5 case drops the native DBO its host script records.

    The script runs DBO at 2/12, but the DBO split path is the current suspect
    for the DSA attention operator tiling failure seen on A5, so the profile
    keeps the recorded thresholds off while that is root-caused. With DBO off no
    DBO flag reaches vLLM, and neither the coverage gate nor forced DEBUG
    logging applies.
    """
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    args = _arguments(
        monkeypatch,
        tmp_path,
        scenario=DSV4_SYNC_CAMP2P_A5_SCENARIO,
    )
    runner.configure_scenario(args)
    assert args.enable_dbo is False

    command = runner.build_vllm_command(args, role="attention")

    for absent in (
        "--enable-dbo",
        "--dbo-decode-token-threshold",
        "--dbo-prefill-token-threshold",
    ):
        assert absent not in command, absent

    env = runner.build_env("2,3", args, role="attention")

    assert "VLLM_LOGGING_LEVEL" not in env


def test_dsv4_sync_a3_has_no_dbo_to_verify(monkeypatch, tmp_path):
    args = _arguments(
        monkeypatch,
        tmp_path,
        scenario=DSV4_SYNC_CAMP2P_A3_SCENARIO,
    )
    runner.configure_scenario(args)

    assert args.enable_dbo is False
    assert runner.dbo_split_evidence_available(args) is True


def test_dsv4_sync_a5_states_the_sum_inside_a_longer_answer():
    """Only the A5 profile relaxes the answer check, and only to a stated sum."""
    assert DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A3_SCENARIO].strict_answer is True
    assert DSV4_SYNC_SHAPES[DSV4_SYNC_CAMP2P_A5_SCENARIO].strict_answer is False

    assert completions.states_expected_sum(
        "The first number is 21, the second is 7. The sum is 28.",
        28,
    )
    assert completions.states_expected_sum("The sum is 28.28", 28)
    assert not completions.states_expected_sum("999", 28)
    # An integer that merely contains the digits is not the stated sum.
    assert not completions.states_expected_sum("128", 28)


def test_dsv4_sync_relaxed_answer_keeps_the_terminal_state_check():
    """A narrated answer still has to end in a terminal state it may reach."""
    narrated = {
        "choices": [
            {"message": {"content": "The sum is 28."}, "finish_reason": "length"},
        ],
    }
    completions.validate_response(narrated, strict_answer=False)
    with pytest.raises(RuntimeError, match="did not finish normally"):
        completions.validate_response(narrated)

    tool_call = {
        "choices": [
            {"message": {"content": "The sum is 28."}, "finish_reason": "tool_calls"},
        ],
    }
    with pytest.raises(RuntimeError, match="did not finish normally"):
        completions.validate_response(tool_call, strict_answer=False)


def test_dsv4_sync_oracle_uses_the_profile_answer_policy(monkeypatch, tmp_path):
    """The runner hands each profile's own answer policy to the oracle."""
    seen: dict = {}

    def capture(**kwargs: object) -> None:
        seen.clear()
        seen.update(kwargs)

    monkeypatch.setattr(runner, "evaluate_completions", capture)

    for scenario, expected in (
        (DSV4_SYNC_CAMP2P_A5_SCENARIO, False),
        (DSV4_SYNC_CAMP2P_A3_SCENARIO, True),
    ):
        args = _arguments(monkeypatch, tmp_path, scenario=scenario)
        runner.configure_scenario(args)

        runner.run_concurrent_completion_evaluation(args)

        assert seen["strict_answer"] is expected


def test_dsv4_sync_a5_environment_follows_the_launch_script(monkeypatch):
    """The A5 profile keeps the script's buffer size, allocator, and start method."""
    monkeypatch.delenv("HCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    monkeypatch.delenv("HCCL_BUFFSIZE", raising=False)
    monkeypatch.delenv("PYTORCH_NPU_ALLOC_CONF", raising=False)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    env = entrypoint.build_environment(DSV4_SYNC_CAMP2P_A5_SCENARIO)
    assert env["HCCL_BUFFSIZE"] == "2048"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert "VLLM_WORKER_MULTIPROC_METHOD" not in env
    assert "AFD_FORCE_SPAWN_MULTIPROCESSING" not in env
    # The NIC variables stay optional on a single host.
    assert "GLOO_SOCKET_IFNAME" not in env
    assert "TP_SOCKET_IFNAME" not in env
    # A caller who pins the buffer size keeps it.
    monkeypatch.setenv("HCCL_BUFFSIZE", "4096")
    monkeypatch.setenv("HCCL_SOCKET_IFNAME", "eth-test")
    env = entrypoint.build_environment(DSV4_SYNC_CAMP2P_A5_SCENARIO)
    assert env["HCCL_BUFFSIZE"] == "4096"
    assert env["GLOO_SOCKET_IFNAME"] == "eth-test"
    assert env["TP_SOCKET_IFNAME"] == "eth-test"


def test_dsv4_sync_a5_afd_host_defaults_to_loopback(monkeypatch, tmp_path):
    """The A5 script announces 127.0.0.1 and needs no NIC variable."""
    _arguments(monkeypatch, tmp_path, scenario=DSV4_SYNC_CAMP2P_A5_SCENARIO)
    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    command = entrypoint.build_runner_command(
        DSV4_SYNC_CAMP2P_A5_SCENARIO,
        tmp_path / "responses.json",
    )
    assert command[command.index("--afd-host") + 1] == "127.0.0.1"


def test_dsv4_sync_a3_requires_the_caller_address(monkeypatch, tmp_path):
    """The A3 profile still takes the caller's advertised rendezvous address."""
    _arguments(monkeypatch, tmp_path, scenario=DSV4_SYNC_CAMP2P_A3_SCENARIO)
    monkeypatch.delenv("HCCL_IF_IP", raising=False)
    with pytest.raises(RuntimeError, match="HCCL_IF_IP"):
        entrypoint.build_runner_command(
            DSV4_SYNC_CAMP2P_A3_SCENARIO,
            tmp_path / "responses.json",
        )
