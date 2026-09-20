# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Fixed DSV4 Flash deployment and acceptance parameters.

Two transports are covered: the asynchronous CAM connector, and the
synchronous CAMP2P connector, which needs no CAM vendor package.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NamedTuple

DSV4_ASYNC_CAM_SCENARIO = "afd-dsv4-flash-async-cam-dp2tp4-ep8"
DSV4_SYNC_CAMP2P_A5_SCENARIO = "afd-dsv4-flash-sync-camp2p-2a2f"
DSV4_SYNC_CAMP2P_A3_SCENARIO = "afd-dsv4-flash-sync-camp2p-4a4f"
DSV4_SYNC_CAMP2P_SCENARIOS = (
    DSV4_SYNC_CAMP2P_A5_SCENARIO,
    DSV4_SYNC_CAMP2P_A3_SCENARIO,
)
DSV4_SCENARIOS = (DSV4_ASYNC_CAM_SCENARIO, *DSV4_SYNC_CAMP2P_SCENARIOS)
DSV4_ATTENTION_RANKS = 8
DSV4_FFN_RANKS = 8
DSV4_ATTENTION_TP_SIZE = 4


# DeepSeek V4 does not fit on one Attention or one FFN die: A5 needs at least
# 2A2F and A3 at least 4A4F. Both shapes are square, so the rank count is the
# tensor-parallel size. The expert-parallel world stays at one — an EP world
# greater than one selects MC2 on A5, whose dispatch operator does not tile
# (docs/npu/A5_BRINGUP_NOTES.md).
class DSV4SyncShape(NamedTuple):
    """Fixed synchronous CAMP2P deployment shape for one host class."""

    attention_ranks: int
    ffn_ranks: int
    tp_size: int

    @property
    def device_count(self) -> int:
        return self.attention_ranks + self.ffn_ranks


DSV4_SYNC_SHAPES = {
    DSV4_SYNC_CAMP2P_A5_SCENARIO: DSV4SyncShape(2, 2, 2),
    DSV4_SYNC_CAMP2P_A3_SCENARIO: DSV4SyncShape(4, 4, 4),
}
DSV4_SYNC_CAMP2P_CONNECTOR = "CAMP2pAFDConnector"
# CAMP2P sizes its AFD HCCL domains through this override; the A5 and A3 runs
# used 2048 MB. quant_mode stays 0, the only mode the runtime accepts today.
DSV4_SYNC_HCCL_BUFFER_SIZE_MB = 2048
DSV4_SYNC_QUANT_MODE = 0
DSV4_ASCEND_QUANTIZATION = "ascend"
# `--quantization none` means "pass no --quantization at all" and let the
# checkpoint's own quantization_config decide. The A5 FP8/W4A8 checkpoint
# needs that; the A3 int8 W8A8 one is loaded through the Ascend method.
DSV4_CHECKPOINT_QUANTIZATION = "none"
DSV4_SYNC_QUANTIZATION_ENV = "AFD_NPU_DSV4_SYNC_E2E_QUANTIZATION"
DSV4_CONCURRENT_REQUESTS = 10
DSV4_REQUEST_TIMEOUT_S = 300
DSV4_COMPLETION_MAX_TOKENS = 256
DSV4_PROMPT_FIRST_OPERAND = 12
DSV4_PROMPT_SECOND_OPERAND = 7
# Sixteen NPU workers take longer than the small cases to destroy HCCL
# resources; the observed launcher shutdown alone exceeded 20 seconds.
DSV4_PROCESS_TERMINATION_TIMEOUT_S = 60


def _declared_quant_method(model: str) -> str | None:
    """Return the quantization method the checkpoint config declares, if any."""
    try:
        config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        return None
    method = quantization_config.get("quant_method")
    return method if isinstance(method, str) and method else None


def sync_camp2p_quantization(model: str) -> str | None:
    """Resolve the `--quantization` value the synchronous case should pass.

    vLLM rejects `--quantization ascend` against a checkpoint that declares a
    different method, which is what the A5 FP8/W4A8 checkpoint does, while the
    A3 int8 W8A8 checkpoint is loaded through the Ascend method. Pass `ascend`
    unless the checkpoint declares another method; an explicit
    `AFD_NPU_DSV4_SYNC_E2E_QUANTIZATION` overrides either way, and `none`
    omits the flag entirely.
    """
    override = os.environ.get(DSV4_SYNC_QUANTIZATION_ENV)
    if override is not None:
        value = override.strip()
        if not value or value.lower() == DSV4_CHECKPOINT_QUANTIZATION:
            return None
        return value
    declared = _declared_quant_method(model)
    if declared is not None and declared != DSV4_ASCEND_QUANTIZATION:
        return None
    return DSV4_ASCEND_QUANTIZATION


def _configure_dsv4_arguments(
    args: argparse.Namespace,
    quantization: str | None,
) -> None:
    """Apply the fixed model arguments shared by every DSV4 scenario.

    `quantization` is the `--quantization` value to pass, or None to omit the
    flag so the checkpoint's own configuration decides.
    """
    if args.completion_output_path is None:
        raise ValueError("--completion-output-path is required for DSV4")
    # Keep these local cases aligned with the DSV4 prefill scripts.
    # Reject ad-hoc overrides so a case ID denotes one fixed deployment.
    if args.common_vllm_arg or args.attention_vllm_arg or args.ffn_vllm_arg:
        raise ValueError("DSV4 scenario does not accept extra vLLM arguments")
    if args.use_decode_bench_connector:
        raise ValueError("DSV4 scenario runs without a KV transfer connector")
    args.common_vllm_arg = [
        "--api-server-count",
        "1",
        "--seed",
        "1024",
        "--max-model-len",
        "1048576",
        "--max-num-batched-tokens",
        "8192",
        "--max-num-seqs",
        "16",
        "--block-size",
        "128",
        "--gpu-memory-utilization",
        "0.7",
        *(["--quantization", quantization] if quantization is not None else []),
        "--tokenizer-mode",
        "deepseek_v4",
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": True, "num_threads": 128}),
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
    ]
    args.attention_vllm_arg = [
        "--data-parallel-address",
        args.afd_host,
        "--no-disable-hybrid-kv-cache-manager",
        "--tool-call-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser",
        "deepseek_v4",
    ]


def configure_scenario(args: argparse.Namespace) -> None:
    """Configure the 16-NPU asynchronous CAM deployment."""
    args.afd_connector = "CAMAsyncAFDConnector"
    args.afd_async = True
    args.compute_gate_on_attention = True
    args.afd_connector_extra_config = [
        json.dumps(
            {
                "dynamicQuant": 1,
                "attn_ranks_per_dp": DSV4_ATTENTION_TP_SIZE,
                "async_moe_ubatching": True,
                "async_moe_num_ubatches": 2,
                "async_moe_split": "token",
            }
        )
    ]
    _configure_dsv4_arguments(args, DSV4_ASCEND_QUANTIZATION)


def configure_sync_camp2p_scenario(args: argparse.Namespace) -> None:
    """Configure a synchronous CAMP2P deployment.

    The caller selects the A5 (2A2F) or A3 (4A4F) shape through the scenario;
    the connector settings are the same for both. CAMP2P carries the Hash-layer
    token ids over the a2e ids channel, so the gate stays on FFN and no CAM
    vendor package is involved.
    """
    args.afd_connector = DSV4_SYNC_CAMP2P_CONNECTOR
    args.afd_async = False
    args.compute_gate_on_attention = False
    args.afd_connector_extra_config = [
        json.dumps(
            {
                "hccl_buffer_size": DSV4_SYNC_HCCL_BUFFER_SIZE_MB,
                "quant_mode": DSV4_SYNC_QUANT_MODE,
            },
            separators=(",", ":"),
        )
    ]
    _configure_dsv4_arguments(args, sync_camp2p_quantization(args.model))


def additional_config() -> dict[str, bool]:
    return {
        "enable_cpu_binding": True,
        "enable_force_load_balance": False,
        "enable_dsa_cp": False,
        "multistream_dsv4_dsa_overlap": False,
        "enable_dsv4_shared_compressor_workspace": False,
    }


def role_environment(role: str | None) -> dict[str, str]:
    return {"VLLM_ASCEND_ENABLE_FLASHCOMM1": "1" if role == "attention" else "0"}
