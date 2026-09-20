# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local NPU DSV4 Flash cases over the synchronous CAMP2P boundary.

DeepSeek V4 does not fit on a single Attention or FFN die, so each host runs
its smallest workable shape: 2A2F on four dies for A5, and 4A4F on eight dies
for A3. The synchronous connector needs no CAM vendor package: the plugin's
own a2e/e2a operators carry the activations and, for the DeepSeek V4 Hash
layers, the token ids that the FFN-side gate routes with.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.conftest import run_runner
from tests.e2e.environment import devices_from_env, required_env
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_SYNC_CAMP2P_SCENARIOS,
    DSV4_SYNC_SHAPES,
)


def build_runner_command(scenario: str, output_path: Path) -> list[str]:
    shape = DSV4_SYNC_SHAPES[scenario]
    if required_env("AFD_E2E_BACKEND") != "npu":
        raise RuntimeError("DSV4 sync CAMP2P E2E requires AFD_E2E_BACKEND=npu")
    devices = devices_from_env("AFD_E2E_DEVICES", shape.device_count)
    return [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        required_env("AFD_NPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_NPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        "npu",
        "--attention-devices",
        ",".join(devices[: shape.attention_ranks]),
        "--ffn-devices",
        ",".join(devices[shape.attention_ranks :]),
        "--scenario",
        scenario,
        "--served-model-name-prefix",
        "dsv4-flash-sync",
        "--afd-host",
        required_env("HCCL_IF_IP"),
        "--api-port-base",
        os.environ.get("AFD_NPU_DSV4_SYNC_E2E_API_PORT", "19380"),
        "--afd-port",
        os.environ.get("AFD_NPU_DSV4_SYNC_E2E_AFD_PORT", "6456"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "1800"),
        "--completion-output-path",
        str(output_path),
    ]


def build_environment() -> dict[str, str]:
    """Runtime environment for the operator transport on a multi-NIC host."""
    env = os.environ.copy()
    interface = required_env("HCCL_SOCKET_IFNAME")
    required_env("HCCL_IF_IP")
    # No CAM vendor package is installed for this path, and CAMP2P sizes its
    # own AFD HCCL domains through connector_extra_config. Drop any inherited
    # HCCL_BUFFSIZE (the async CAM recipes export one) so the run does not
    # depend on the caller's shell, the same way the NPU async CAM entrypoint
    # drops it.
    env.pop("HCCL_BUFFSIZE", None)
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    env.update(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
            "HCCL_CONNECT_TIMEOUT": "1800",
            "HCCL_EXEC_TIMEOUT": "1800",
            "VLLM_RPC_TIMEOUT": "3600000",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "10",
            "AFD_FORCE_BALANCED_TOPK_IDS": "0",
            "GLOO_SOCKET_IFNAME": interface,
            "TP_SOCKET_IFNAME": interface,
        }
    )
    return env


@pytest.mark.npu
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.parametrize("scenario", DSV4_SYNC_CAMP2P_SCENARIOS)
def test_deepseek_v4_flash_sync_camp2p(scenario: str, tmp_path: Path) -> None:
    run_runner(
        build_runner_command(scenario, tmp_path / f"{scenario}.json"),
        env=build_environment(),
    )
