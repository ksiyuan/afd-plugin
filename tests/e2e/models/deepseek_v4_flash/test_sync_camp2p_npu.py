# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Local NPU DSV4 Flash cases over the synchronous CAMP2P boundary.

DeepSeek V4 does not fit on a single Attention or FFN die, so each host runs
its own recorded launch profile: the A5 case runs Attention DP2/TP1 and FFN
DP2/TP1 with expert parallelism and graph capture on four dies, without the
native DBO its launch script records, while the A3 case shards by tensor
parallel on eight. The synchronous connector needs no CAM vendor package: the
plugin's own a2e/e2a operators carry the activations and, for the DeepSeek V4
Hash layers, the token ids that the FFN-side gate routes with.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.conftest import run_runner
from tests.e2e.models.deepseek_v4_flash.config import (
    DSV4_SYNC_CAMP2P_SCENARIOS,
    DSV4_SYNC_HCCL_BUFFER_SIZE_MB,
    DSV4_SYNC_LOCAL_AFD_HOST,
    DSV4_SYNC_SHAPES,
    DSV4SyncShape,
)


def device_indices(shape: DSV4SyncShape) -> list[str]:
    """Return the device list to split between the two roles.

    Each profile carries the mapping its host's launch script records, so a case
    run needs no device list. `AFD_E2E_DEVICES` overrides it and must still match
    the profile's die count, because the list is what selects the host: a list
    sized for the other host fails instead of skipping.
    """
    override = os.environ.get("AFD_E2E_DEVICES")
    devices = (
        [item.strip() for item in override.split(",") if item.strip()]
        if override
        else list(shape.devices)
    )
    if len(devices) != shape.device_count:
        raise RuntimeError(
            f"AFD_E2E_DEVICES must contain exactly {shape.device_count} devices",
        )
    if len(devices) != len(set(devices)):
        raise RuntimeError("AFD_E2E_DEVICES devices must be unique")
    return devices


def model_path(shape: DSV4SyncShape) -> str:
    """Return the weights path, preferring the caller's over the recorded one."""
    model = os.environ.get("AFD_NPU_E2E_MODEL") or shape.model
    if not model:
        raise RuntimeError(
            "AFD_NPU_E2E_MODEL must be set for this host: its profile records no "
            "weights path",
        )
    return model


def build_runner_command(scenario: str, output_path: Path) -> list[str]:
    shape = DSV4_SYNC_SHAPES[scenario]
    backend = os.environ.get("AFD_E2E_BACKEND")
    if backend and backend != "npu":
        raise RuntimeError("DSV4 sync CAMP2P E2E runs on NPU only")
    devices = device_indices(shape)
    return [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        model_path(shape),
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
        os.environ.get("HCCL_IF_IP") or DSV4_SYNC_LOCAL_AFD_HOST,
        "--api-port-base",
        os.environ.get("AFD_NPU_DSV4_SYNC_E2E_API_PORT", "19380"),
        "--afd-port",
        os.environ.get("AFD_NPU_DSV4_SYNC_E2E_AFD_PORT", "6456"),
        "--startup-timeout",
        os.environ.get("AFD_NPU_E2E_STARTUP_TIMEOUT", "1800"),
        "--completion-output-path",
        str(output_path),
    ]


def build_environment(scenario: str) -> dict[str, str]:
    """Runtime environment for the operator transport.

    Every entry follows the selected host profile's recorded launch script. The
    A3 profile drops any inherited HCCL_BUFFSIZE, because CAMP2P sizes its own
    AFD HCCL domains there. The A5 profile keeps the script's global buffer size,
    its plain allocator setting, and the platform's default multiprocessing start
    method. Both roles run on one host, so the NIC variables are optional and a
    supplied interface is forwarded to Gloo/TP. No CAM vendor package is
    involved on either host, so no CAM vendor variable is installed.
    """
    shape = DSV4_SYNC_SHAPES[scenario]
    environment = shape.environment
    env = os.environ.copy()
    interface = os.environ.get("HCCL_SOCKET_IFNAME", "")
    if environment.keep_hccl_buffsize:
        env.setdefault("HCCL_BUFFSIZE", str(DSV4_SYNC_HCCL_BUFFER_SIZE_MB))
    else:
        env.pop("HCCL_BUFFSIZE", None)
    if environment.force_spawn:
        env.update(
            {
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "AFD_FORCE_SPAWN_MULTIPROCESSING": "1",
            }
        )
    else:
        env.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
        env.pop("AFD_FORCE_SPAWN_MULTIPROCESSING", None)
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", environment.npu_alloc_conf)
    env.update(
        {
            "HCCL_CONNECT_TIMEOUT": "1800",
            "HCCL_EXEC_TIMEOUT": "1800",
            "VLLM_RPC_TIMEOUT": "3600000",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "30000",
            "OMP_PROC_BIND": "false",
            "OMP_NUM_THREADS": "10",
            "AFD_FORCE_BALANCED_TOPK_IDS": "0",
        }
    )
    if interface:
        env.update(
            {
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
        env=build_environment(scenario),
    )
