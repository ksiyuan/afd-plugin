"""A5 HCCL p2p carrier probe for the AFD B2 route.

Validates the data-movement primitive that Route B2 (see
``docs/npu/A5_ADAPTATION.md``) is built on: blocking
``torch.distributed.send`` / ``recv`` over the plugin-owned ``afd`` HCCL
process group can move hidden states between Attention and FFN ranks on
Atlas A5 without any MoE custom/native operator (which both fail on the
AFD mixed group with 507035).

The probe runs two legs on the AFD group:

1. A2E: Attention ranks send ``hidden_states`` to their mapped FFN rank;
2. E2A: FFN ranks echo the received data back (times 2) to the Attention
   ranks they got tokens from.

Attention ranks then compare ``received == sent * 2`` end-to-end, which
proves both legs move the right bytes with no cross-rank comparison.

Mapping (round-robin, mirrors the connector's metadata destinations):
Attention rank ``r`` (>= ffn_size) maps to FFN rank ``(r - ffn_size) % ffn_size``.

Launch (from ``afd-plugin/``, healthy devices 0/1 on an A5 node)::

    ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 --nnodes=1 \\
        tools/npu_a5_p2p_probe.py --ffn-size 1 --attn-size 1 \\
        --tokens-per-attn 8 --hidden 1024
"""

from __future__ import annotations

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401  (probe runs on NPU only)

from afd_plugin.distributed import init_afd_process_group


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffn-size", type=int, default=1)
    parser.add_argument("--attn-size", type=int, default=1)
    parser.add_argument("--tokens-per-attn", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    return parser.parse_args()


def _log(msg: str) -> None:
    print(f"[rank {os.environ.get('RANK', '?')}] {msg}", flush=True)


def _make_afd_group(world_size: int, rank: int):
    """Create the plugin-owned AFD HCCL process group (same as ``camp2p.py``)."""
    host = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "29500")
    return init_afd_process_group(
        backend="hccl",
        init_method=f"tcp://{host}:{port}",
        world_size=world_size,
        rank=rank,
        group_name="afd",
        timeout=timedelta(minutes=30),
    )


def _run_a2e_leg(
    *,
    afd_pg,
    role: str,
    ffn_size: int,
    attn_size: int,
    tokens_per_attn: int,
    hidden: int,
    device: torch.device,
    rank: int,
) -> tuple[torch.Tensor | None, list[tuple[int, torch.Tensor]]]:
    """Attention -> FFN: move hidden states over the AFD group.

    Returns on FFN ranks ``(hidden_states, [(attn_rank, block), ...])`` and
    on Attention ranks ``(sent_x, [])``.
    """
    if role == "attn":
        x = torch.randn(
            tokens_per_attn,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        dst = (rank - ffn_size) % ffn_size
        _log(f"[attn] a2e send x={tuple(x.shape)} -> ffn rank {dst}")
        dist.send(x, dst=dst, group=afd_pg)
        return x, []

    # FFN rank: receive from every Attention rank mapped to it.
    blocks: list[tuple[int, torch.Tensor]] = []
    max_tokens = tokens_per_attn * attn_size // ffn_size
    buf = torch.zeros(
        max_tokens,
        hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    offset = 0
    for r in range(ffn_size, ffn_size + attn_size):
        if (r - ffn_size) % ffn_size != rank:
            continue
        block = buf[offset : offset + tokens_per_attn]
        dist.recv(block, src=r, group=afd_pg)
        blocks.append((r, block))
        offset += tokens_per_attn
    hidden_states = buf[:offset]
    _log(
        f"[ffn] a2e recv hidden_states={tuple(hidden_states.shape)} "
        f"from {[r for r, _ in blocks]}",
    )
    return hidden_states, blocks


def _run_e2a_leg(
    *,
    afd_pg,
    role: str,
    ffn_size: int,
    hidden: int,
    device: torch.device,
    rank: int,
    sent_x: torch.Tensor | None,
    ffn_blocks: list[tuple[int, torch.Tensor]],
) -> None:
    """FFN -> Attention: echo results back and verify the round trip."""
    if role == "ffn":
        for attn_rank, block in ffn_blocks:
            result = block * 2.0
            _log(
                f"[ffn] e2a send result={tuple(result.shape)} -> attn rank {attn_rank}",
            )
            dist.send(result, dst=attn_rank, group=afd_pg)
        return

    # Attention rank: receive the result for the tokens it sent.
    result = torch.zeros_like(sent_x)
    src = (rank - ffn_size) % ffn_size
    _log(f"[attn] e2a recv result <- ffn rank {src}")
    dist.recv(result, src=src, group=afd_pg)
    torch.npu.synchronize()
    expected = sent_x * 2.0
    max_err = (result - expected).abs().max().item()
    ok = torch.allclose(result, expected)
    _log(f"[attn] e2a round-trip max_abs_err={max_err} MATCH={ok}")
    if not ok:
        raise SystemExit(f"[attn] e2a round-trip MISMATCH (max_abs_err={max_err})")


def main() -> None:
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != args.ffn_size + args.attn_size:
        raise SystemExit(
            f"WORLD_SIZE {world_size} != ffn_size + attn_size "
            f"({args.ffn_size} + {args.attn_size})",
        )
    torch.npu.set_device(local_rank)
    device = torch.device("npu")

    _log(
        f"start: ffn={args.ffn_size} attn={args.attn_size} "
        f"world={world_size} rank={rank}",
    )
    _log(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")

    afd_pg = _make_afd_group(world_size, rank)
    _log(f"afd pg created (backend={dist.get_backend(afd_pg)})")

    role = "ffn" if rank < args.ffn_size else "attn"

    sent_x, ffn_blocks = _run_a2e_leg(
        afd_pg=afd_pg,
        role=role,
        ffn_size=args.ffn_size,
        attn_size=args.attn_size,
        tokens_per_attn=args.tokens_per_attn,
        hidden=args.hidden,
        device=device,
        rank=rank,
    )
    torch.npu.synchronize()

    _run_e2a_leg(
        afd_pg=afd_pg,
        role=role,
        ffn_size=args.ffn_size,
        hidden=args.hidden,
        device=device,
        rank=rank,
        sent_x=sent_x,
        ffn_blocks=ffn_blocks,
    )

    torch.npu.synchronize()
    _log("done")


if __name__ == "__main__":
    main()
