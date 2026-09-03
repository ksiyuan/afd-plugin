# A5 (Ascend 950) custom-op — can the 507035 remote-window blocker be salvaged?

Status: **open follow-up** · Last updated: 2026-09-03

> Read [`A5_ADAPTATION.md`](A5_ADAPTATION.md) first. That doc records the prior
> investigation (branch tags `backup/a5-debug-507035`, `backup/a5-full-record`)
> which concluded the custom-op path is **infeasible on A5 as written**, with
> driver-log proof. This doc only covers whether a *different* kernel approach
> can still recover custom-op performance, since Route B2 (HCCL p2p) throughput
> is not acceptable for production.

## What was already proven (do NOT re-run)

| Step | Result |
| --- | --- |
| Build for `ascend950` (PR #295 / `b089f3c`) | ✅ compiles; `HcclAllocComResourceByTiling ret=5` fixed by gated `SetCommEngine(3)` + flat-window ABI |
| Struct layout check (hypothesis A) | Dump code exists: `AFD_A5_DUMP_WINDOWS` in `a2e.h`/`e2a.h` (commit `22ebf96`) prints `rankId/rankDim/winSize/workSpace` + `windowsIn[]/windowsOut[]`. Prefix fields read back sane. |
| Kernel bisection (local vs remote) | ✅ **local window: magic write, 8 MB copy, flag write all OK. ANY read/write of `windowsIn[peerRank]` → 507035.** Reproduced on 2 machines, incl. clean code, both a2e (write) and e2a (read). |
| Driver log | `MTE_ERROR_T0_0=0x80000000` (MTE out-of-range), faulting insn `A2e_..._mix_aiv+0x26d4`, all 8 AIV cores. |
| Native MC2 fallback (`npu_moe_distribute_dispatch_v2` / `combine_v2`) | ✅ param validation passes; ❌ **same 507035 in the dispatch kernel** on the AFD mixed group. `tools/npu_a5_probe.py` stage 2 (pure EP) vs stage 3 (mixed) isolates this. |

**Conclusion carried forward:** on A5, the peer entries of `HcclCombinOpParam.windowsIn[]`
are *not* directly MTE-addressable from another card for this group type —
contrary to the vLLM-Ascend header comment. The 910C protocol (peer writes into
a remote IPC window, ping-pong flags) has no direct A5 equivalent, and native
MC2 ops only support all-expert (standard EP) groups.

## What is actually still open

The only way custom-op comes back is a kernel design that does **not** rely on
direct cross-card window MTE for the attention↔FFN hop. Candidates, roughly in
order of effort:

1. **Authoritative answer from the CANN / HCCL team.** Is `windowsIn[peer]`
   *supposed* to be MTE-accessible on A5 for a group that mixes ranks with and
   without expert resources? Is there a missing transport/registration step
   (the A3 side builds a `remoteRes` tree — what is the A5 equivalent, and does
   AFD's group creation skip it)? This is the highest-leverage question and
   avoids months of trial-and-error. Package the `AFD_A5_DUMP_WINDOWS` output +
   the plog `MTE_ERROR` extract when asking.

2. **Peer access via an HCCL primitive instead of raw MTE.** Check how A5's own
   symmetric-memory / `HcclCombinOpParam` consumers move data to a peer window
   (SDMA copy, `HcclXxx` device API, `AscendC` remote-copy intrinsic). If such
   a path exists, port the 4 transfer points in `a2e.h`/`e2a.h` to it. Look at
   the A5 branch of vLLM-Ascend `TokenDispatcherWithMC2` and CANN 950 MoE
   samples.

3. **Make the group a real standard-EP group.** Register attention ranks as
   zero-expert members so native `dispatch_v2`/`combine_v2` accept the group.
   Colleague tried a first cut and still hit 507035 (stage 3); worth confirming
   whether that was the zero-token-receive / capacity-overflow path
   (`tools/npu_a5_probe.py` notes both hypotheses) rather than a hard limit.

4. **Optimise Route B2 instead.** Quantify what "perf not acceptable" means
   first (see below) — the gap may be closable without custom ops: batch the
   per-peer `send/recv`, drop the debug `torch.npu.synchronize()`, overlap the
   transfer with compute, use `batch_isend_irecv`. This is the pragmatic path
   if 1–3 stall.

## B2 perf analysis (code read, 2026-09-03)

`RemoteFFNProxy.forward` (`model_executor/models/deepseek_v2.py`) runs **once per
MoE layer** — 26 for DeepSeek-V2-Lite, 58 for V3. Each call:

1. attention rank: `send_attn_output` → **blocking `dist.send`** to its FFN rank
2. `maybe_apply_dbo_yield`
3. attention rank: `recv_ffn_output` → **blocking `dist.recv`**

FFN side mirrors it: `recv_attn_output` (a **per-peer loop of blocking
`dist.recv`** when `attn_size > ffn_size`) → `torch.cat` → MoE → `send_ffn_output`
(per-peer blocking `dist.send`).

**The problem is serialisation, not HCCL bandwidth.** Per decode step the
attention rank is idle for the entire FFN compute of all 26–58 layers and vice
versa — throughput ≈ `1/(t_attn + t_ffn + 2·t_comm)` instead of
`1/max(t_attn, t_ffn)`. The 910C custom op does the same logical round trip but
via AIV-core flag polling on IPC windows (no host HCCL launch, no full device
sync, pipelineable).

### Quick wins (low risk, still need NPU to verify)

- `p2p_recv` does `torch.empty(shape)` on every call → **preallocate / cache
  recv buffers** per `(shape, dtype, peer)`.
- FFN-side per-peer recv loop → **`dist.batch_isend_irecv`** (one launch, not N).
  `vllm_ascend/eplb/eplb_updator.py` already uses this pattern.
- Blocking `send`/`recv` → **`isend`/`irecv` + explicit `.wait()`** so the send
  of layer L overlaps recv-buffer setup for L.
- Confirm `.contiguous()` in `p2p_send` is a no-op for the normal path (it
  should be — hidden_states is contiguous).

### The real lever: 2-ubatch overlap (DBO-lite)

While FFN computes ubatch A of layer L, attention computes ubatch B of layer L.
Hides comm latency and removes most of the cross-role idle. Full native DBO is a
known A5 gap, but a **B2-specific two-ubatch ping-pong in the connector** may be
much simpler than native DBO (no custom yield op, just alternate the `afd`/`afd1`
groups already created per ubatch). This is the highest-value work and needs no
external input. Design it against `attention_model_runner` / `ffn_model_runner`
ubatch handling.

### Benchmark first (needs NPU)

Route B2 is only validated for 2-rank eager. Before optimising, measure on A5:

- decode TPOT / throughput: B2 vs the 910C custom-op number (same model, same
  A/F split) — 10% gap or 2x?
- profile the split: comm bytes/step, `send`/`recv` count, sync stalls, whether
  comm serialises with FFN compute (it does — confirm the magnitude).

Harness: extend `tests/e2e/runner.py` (`--device-backend npu`, an A5 2-rank
scenario) with a decode-only timing loop, or a standalone script modelled on
`tools/npu_a5_p2p_probe.py`.

## RESULT (2026-09-03): the mixed group has NO remote resources on A5

Ran the tree-ABI default + `AFD_A5_DUMP_WINDOWS=1`, 2-rank 1A+1F. Still 507035.
Dump (both ranks):

```
a2e[rank=1] tree: localUsrRankId=1 rankSize=2 winSize=4296015872
                  localWindowsIn=403440000000 localWindowsOut=1200c0000000 remoteResNum=0
a2e[rank=1] tree: remoteRes[0].nextDevicePtr=0 windowsIn=0
a2e[rank=1] tree: remoteRes[1].nextDevicePtr=0 windowsIn=0
a2e[rank=1] flat: windowsIn[0]=403440000000  windowsOut[0]=4034c0080000
a2e[rank=1] flat: windowsIn[1]=1200c0000000  windowsOut[1]=120140080000
a2e[rank=0] tree: localUsrRankId=0 rankSize=2 winSize=4296015872
                  localWindowsIn=120b00000000 localWindowsOut=124000000000 remoteResNum=0
a2e[rank=0] tree: remoteRes[0..1].nextDevicePtr=0 windowsIn=0
a2e[rank=0] flat: windowsIn[0]=120b00000000  windowsIn[1]=124000000000
```

Conclusions:

1. **The struct is read correctly.** Prefix fields are all sane; vLLM-Ascend's
   MC2 uses the identical `HcclOpResParam` layout on A5 and works. `remoteResNum
   = 0` is a true read.
2. **The flat ABI is bogus.** On each rank `flat.windowsIn[1]` == that rank's own
   `localWindowsOut` (offset 40 aliases offset 40). It was never a peer address.
3. **Root cause:** `HcclAllocComResourceByTiling` allocates **no** remote window
   resources for the AFD **mixed attention+FFN** group on A5. `SetCommEngine(3)`
   only turned the old `ret=5 (NOT_SUPPORT)` into "runs, allocates nothing".
   Pure-EP groups *do* get resources on A5 (native MC2 dispatch works there, per
   the earlier probe) and vLLM-Ascend's `dispatch_ffn_combine` uses the exact
   same `Mc2CcTilingConfig(group, 8, "AlltoAll=...")` with **no** `SetCommEngine`
   — so the blocker is specifically the non-uniform (mixed expert / non-expert)
   group shape, not the op or the kernel.

This is **not fixable in the kernel.** The A3 protocol needs a peer window that
A5's HCCL will not hand out for this group.

### What is left for the custom-op path

| Option | Cost | Odds | Notes |
| --- | --- | --- | --- |
| **Comm-engine sweep** | ~1 h | low | `AFD_A5_COMM_ENGINE={0,1,2,4,5}` (env, no rebuild — reads `std::getenv` in tiling) + `AFD_A5_DUMP_WINDOWS=1`; look for `remoteResNum > 0`. |
| **SHMEM / symmetric-memory rework** | weeks, needs NPU | medium | vLLM-Ascend's non-`HCCL_COMM` MC2 path addresses peers via `shmem_ptr(symmetricPtr, rank)` (CANN `shmem_api.h`), not the HCCL window tree. Host side allocates a cross-rank symmetric region and passes its pointer as an op input; kernel replaces `winBaseOf` with `shmem_ptr`. Independent of HCCL resource alloc. |
| **Split into a group A5 accepts** | large | ? | Run a2e/e2a over a uniform sub-topology. Unclear it maps to AFD's A/F split. |
| **Escalate to CANN/HCCL** | blocked | — | Precise ask now: "`HcclAllocComResourceByTiling` returns no `remoteRes` for a group mixing MC2-expert and non-expert ranks on ascend950; which comm engine / tiling config allocates cross-card windows for it?" Attach this dump. |
| **Optimise B2 instead** | days, needs NPU | high | 2-ubatch overlap is the big lever. See B2 section. |

Given CANN/HCCL is currently unreachable and the SHMEM rework is large, the
pragmatic path is **B2 optimisation**, with the comm-engine sweep as a cheap
side-bet first.

---

## (superseded) Earlier hypothesis: the flat window ABI is wrong, A5 uses the tree

Offsets of the first three fields (`rankId`/`rankDim`/`winSize`) are byte-identical
between the flat `HcclCombinOpParam` and the tree `HcclOpResParam`, so the
`AFD_A5_DUMP_WINDOWS` "prefix looks sane" check never distinguished them. But:

- `flat.windowsIn[0]` (offset 32) aliases `tree.localWindowsIn` → a valid local
  window → "local access works".
- `flat.windowsIn[1]` (offset 40) aliases `tree.localWindowsOut` (documented
  "全F为无效值" / all-F = invalid in some modes).
- `flat.windowsIn[>=2]` aliases bytes of `tree.hcomId[128]` → garbage pointer →
  **507035 on any remote access**.

vLLM-Ascend's own MC2 op `dispatch_ffn_combine` (`csrc/mc2/`, supports arch35 /
A5) reads peer windows from `HcclOpResParamCustom` + `remoteRes[i].nextDevicePtr
->windowsIn` — the tree — on every arch. No flat array anywhere.

**Nobody has tried the tree ABI *with* PR #295's working host config**
(`SetCommEngine(3)` + `AddConfig("ascend950")` + `ascend950` build). PR #276 =
flat ABI + `SetCommEngine(MTE)` → `ret=5`. PR #295 = flat ABI + `SetCommEngine(3)`
→ `ret=5` fixed but 507035. Tree ABI + `SetCommEngine(3)` = the untested cell.

## The change on this branch (`a5-custom-op-research`, 2026-09-03)

`a2e.h` / `e2a.h` / `comm_args.h` now **default A5 to the A3 remoteRes-tree
window path** while keeping PR #295's host-side tiling/build. Two env toggles are
forwarded to the AscendC kernel compile via `*/op_host/CMakeLists.txt`:

| Env at build time | Effect |
| --- | --- |
| *(none)* | A5 uses the tree path (`remoteRes[peer].nextDevicePtr->windowsIn`), same as 910C. **Default — this is the test.** |
| `AFD_A5_FLAT_WINDOW_ABI=1` | restores the old flat `HcclCombinOpParam.windowsIn[]` interpretation (PR #295 behaviour). |
| `AFD_A5_DUMP_WINDOWS=1` | on block 0, `printf` both interpretations: tree `localWindowsIn/Out`, `remoteRes[i].nextDevicePtr` + `windowsIn`, and flat `windowsIn[i]/windowsOut[i]`. |

910C path is untouched (`AFD_ARCH_A5` undefined → every toggle is inert).

## Concrete next actions (A5 node)

1. **Just try the new default.** Rebuild the ops for A5 and run the 2-rank eager
   1A+1F DeepSeek-V2-Lite smoke:
   ```bash
   AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=ascend950_9391 \
     python -m pip install -v --no-build-isolation --no-deps -e .
   # then the CAMP2p 2-rank smoke (tests/e2e/runner.py --device-backend npu ...)
   ```
   - Completes → the flat ABI was the bug. Custom-op path is back; move to graph
     capture / DBO / multi-rank.
   - `HcclAllocComResourceByTiling ret=5` returns → `SetCommEngine(3)` and the
     flat ABI are coupled; go to step 3.
   - Still 507035 → go to step 2.
2. **Dump and read the values:**
   ```bash
   AFD_A5_DUMP_WINDOWS=1 AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=ascend950_9391 \
     python -m pip install -v --no-build-isolation --no-deps -e .
   ```
   Run once, capture the `A5 a2e/e2a[...] tree:` / `flat:` lines. Paste them into
   this doc. `remoteRes[peer].windowsIn` being sane GM addresses while flat
   `windowsIn[peer]` is 0 / 0xFFFF... / ASCII confirms the hypothesis; if the
   tree values are *also* garbage the group's transports were never built and
   this is a group-creation problem (option 3 / escalate).
3. **If `ret=5` came back:** try the *real* `HcclA2CombineOpParam` layout from
   vLLM-Ascend `csrc/utils/inc/kernel/moe_distribute_base.h` (with `res[8328]`,
   `windowsIn[HCCL_MAX_RANK_NUM]`) in place of the hand-rolled 64-entry struct,
   still with `AFD_A5_FLAT_WINDOW_ABI=1`.
4. File the option-1 question with Huawei when reachable — attach the step-2 dump.
5. In parallel (needs no NPU decision): benchmark B2 so option 4 has numbers.

## Files / refs

- [`A5_ADAPTATION.md`](A5_ADAPTATION.md) — prior investigation, authoritative.
- Tags: `backup/a5-debug-507035` (kernel bisection commits),
  `backup/a5-full-record` (B2 + full doc).
- `tools/npu_a5_probe.py` (in `backup/a5-*` / commit `26da97a`) — stage 2 pure
  EP vs stage 3 mixed group, native `dispatch_v2`/`combine_v2`.
- `tools/npu_a5_p2p_probe.py` — B2 carrier probe (2-rank round trip).
- PR #295 (`b089f3c`) — A5 build + flat-window ABI. PR #303 — Route B2.
- A5 node: devices `0/1` healthy only; kill stale vLLM procs before re-runs.
