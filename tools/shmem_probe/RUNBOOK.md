# SHMEM P3/P4 probe — Linux-agent runbook

Copy-paste steps for an agent on the A5 Linux node. Goal: build + run the
preflight P3/P4 probe, report the output. Decision gate for the AFD A5 SHMEM
rewrite — see [`../../docs/npu/A5_shmem_preflight.md`](../../docs/npu/A5_shmem_preflight.md).

The probe is built **as an in-tree CANN SHMEM example** (the example CMake uses
the repo's `aclshmem_add_collective_example` macro — it is NOT standalone).

Assumed on the node:
- afd-plugin checked out, branch `a5-custom-op-research`, pulled latest
  (so `tools/shmem_probe/` exists)
- SHMEM 1.7.0 source at `/home/k00930897/shmem` (P2); it builds for Ascend950
- CANN 9.1.0, `set_env.sh` sourceable
- A5: only device 0 and 1 healthy; kill leftover vLLM procs first

---

## Step 0 — env

```bash
set -e
export AFD=$(cd <afd-plugin-root> && pwd)
export SHMEM_SRC=/home/k00930897/shmem
export SHMEM_HOME=$SHMEM_SRC/install/shmem
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pkill -f 'vllm|python.*engine' 2>/dev/null || true
npu-smi info | head -30
git -C "$AFD" pull --ff-only
```

---

## Step 1 — inspect (paste these back if the build fails)

```bash
# the example main.cpp shape + bootstrap helper
sed -n '1,120p' "$SHMEM_SRC/examples/dispatch/dispatch_classic/main.cpp"
cat "$SHMEM_SRC/examples/dispatch/dispatch_classic/dispatch_kernel.h"
ls "$SHMEM_SRC/examples/utils/"
sed -n '1,120p' "$SHMEM_SRC/examples/init/main.cpp" 2>/dev/null || true
# the closest existing UDMA example (transport we expect on A5)
ls "$SHMEM_SRC/examples/udma_demo/"; sed -n '1,120p' "$SHMEM_SRC/examples/udma_demo/main.cpp"
# how the SOC / backend flag is passed
sed -n '540,600p' "$SHMEM_SRC/scripts/build.sh"
grep -n "SOC_TYPE\|-950\|backend" "$SHMEM_SRC/scripts/build.sh" | head -40
```

VERIFY the probe sources against the real headers and fix the `VERIFY` spots in
`$AFD/tools/shmem_probe/{main.cpp,afd_probe_kernel.cpp}`:

```bash
cd "$SHMEM_HOME/include" 2>/dev/null || cd "$SHMEM_SRC/include"
grep -rn "aclshmemx_init_attr_t\|ACLSHMEM_UNIQUEID_INITIALIZER\|aclshmemx_uniqueid_t\|aclshmemx_get_uniqueid\|aclshmemx_set_attr_uniqueid_args\|ACLSHMEMX_INIT_WITH_UNIQUEID" .
grep -rn "aclshmem_malloc\|aclshmem_barrier_all\|aclshmem_free\|aclshmem_finalize\|aclshmem_putmem\|aclshmemx_signal_op\|aclshmem_signal_wait_until\|ACLSHMEM_SIGNAL_SET\|ACLSHMEM_CMP_EQ" .
grep -rn "ACLSHMEM_TRANSPORT_MTE\|ACLSHMEM_TRANSPORT_UDMA\|TRANSPORT_SDMA\|TRANSPORT_ROCE" .
grep -rn "aclshmemi_get_state\|topo_list\|p2p_device_heap_base\|rdma_device_heap_base" . ../src 2>/dev/null | head
```

Likely fix-ups: `aclshmemx_set_attr_uniqueid_args` exact name/args; whether the
transport-bit macros already exist (delete the fallback block in
`afd_probe_kernel.cpp`); `dcci_cacheline` helper name; if `main.cpp` should use
an `examples/utils` bootstrap helper instead of the file-based UID exchange.

---

## Step 2 — add the probe as an in-tree example

```bash
mkdir -p "$SHMEM_SRC/examples/afd_probe"
cp "$AFD/tools/shmem_probe/afd_probe_kernel.cpp" "$SHMEM_SRC/examples/afd_probe/"
cp "$AFD/tools/shmem_probe/main.cpp"             "$SHMEM_SRC/examples/afd_probe/"
cp "$AFD/tools/shmem_probe/example_CMakeLists.txt" "$SHMEM_SRC/examples/afd_probe/CMakeLists.txt"
# -> CMakeLists.txt is one line: aclshmem_add_collective_example(afd_probe)
#    the macro compiles afd_probe_kernel.cpp -> libafd_probe_kernel.so and
#    main.cpp -> executable `afd_probe` (linked against shmem + afd_probe_kernel).

# register it: add `afd_probe` to the first foreach(EXAMPLE ...) list in
#   $SHMEM_SRC/examples/CMakeLists.txt   (next to `dispatch`, `sdma`, `udma_demo`)
sed -i 's/^\(\s*\)dispatch$/\1dispatch\n\1afd_probe/' "$SHMEM_SRC/examples/CMakeLists.txt"
grep -n "afd_probe" "$SHMEM_SRC/examples/CMakeLists.txt"
```

---

## Step 3 — build

```bash
# after cmake has picked up examples/afd_probe once (Step 2), incremental rebuilds
# of just the probe are fast:
cd "$SHMEM_SRC/build" && make afd_probe -j
strings bin/afd_probe | grep -E 'ShmemUdmaPutSignalProbe|init_attr rc='   # confirm new binary

# first time / after editing examples/CMakeLists.txt — full example configure:
cd "$SHMEM_SRC" && bash scripts/build.sh -examples <the SOC/backend flag P2 used for 950>

find "$SHMEM_SRC" -name 'afd_probe' -type f -perm -u+x
find "$SHMEM_SRC" -name 'libafd_probe_kernel.so'
```

**Re-copy the sources into `examples/afd_probe/` after every `git pull`** — the
probe is iterated often:
```bash
cp "$AFD"/tools/shmem_probe/{afd_probe_kernel.cpp,main.cpp} "$SHMEM_SRC/examples/afd_probe/"
```

If only the probe example fails to compile but the rest is fine, iterate on the
`VERIFY` fixes from Step 1; the other examples building proves the toolchain +
macro are OK.

---

## Step 4 — run (2-rank, device 0/1)

```bash
export SHMEM_INSTALL="$SHMEM_HOME"
BIN=$(find "$SHMEM_SRC" -name afd_probe -type f -perm -u+x | head -1)

# P3 topo + P4 UDMA bisect
for m in 1 2 3; do
  echo "== P4_MODE $m =="
  PROBE_P4_MODE=$m "$AFD/tools/shmem_probe/run_probe.sh" "$BIN" 2>&1 \
    | grep -E "P4 RESULT|stream sync rc|5070|topo_list"
done

# Slice 1a — a2e computeGate==0 data pattern (attn push + flag protocol)
PROBE_TEST=a2e "$AFD/tools/shmem_probe/run_probe.sh" "$BIN" 2>&1 \
  | grep -E "a2e-g0|5070|stream sync rc"
```

P3 only: `PROBE_RUN_P4=0 run_probe.sh ...`

### Slice 1a expected

```
[pe1] [probe] pe=1 a2e-g0: role=attn ers=1 ars=1 ratio=1 recv_batch=8 hidden=512 seg=16384B
[pe1] [a2e-g0] pe=1 attn push 16384 B -> ffn 0 slot 0
[pe0] [a2e-g0] pe=0 ffn flag slot 0 ok
[pe0] [probe] pe=0 ===== a2e-g0 RESULT: pass=1 bad_k=-1 first_bad_idx=-1 =====
[pe*] [probe] pe=* a2e-g0 kernel stream sync rc=0
```

`pass=1` + `rc=0` + no 5070xx → the a2e-gate0 kernel logic is validated and
drops into `csrc/npu/a2e/op_kernel/a2e.h`. Any 5070xx → paste the fault line
(likely the `DataCacheCleanAndInvalid` in the FFN-side spin — swap it for a
plain volatile read or a different flush).

---

## Step 5 — report back

Full stdout, plus the decisive lines:

```
[pe0] [shmem-probe P3] me=0 topo_list[1]=0x??  MTE=? ROCE=? SDMA=? UDMA=?
[pe1] [shmem-probe P3] me=1 topo_list[0]=0x??  MTE=? ROCE=? SDMA=? UDMA=?
[pe*] [shmem-probe P3] me=* pe=* p2p_heap=0x?? rdma_heap=0x?? sdma_heap=0x??
[pe1] [probe] ===== P4 RESULT: pass=?  first_bad_idx=? =====
```

| result | verdict |
| --- | --- |
| `topo_list[peer]` has UDMA / ROCE / SDMA bit **and** P4 `pass=1` | ✅ GREEN — start the rewrite |
| `topo_list[peer]` MTE-only (0x01) or 0x00 | ❌ RED — SHMEM = same root cause as a2e/e2a, fall back to B2 |
| P4 `pass=0` / 507035 / hang | ❌ RED — which primitive? paste driver log |

Also report: any 507035 in `dmesg`, the `aclshmemx_init_attr` return code if it
failed, the exact `build.sh` line that worked, and every diff you had to make to
`main.cpp` / `afd_probe_kernel.cpp` / `examples/CMakeLists.txt` (so it can be
upstreamed into `tools/shmem_probe/`).
