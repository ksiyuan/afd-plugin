# SHMEM P3/P4 probe — Linux-agent runbook

Copy-paste steps for an agent on the A5 Linux node. Goal: build + run the
preflight P3/P4 probe, report the output. Decision gate for the AFD A5 SHMEM
rewrite — see [`../../docs/npu/A5_shmem_preflight.md`](../../docs/npu/A5_shmem_preflight.md).

Assumed on the node:
- this repo checked out, branch `a5-custom-op-research`, pulled to latest
  (so `tools/shmem_probe/` exists)
- SHMEM 1.7.0 built at `/home/k00930897/shmem` (P2 artifact);
  install prefix `/home/k00930897/shmem/install/shmem` (`include/` + `lib/libshmem.so`)
- CANN 9.1.0, `set_env.sh` sourceable
- A5 node: only device 0 and 1 are healthy; kill leftover vLLM procs first

---

## Step 0 — env + sanity

```bash
set -e
export AFD=$(git -C "$(pwd)" rev-parse --show-toplevel)          # afd-plugin root
export SHMEM_SRC=/home/k00930897/shmem
export SHMEM_HOME=$SHMEM_SRC/install/shmem
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# kill leftovers, confirm device health
pkill -f 'vllm|python.*engine' 2>/dev/null || true
npu-smi info | head -30

test -f "$SHMEM_HOME/lib/libshmem.so" && echo "libshmem OK"
ls "$SHMEM_HOME/include/"                       # expect shmem.h, shmem_host_init.h, host/, device/, host_device/
git -C "$AFD" log --oneline -1
```

---

## Step 1 — VERIFY the probe against the real headers

The probe sources were written without these headers. Check each and fix the
`VERIFY` spots in `tools/shmem_probe/probe_host.cpp` / `probe_kernel.cpp`:

```bash
cd "$SHMEM_HOME/include"
grep -rn "aclshmemx_init_attr_t\|ACLSHMEM_UNIQUEID_INITIALIZER\|aclshmemx_uniqueid_t" .
grep -rn "aclshmemx_get_uniqueid\|aclshmemx_set_attr_uniqueid_args\|ACLSHMEMX_INIT_WITH_UNIQUEID" .
grep -rn "aclshmem_malloc\|aclshmem_barrier_all\|aclshmem_free\|aclshmem_finalize" .
grep -rn "ACLSHMEM_TRANSPORT_MTE\|ACLSHMEM_TRANSPORT_UDMA\|TRANSPORT_SDMA\|TRANSPORT_ROCE" .
grep -rn "aclshmem_putmem\|aclshmem_getmem\|aclshmemx_signal_op\|aclshmem_signal_wait_until\|ACLSHMEM_SIGNAL_SET\|ACLSHMEM_CMP_EQ" .
grep -rn "aclshmemi_get_state\|aclshmem_device_host_state_t" .
grep -rn "topo_list\|p2p_device_heap_base\|rdma_device_heap_base\|heap_base\|heap_size" host_device/
```

Fix-ups likely needed:
- **`aclshmemx_set_attr_uniqueid_args` signature / name** — align the call in
  `probe_host.cpp` (arg order `pe, pe_size, local_mem_size, &uid, &attr`).
- **`aclshmemx_init_attr_t`** — if it needs explicit field init beyond `memset 0`.
- **transport bit macros** — if the header already `#define`s
  `ACLSHMEM_TRANSPORT_*`, delete the fallback block in `probe_kernel.cpp`.
- **`dcci_cacheline`** helper name in `probe_kernel.cpp` (cache flush).

Also grab the launch/build model from an example:

```bash
ls "$SHMEM_SRC/examples/"
cat "$SHMEM_SRC/examples/dispatch/dispatch_classic/CMakeLists.txt"
sed -n '1,80p' "$SHMEM_SRC/examples/dispatch/dispatch_classic/main.cpp"
cat "$SHMEM_SRC/examples/init/main.cpp" 2>/dev/null | sed -n '1,80p'
```

---

## Step 2 — build (graft onto dispatch_classic)

```bash
rm -rf /tmp/shmem_probe_build
cp -r "$SHMEM_SRC/examples/dispatch/dispatch_classic" /tmp/shmem_probe_build
cd /tmp/shmem_probe_build

# drop in the probe sources
cp "$AFD"/tools/shmem_probe/probe_kernel.cpp  ./
cp "$AFD"/tools/shmem_probe/probe_launch.cpp  ./
cp "$AFD"/tools/shmem_probe/probe_host.cpp    ./

# Edit CMakeLists.txt (or the example's build script):
#  - kernel target sources -> probe_kernel.cpp + probe_launch.cpp
#  - host executable source -> probe_host.cpp, target name -> shmem_probe
#  - keep the example's ascendc/bisheng kernel compile rules and its
#    -I$SHMEM_HOME/include / -L$SHMEM_HOME/lib -lshmem -lascendcl link
#  - SOC / arch: ascend950  (__NPU_ARCH__==3510 / dav-c310)

cmake -B build -DSOC_VERSION=ascend950 -DSHMEM_HOME="$SHMEM_HOME"
cmake --build build -j
ls -la build/shmem_probe        # or wherever the example puts its binary
```

If the example uses a plain `build.sh` instead of top-level CMake, adapt that
instead — same three edits.

---

## Step 3 — run (2-rank, device 0/1)

```bash
export SHMEM_INSTALL="$SHMEM_HOME"
"$AFD"/tools/shmem_probe/run_probe.sh /tmp/shmem_probe_build/build/shmem_probe
```

P3 only (skip P4): prefix `PROBE_RUN_P4=0`.
Bigger heap / other devices: `PROBE_HEAP_MB=512 PROBE_DEVICES="0 1"`.

---

## Step 4 — report back

Paste the **full stdout/stderr**. The decisive lines:

```
[pe0] [shmem-probe P3] me=0 topo_list[1]=0x??  MTE=? ROCE=? SDMA=? UDMA=?
[pe1] [shmem-probe P3] me=1 topo_list[0]=0x??  MTE=? ROCE=? SDMA=? UDMA=?
[pe*] [shmem-probe P3] me=* pe=* p2p_heap=0x?? rdma_heap=0x?? sdma_heap=0x??
[pe1] [probe] ===== P4 RESULT: pass=?  first_bad_idx=? =====
```

Gate:
| result | verdict |
| --- | --- |
| `topo_list[peer]` has UDMA / ROCE / SDMA bit **and** P4 `pass=1` | ✅ GREEN — start the rewrite |
| `topo_list[peer]` MTE-only (0x01) or 0x00 | ❌ RED — SHMEM = same root cause as a2e/e2a, fall back to B2 |
| P4 `pass=0` / 507035 / hang | ❌ RED — note which primitive; report the driver log |

Also report: any 507035 in `dmesg` / driver logs, the `aclshmemx_init_attr`
return code if it failed, and the final CMakeLists.txt diff you had to make
(so the probe target can be upstreamed into `tools/shmem_probe/CMakeLists.txt`).
