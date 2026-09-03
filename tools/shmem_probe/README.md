# SHMEM preflight P3 / P4 probe

Decides whether the A5 SHMEM rewrite is feasible. See
[`docs/npu/A5_shmem_preflight.md`](../../docs/npu/A5_shmem_preflight.md).

- **P3** — `ShmemTopoProbe`: on a 2-rank 1A+1F group, print
  `state->topo_list[peer]` and the heap-base pointers. **Gate**: the peer entry
  must carry a non-MTE engine bit (UDMA / ROCE / SDMA). MTE-only or 0 → SHMEM
  hits the same 507035 root cause as a2e/e2a, fall back to B2.
- **P4** — `ShmemPutSignalProbe`: rank0 `aclshmem_putmem` + `aclshmemx_signal_op`
  a 16 KB payload into rank1's symmetric window; rank1
  `aclshmem_signal_wait_until` then verifies. **Gate**: `pass=1`, no 507035.

P3 + P4 both green → start the rewrite (see
[`A5_shmem_integration_map.md`](../../docs/npu/A5_shmem_integration_map.md)).

## Files

| file | what |
| --- | --- |
| `probe_kernel.cpp` | the two `__aicore__` kernels (single AIV core, no SyncAll) |
| `probe_launch.cpp` | `<<<>>>` launch wrappers (bisheng-compiled) |
| `probe_host.cpp` | host driver: ACL init → file-based UID bootstrap → `aclshmemx_init_attr` → `aclshmem_malloc` → launch → print |
| `run_probe.sh` | 2-process launcher, PE0→dev0 / PE1→dev1, wires `LD_LIBRARY_PATH` + UID file |
| `CMakeLists.txt` | **skeleton** — reconcile with the SHMEM examples' build (see below) |

## Build (A5 node)

These files were written on the Windows side against the colleague's P3 material,
**not** the real `install/shmem/include` headers. Every risky spot is marked
`VERIFY` in the source. Two ways to build:

**Option A (recommended, least guesswork)** — graft onto an existing example:

```bash
cd /home/k00930897/shmem
cp -r examples/dispatch/dispatch_classic /tmp/shmem_probe_build
cd /tmp/shmem_probe_build
# replace the example's kernel + main with the probe sources
cp <afd-plugin>/tools/shmem_probe/probe_kernel.cpp   ./
cp <afd-plugin>/tools/shmem_probe/probe_launch.cpp   ./
cp <afd-plugin>/tools/shmem_probe/probe_host.cpp     ./
# edit the example's CMakeLists.txt: swap source file names, keep its ascendc
# kernel target + link rules; target name -> shmem_probe
cmake -B build -DSOC_VERSION=ascend950 && cmake --build build
```

**Option B** — make `CMakeLists.txt` here work by filling in the ascendc kernel
target from `examples/dispatch/dispatch_classic/CMakeLists.txt`
(`add_ascendc_kernel` / bisheng flags / `--cce-aicore-arch=dav-c310`).

## Run (A5 node, 2-rank, device 0/1)

```bash
# clear leftover vLLM procs first; only device 0/1 are healthy on the A5 node
export SHMEM_INSTALL=/home/k00930897/shmem/install/shmem
<afd-plugin>/tools/shmem_probe/run_probe.sh /tmp/shmem_probe_build/build/shmem_probe
```

P3-only: `PROBE_RUN_P4=0 run_probe.sh ...`

## Reading the output

```
[pe0] [shmem-probe P3] me=0 topo_list[1]=0x08  MTE=0 ROCE=0 SDMA=0 UDMA=1   <- GREEN
[pe1] [shmem-probe P3] me=1 topo_list[0]=0x01  MTE=1 ROCE=0 SDMA=0 UDMA=0   <- RED (MTE-only)
...
[pe1] [probe] ===== P4 RESULT: pass=1 first_bad_idx=-1 =====                <- GREEN
```

Paste the full output back into `A5_shmem_preflight.md` P3/P4 sections either way.

## UID bootstrap note

`run_probe.sh` uses **file-based** UID exchange (PE0 writes the `aclshmemx_uniqueid_t`
bytes to a temp file, PE1 spin-reads). This keeps the probe standalone. In the
real connector integration the same UID is broadcast over the existing `afd`
torch process group instead — see `A5_shmem_integration_map.md` §4.
