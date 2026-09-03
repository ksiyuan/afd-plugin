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
| `afd_probe_kernel.cpp` | both `__aicore__` kernels + host-callable `<<<>>>` launch wrappers (one file → `libafd_probe_kernel.so`, per the `aclshmem_add_collective_example` macro) |
| `main.cpp` | host driver: ACL init → file-based UID bootstrap → `aclshmemx_init_attr` → `aclshmem_malloc` → launch → print |
| `example_CMakeLists.txt` | one-liner `aclshmem_add_collective_example(afd_probe)` → copy to `$SHMEM_SRC/examples/afd_probe/CMakeLists.txt` |
| `run_probe.sh` | 2-process launcher, PE0→dev0 / PE1→dev1, wires `LD_LIBRARY_PATH` + UID file |
| `RUNBOOK.md` | **the step-by-step** — build as an in-tree SHMEM example, run, report |

## Build + run

Written on the Windows side against the colleague's P3 material, **not** the real
`install/shmem/include` headers — every risky spot is marked `VERIFY`. The
example CMake is not standalone (uses the SHMEM repo's `aclshmem_add_collective_example`
macro), so the probe builds **as an in-tree example**:

1. drop `afd_probe_kernel.cpp` + `main.cpp` + `CMakeLists.txt` into
   `$SHMEM_SRC/examples/afd_probe/`
2. add `afd_probe` to the `foreach(EXAMPLE ...)` list in `examples/CMakeLists.txt`
3. `bash scripts/build.sh -examples <950 flag>`
4. `run_probe.sh <built afd_probe binary>` on device 0/1

Full commands + inspection steps + fix-up list: **see [`RUNBOOK.md`](RUNBOOK.md)**.

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
