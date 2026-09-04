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
| `main.cpp` | host driver: ACL init → `test_set_attr` + `aclshmemx_init_attr(WITH_DEFAULT)` (stock-example TCP bootstrap) → `aclshmem_malloc` → launch → print |
| `example_CMakeLists.txt` | one-liner `aclshmem_add_collective_example(afd_probe)` → copy to `$SHMEM_SRC/examples/afd_probe/CMakeLists.txt` |
| `run_probe.sh` | 2-process launcher, PE0→dev0 / PE1→dev1, wires `LD_LIBRARY_PATH` + the `tcp://` bootstrap address |
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

## Bootstrap note

`main.cpp` uses the **stock-example TCP bootstrap**: `test_set_attr(pe, n_pes,
mem, "tcp://127.0.0.1:<port>", uid, &attr)` (from `examples/utils/utils.h`) +
`aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr)` — identical to the
`allgather` / `dispatch` examples, which init fine on the A5 node.
`ACLSHMEMX_INIT_WITH_UNIQUEID` was tried first and failed with rc=-2
(`ACC_NEW_OBJECT_FAIL`, TCP bootstrap layer) — it auto-picks a NIC/port.

`run_probe.sh` passes the same `tcp://` address to every PE (`PROBE_IPPORT`,
random port by default). In the real connector integration the bootstrap
address comes from the AFD config (`tcp://{host}:{port}`) — see
`A5_shmem_integration_map.md` §4.
