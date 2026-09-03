/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * <<<>>> launch wrappers for the SHMEM P3/P4 probe kernels.
 *
 * Split out from probe_host.cpp because kernel launch syntax must be compiled by
 * the bisheng / ascendc device compiler, while probe_host.cpp is plain host C++.
 * Model: examples/dispatch/dispatch_classic (dispatch_demo<T> wrapper +
 * ShmemDispatch_xxx<<<block_dim, nullptr, stream>>>).
 *
 * VERIFY: the FFTS control address. dispatch_classic gets it host-side (e.g.
 * rtGetC2cCtrlAddr) and threads it through as the kernel's first arg. Both probe
 * kernels here are single-core with no SyncAll / cross-core sync, so they do not
 * need FFTS. If a build error or a runtime hang says otherwise, add a
 * `uint64_t fftsAddr` first parameter to each kernel + `util_set_ffts_config`.
 */
#include <cstdint>
#include "kernel_operator.h"

// The kernel definitions (probe_kernel.cpp) are compiled into this same target.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemTopoProbe(
    uint8_t* shmem_window);
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemPutSignalProbe(
    uint8_t* shmem_window, uint8_t* result_out);

void launch_shmem_topo_probe(uint32_t block_dim, void* stream, uint8_t* shmem_window)
{
    ShmemTopoProbe<<<block_dim, nullptr, stream>>>(shmem_window);
}

void launch_shmem_put_signal_probe(
    uint32_t block_dim, void* stream, uint8_t* shmem_window, uint8_t* result_out)
{
    ShmemPutSignalProbe<<<block_dim, nullptr, stream>>>(shmem_window, result_out);
}
