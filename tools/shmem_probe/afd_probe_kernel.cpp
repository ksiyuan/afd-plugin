/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SHMEM preflight P3/P4 probe — kernel library for the CANN SHMEM example tree.
 * See afd-plugin/docs/npu/A5_shmem_preflight.md.
 *
 * Built by aclshmem_add_collective_example(afd_probe): this ONE file becomes
 * libafd_probe_kernel.so. It holds both __aicore__ kernels AND the host-callable
 * <<<>>> launch wrappers (main.cpp links against this .so and calls the wrappers).
 *
 * Kernels: single AIV core (block 0), no SyncAll, no FFTS, no tiling.
 *   ShmemTopoProbe      — P3: print state->topo_list[pe] + heap-base pointers.
 *   ShmemPutSignalProbe — P4: rank0 -> rank1 put + signal round-trip + verify.
 *
 * VERIFY markers = check against install/shmem/include (SHMEM 1.7.0) before build.
 */
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

// topo_list transport bits (colleague-confirmed; if the real header already
// #defines ACLSHMEM_TRANSPORT_*, this #ifndef is a no-op).
#ifndef ACLSHMEM_TRANSPORT_MTE
#define ACLSHMEM_TRANSPORT_MTE  (1u << 0)   // red : AI core addresses peer heap -> 507035 on A5
#define ACLSHMEM_TRANSPORT_ROCE (1u << 1)   // green (2nd choice)
#define ACLSHMEM_TRANSPORT_SDMA (1u << 2)   // green
#define ACLSHMEM_TRANSPORT_UDMA (1u << 3)   // green (most likely on A5 single node)
#endif

constexpr uint32_t PROBE_PAYLOAD_ELEMS = 4096;                  // int32 elements (16 KB)
constexpr uint32_t PROBE_PAYLOAD_BYTES = PROBE_PAYLOAD_ELEMS * 4;
constexpr int32_t  PROBE_SIGNAL_VALUE  = 0x5A5A;

// ===========================================================================
// P3 — engine-selection probe
// ===========================================================================
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemTopoProbe(
    GM_ADDR shmem_window)
{
    (void)shmem_window;
    if (GetBlockIdx() != 0) {
        return;
    }

    // VERIFY: aclshmemi_get_state() -> __gm__ aclshmem_device_host_state_t*
    // (src/device/shmemi_device_common.hpp, pulled in by <shmem.h>).
    __gm__ aclshmem_device_host_state_t* st = aclshmemi_get_state();
    const int me = aclshmem_my_pe();
    const int npes = aclshmem_n_pes();

    AscendC::printf("[shmem-probe P3] me=%d npes=%d UDMA_SUPPORTED=%d\n",
        me, npes, (int)ACLSHMEM_UDMA_SUPPORTED);

    for (int pe = 0; pe < npes && pe < 16; ++pe) {
        const unsigned t = (unsigned)st->topo_list[pe];
        AscendC::printf(
            "[shmem-probe P3] me=%d topo_list[%d]=0x%02x MTE=%d ROCE=%d SDMA=%d UDMA=%d\n",
            me, pe, t,
            (int)((t & ACLSHMEM_TRANSPORT_MTE) != 0),
            (int)((t & ACLSHMEM_TRANSPORT_ROCE) != 0),
            (int)((t & ACLSHMEM_TRANSPORT_SDMA) != 0),
            (int)((t & ACLSHMEM_TRANSPORT_UDMA) != 0));
    }

    // heap-base arrays are plain `void**` in aclshmem_device_host_state_t
    // (no address-space qualifier) -> use plain `void*` here.
    // AscendC::printf is limited — stick to %d / %u / %llx like the a2e dump.
    for (int pe = 0; pe < npes && pe < 16; ++pe) {
        unsigned long long p2p  = st->p2p_device_heap_base  ? (unsigned long long)st->p2p_device_heap_base[pe]  : 0ull;
        unsigned long long rdma = st->rdma_device_heap_base ? (unsigned long long)st->rdma_device_heap_base[pe] : 0ull;
        unsigned long long sdma = st->sdma_device_heap_base ? (unsigned long long)st->sdma_device_heap_base[pe] : 0ull;
        AscendC::printf(
            "[shmem-probe P3] me=%d pe=%d p2p_heap=%llx rdma_heap=%llx sdma_heap=%llx\n",
            me, pe, p2p, rdma, sdma);
    }
    AscendC::printf("[shmem-probe P3] me=%d heap_base=%llx heap_size=%llu\n",
        me, (unsigned long long)st->heap_base, (unsigned long long)st->heap_size);
}

// ===========================================================================
// P4 — minimal put + signal round-trip
// Symmetric window: [0, PAYLOAD_BYTES) payload | [PAYLOAD_BYTES, +64) int32 signal
// ===========================================================================
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemPutSignalProbe(
    GM_ADDR shmem_window, GM_ADDR result_out)
{
    if (GetBlockIdx() != 0) {
        return;
    }
    const int me = aclshmem_my_pe();

    __gm__ int32_t* payload = reinterpret_cast<__gm__ int32_t*>(shmem_window);
    __gm__ int32_t* sig     = reinterpret_cast<__gm__ int32_t*>(shmem_window + PROBE_PAYLOAD_BYTES);
    __gm__ int32_t* result  = reinterpret_cast<__gm__ int32_t*>(result_out); // [0]=pass [1]=first bad idx

    if (me == 0) {
        for (uint32_t i = 0; i < PROBE_PAYLOAD_ELEMS; ++i) {
            payload[i] = (int32_t)(i * 7u + 1u);
        }
        // NOTE: no explicit cacheline flush here — aclshmem_quiet() after the put
        // and the data-then-flag signal ordering are expected to cover
        // visibility. If P4 shows stale data, add the shmem cache-flush helper
        // (grep dcci / DataCacheCleanAndInvalid in src/device/).

        const int dst_pe = 1;
        // topo-aware high-level RMA: SHMEM picks the engine from topo_list[dst_pe].
        // VERIFY: aclshmem_putmem(dst_sym, src_local, nbytes, pe)
        aclshmem_putmem(payload, payload, PROBE_PAYLOAD_BYTES, dst_pe);
        aclshmem_quiet();
        // data-then-flag: signal only after the put is visible.
        // VERIFY: aclshmemx_signal_op(sig_addr, value, ACLSHMEM_SIGNAL_SET, pe)
        aclshmemx_signal_op(sig, PROBE_SIGNAL_VALUE, ACLSHMEM_SIGNAL_SET, dst_pe);
        aclshmem_quiet();
        AscendC::printf("[shmem-probe P4] me=0 put+signal to pe1 done\n");
    } else if (me == 1) {
        // VERIFY: aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, value)
        aclshmem_signal_wait_until(sig, ACLSHMEM_CMP_EQ, PROBE_SIGNAL_VALUE);

        int32_t bad_idx = -1;
        for (uint32_t i = 0; i < PROBE_PAYLOAD_ELEMS; ++i) {
            if (payload[i] != (int32_t)(i * 7u + 1u)) { bad_idx = (int32_t)i; break; }
        }
        result[0] = (bad_idx < 0) ? 1 : 0;
        result[1] = bad_idx;
        AscendC::printf("[shmem-probe P4] me=1 verify pass=%d first_bad_idx=%d\n",
            (int)result[0], (int)bad_idx);
    }
}

// ===========================================================================
// Host-callable launch wrappers (compiled into libafd_probe_kernel.so).
// main.cpp passes host-side device pointers (uint8_t*); the <<<>>> mechanism
// forwards them to the GM_ADDR kernel params, same as examples/.../dispatch_demo.
// ===========================================================================
void launch_shmem_topo_probe(uint32_t block_dim, void* stream, uint8_t* shmem_window)
{
    ShmemTopoProbe<<<block_dim, nullptr, stream>>>(shmem_window);
}

void launch_shmem_put_signal_probe(
    uint32_t block_dim, void* stream, uint8_t* shmem_window, uint8_t* result_out)
{
    ShmemPutSignalProbe<<<block_dim, nullptr, stream>>>(shmem_window, result_out);
}
