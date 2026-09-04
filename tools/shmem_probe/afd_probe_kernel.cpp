/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SHMEM preflight P3/P4 probe — kernel library for the CANN SHMEM example tree.
 * See afd-plugin/docs/npu/A5_shmem_preflight.md.
 *
 * Built by aclshmem_add_collective_example(afd_probe): this ONE file becomes
 * libafd_probe_kernel.so. It holds the __aicore__ kernels AND the host-callable
 * <<<>>> launch wrappers (main.cpp links against this .so and calls the wrappers).
 *
 *   ShmemTopoProbe          — P3: print state->topo_list[pe] + heap-base pointers.
 *   ShmemUdmaPutSignalProbe — P4: explicit-UDMA put+signal, modelled 1:1 on
 *                                 examples/udma_demo/udma_demo_kernel.cpp
 *                                 (udma_put_signal_kernel). No report_exception,
 *                                 no host aclshmem_barrier_all -> the aux
 *                                 exception-report kernel (507035 on A5) never
 *                                 runs.
 */
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

// topo_list transport bits (colleague-confirmed topo layout — NOTE this differs
// from the ACLSHMEM_DATA_OP_* enum which is MTE=0x01 SDMA=0x02 ROCE=0x04
// UDMA=0x08; bits 0 and 3 agree, only SDMA/ROCE positions swap. The raw
// topo_list[pe]=0x?? hex below is the source of truth).
#ifndef ACLSHMEM_TRANSPORT_MTE
#define ACLSHMEM_TRANSPORT_MTE  (1u << 0)
#define ACLSHMEM_TRANSPORT_ROCE (1u << 1)
#define ACLSHMEM_TRANSPORT_SDMA (1u << 2)
#define ACLSHMEM_TRANSPORT_UDMA (1u << 3)
#endif

// ===========================================================================
// P3 — engine-selection probe (unchanged; single AIV core, no sync/quiet)
// ===========================================================================
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemTopoProbe(
    GM_ADDR shmem_window)
{
    (void)shmem_window;
    if (GetBlockIdx() != 0) {
        return;
    }

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

void launch_shmem_topo_probe(uint32_t block_dim, void* stream, uint8_t* shmem_window)
{
    ShmemTopoProbe<<<block_dim, nullptr, stream>>>(shmem_window);
}

// ===========================================================================
// P4 — explicit UDMA put + signal (copy of udma_demo's udma_put_signal_kernel)
// ===========================================================================
constexpr uint32_t UDMA_WQE_SCRATCH_BYTES = ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE;

__aicore__ inline void init_udma_wqe_scratch(__ubuf__ uint8_t* scratch, uint32_t bytes)
{
    __ubuf__ uint64_t* s = reinterpret_cast<__ubuf__ uint64_t*>(scratch);
    for (uint32_t i = 0; i < bytes / sizeof(uint64_t); ++i) {
        s[i] = 0U;
    }
}

// gva          : symmetric payload window (aclshmem_malloc). Each PE owns segment
//                [message_length * pe, message_length * (pe+1)).
// sig_addr     : symmetric signal window, one uint64 slot per PE.
// message_length : segment size in BYTES.
extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemUdmaPutSignalProbe(
    GM_ADDR gva, GM_ADDR sig_addr, int message_length, uint64_t signal)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buf;
    pipe.InitBuffer(buf, UDMA_WQE_SCRATCH_BYTES);
    AscendC::LocalTensor<uint8_t> ubLocal = buf.GetWithOffset<uint8_t>(UDMA_WQE_SCRATCH_BYTES, 0);
    constexpr uint32_t SYNC_ID = 0;

    __ubuf__ uint8_t* wqe_scratch = (__ubuf__ uint8_t*)ubLocal.GetPhyAddr();
    init_udma_wqe_scratch(wqe_scratch, UDMA_WQE_SCRATCH_BYTES);

    const int64_t my_pe = aclshmem_my_pe();
    const int64_t pe_size = aclshmem_n_pes();

    for (int i = 0; i < pe_size; ++i) {
        if (i == my_pe) {
            continue;
        }
        __gm__ uint64_t* dst_sig = (__gm__ uint64_t*)(sig_addr + sizeof(uint64_t) * my_pe);
        aclshmemx_udma_put_signal_nbi(
            (__gm__ uint8_t*)(gva + message_length * my_pe),
            (__gm__ uint8_t*)(gva + message_length * my_pe),
            (uint32_t)message_length, dst_sig, signal, i, wqe_scratch, SYNC_ID);
        aclshmemx_udma_quiet(i);
    }
    aclshmemx_sync_vec_all();
}

void launch_shmem_udma_put_signal_probe(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr,
    int message_length, uint64_t signal)
{
    ShmemUdmaPutSignalProbe<<<block_dim, nullptr, stream>>>(gva, sig_addr, message_length, signal);
}
