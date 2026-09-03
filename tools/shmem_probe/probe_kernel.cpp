/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SHMEM preflight P3/P4 probe kernels for the AFD A5 (Ascend 950) SHMEM rewrite.
 * See docs/npu/A5_shmem_preflight.md.
 *
 * Two kernels, both single-AIV-core (block 0 only), no SyncAll, no tiling:
 *   ShmemTopoProbe      — P3: print state->topo_list[pe] + heap-base pointers.
 *   ShmemPutSignalProbe — P4: rank0 put+signal a payload into rank1's symmetric
 *                              window; rank1 waits on the signal and verifies.
 *
 * VERIFY markers flag every spot that must be checked against the real
 * install/shmem/include headers (SHMEM 1.7.0) before first build.
 */
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// topo_list transport bit constants.
// From colleague's P3 material (confirm against host_device/shmem_common_types.h
// — the local shmem-src snapshot had this header truncated).
//   MTE  = 1<<0  red   : AI core addresses the peer heap directly -> 507035 on A5
//   ROCE = 1<<1  green (2nd choice)
//   SDMA = 1<<2  green
//   UDMA = 1<<3  green (most likely on A5 single node)
// VERIFY: the real header very likely already #defines these as
//   ACLSHMEM_TRANSPORT_MTE / _ROCE / _SDMA / _UDMA. If so, delete this block
//   and use those names.
#ifndef ACLSHMEM_TRANSPORT_MTE
#define ACLSHMEM_TRANSPORT_MTE  (1u << 0)
#define ACLSHMEM_TRANSPORT_ROCE (1u << 1)
#define ACLSHMEM_TRANSPORT_SDMA (1u << 2)
#define ACLSHMEM_TRANSPORT_UDMA (1u << 3)
#endif

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

    // VERIFY: aclshmemi_get_state() lives in src/device/shmemi_device_common.hpp
    // and is pulled in transitively by <shmem.h>. Return type is
    // __gm__ aclshmem_device_host_state_t*.
    __gm__ aclshmem_device_host_state_t* st = aclshmemi_get_state();
    const int me = aclshmem_my_pe();
    const int npes = aclshmem_n_pes();

    AscendC::printf("[shmem-probe P3] me=%d npes=%d  UDMA_SUPPORTED=%d\n",
        me, npes, (int)ACLSHMEM_UDMA_SUPPORTED);

    for (int pe = 0; pe < npes && pe < 16; ++pe) {
        const unsigned t = (unsigned)st->topo_list[pe];
        AscendC::printf(
            "[shmem-probe P3] me=%d topo_list[%d]=0x%02x  MTE=%d ROCE=%d SDMA=%d UDMA=%d\n",
            me, pe, t,
            (int)((t & ACLSHMEM_TRANSPORT_MTE) != 0),
            (int)((t & ACLSHMEM_TRANSPORT_ROCE) != 0),
            (int)((t & ACLSHMEM_TRANSPORT_SDMA) != 0),
            (int)((t & ACLSHMEM_TRANSPORT_UDMA) != 0));
    }

    // Heap-base pointers. Non-null p2p_device_heap_base[peer] while topo_list has
    // ONLY the MTE bit == the "looks usable locally, 507035 on the first remote
    // access" fingerprint from the investigation doc.
    for (int pe = 0; pe < npes && pe < 16; ++pe) {
        __gm__ void* p2p = st->p2p_device_heap_base ? st->p2p_device_heap_base[pe] : (__gm__ void*)0;
        __gm__ void* rdma = st->rdma_device_heap_base ? st->rdma_device_heap_base[pe] : (__gm__ void*)0;
        __gm__ void* sdma = st->sdma_device_heap_base ? st->sdma_device_heap_base[pe] : (__gm__ void*)0;
        AscendC::printf(
            "[shmem-probe P3] me=%d pe=%d p2p_heap=%p rdma_heap=%p sdma_heap=%p\n",
            me, pe, p2p, rdma, sdma);
    }
    AscendC::printf("[shmem-probe P3] me=%d heap_base=%p heap_size=%lu\n",
        me, st->heap_base, (unsigned long)st->heap_size);
}

// ===========================================================================
// P4 — minimal put + signal round-trip (a2e/e2a data-plane in miniature)
// ===========================================================================
// Symmetric window layout (identical on every PE):
//   [0,            PAYLOAD_BYTES)          payload region
//   [PAYLOAD_BYTES, PAYLOAD_BYTES+64)      one int32 signal slot (64B aligned)
constexpr uint32_t PROBE_PAYLOAD_ELEMS = 4096;                     // int32 elements
constexpr uint32_t PROBE_PAYLOAD_BYTES = PROBE_PAYLOAD_ELEMS * 4;
constexpr int32_t  PROBE_SIGNAL_VALUE  = 0x5A5A;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void ShmemPutSignalProbe(
    GM_ADDR shmem_window, GM_ADDR result_out)
{
    if (GetBlockIdx() != 0) {
        return;
    }

    const int me = aclshmem_my_pe();

    __gm__ int32_t* payload = reinterpret_cast<__gm__ int32_t*>(shmem_window);
    __gm__ int32_t* sig     = reinterpret_cast<__gm__ int32_t*>(shmem_window + PROBE_PAYLOAD_BYTES);
    __gm__ int32_t* result  = reinterpret_cast<__gm__ int32_t*>(result_out); // [0]=pass, [1]=first bad idx

    if (me == 0) {
        // Fill local payload with a known pattern (i*7 + 1), push it to pe1's
        // symmetric window, then raise pe1's signal.
        // VERIFY: local GM writes from the kernel are fine; if the scalar loop is
        // too slow swap for a UB stage + DataCopy. For a 16KB probe it's OK.
        for (uint32_t i = 0; i < PROBE_PAYLOAD_ELEMS; ++i) {
            payload[i] = (int32_t)(i * 7u + 1u);
        }
        AscendC::dcci_cacheline((__gm__ uint8_t*)payload);   // VERIFY helper name

        const int dst_pe = 1;
        // topo-aware high-level RMA — SHMEM picks the engine from topo_list[dst_pe].
        // VERIFY: aclshmem_putmem(dst_sym, src_local, nbytes, pe). "dst" is the
        // *symmetric* address (our own `payload` ptr doubles as the symmetric
        // handle for dst_pe).
        aclshmem_putmem(payload, payload, PROBE_PAYLOAD_BYTES, dst_pe);
        aclshmem_quiet();                                     // or aclshmemx_udma_quiet(dst_pe)

        // data-then-flag ordering: signal only after the put is visible.
        // VERIFY: aclshmemx_signal_op(sig_addr, value, ACLSHMEM_SIGNAL_SET, pe)
        aclshmemx_signal_op(sig, PROBE_SIGNAL_VALUE, ACLSHMEM_SIGNAL_SET, dst_pe);
        aclshmem_quiet();
        AscendC::printf("[shmem-probe P4] me=0 put+signal to pe1 done\n");
    } else if (me == 1) {
        // Wait for the signal, then verify the payload pe0 pushed into our window.
        // VERIFY: aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, value)
        aclshmem_signal_wait_until(sig, ACLSHMEM_CMP_EQ, PROBE_SIGNAL_VALUE);
        AscendC::dcci_cacheline((__gm__ uint8_t*)payload);

        int32_t bad_idx = -1;
        for (uint32_t i = 0; i < PROBE_PAYLOAD_ELEMS; ++i) {
            if (payload[i] != (int32_t)(i * 7u + 1u)) {
                bad_idx = (int32_t)i;
                break;
            }
        }
        result[0] = (bad_idx < 0) ? 1 : 0;
        result[1] = bad_idx;
        AscendC::dcci_cacheline((__gm__ uint8_t*)result);
        AscendC::printf("[shmem-probe P4] me=1 verify pass=%d first_bad_idx=%d\n",
            (int)result[0], (int)bad_idx);
    }
}
