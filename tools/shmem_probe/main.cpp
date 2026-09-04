/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SHMEM preflight P3/P4 probe host driver — AFD A5 (Ascend 950) SHMEM rewrite.
 * See docs/npu/A5_shmem_preflight.md.
 *
 * Built as the `main.cpp` of an in-tree CANN SHMEM example
 * ($SHMEM_SRC/examples/afd_probe/), via aclshmem_add_collective_example(afd_probe).
 *
 * Bootstrap + init = exactly the stock udma_demo:
 *   test_set_attr(pe, n_pes, mem, "tcp://ip:port", uid, &attr)   (examples/utils/utils.h)
 *   attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA  <- must override; test_set_attr sets MTE
 *   aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr)
 *
 * DELIBERATELY OMITTED vs udma_demo:
 *   - aclshmemx_enable_exception_report(...)
 *   - aclshmemx_report_exception()
 *   - aclshmem_barrier_all()   (needs FFTS; the barrier_on_stream aux kernel
 *                               507015'd on A5)
 * so the SHMEM aux kernels never launch. Cross-rank ordering for P4 is carried
 * by the UDMA signal (rank1 spins on the signal slot before validating).
 *
 * Build + run: see RUNBOOK.md.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <thread>
#include <chrono>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"        // examples/utils/ — on the include path via the example macro

// P4 payload: PROBE_ELEMS int32 per PE segment; deterministic pe-independent pattern.
static constexpr uint32_t PROBE_ELEMS       = 4096;
static constexpr int      PROBE_SEG_BYTES   = (int)(PROBE_ELEMS * sizeof(int32_t)); // 16 KB
static constexpr uint64_t PROBE_SIGNAL      = 1000;

// Launch wrappers exported from libafd_probe_kernel.so (afd_probe_kernel.cpp).
extern void launch_shmem_topo_probe(uint32_t block_dim, void* stream, uint8_t* shmem_window);
extern void launch_shmem_udma_put_signal_probe(
    uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* sig_addr,
    int message_length, uint64_t signal, int mode);

namespace {

int env_int(const char* name, int dflt)
{
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : dflt;
}
const char* env_str(const char* name, const char* dflt)
{
    const char* v = std::getenv(name);
    return (v && *v) ? v : dflt;
}
int32_t pat(uint32_t i) { return (int32_t)(i * 7u + 1u); }

} // namespace

int main(int argc, char** argv)
{
    (void)argc; (void)argv;

    const int pe        = env_int("SHMEM_PROBE_PE", -1);
    const int pe_size   = env_int("SHMEM_PROBE_PE_SIZE", 2);
    const int device_id = env_int("SHMEM_PROBE_DEVICE", pe);
    const int run_p4    = env_int("SHMEM_PROBE_RUN_P4", 1);
    const int p4_mode   = env_int("SHMEM_PROBE_P4_MODE", 2);  // 1=quiet 2=+sync_vec_all 3=bare
    const char* ip_port = env_str("SHMEM_PROBE_IPPORT", "tcp://127.0.0.1:8998");
    const uint64_t local_mem_size =
        (uint64_t)env_int("SHMEM_PROBE_HEAP_MB", 256) * 1024ull * 1024ull;

    if (pe < 0 || pe >= pe_size) {
        std::fprintf(stderr, "SHMEM_PROBE_PE must be in [0,%d); got %d\n", pe_size, pe);
        return 2;
    }
    std::printf("[probe] pe=%d/%d device=%d heap=%lluMB ipport=%s\n",
        pe, pe_size, device_id, (unsigned long long)(local_mem_size >> 20), ip_port);

    // ---- ACL init -------------------------------------------------------
    if (aclInit(nullptr) != ACL_SUCCESS) { std::fprintf(stderr, "aclInit failed\n"); return 1; }
    if (aclrtSetDevice(device_id) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice(%d) failed\n", device_id); return 1;
    }
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // ---- SHMEM init: udma_demo path, engine = UDMA ----------------------
    aclshmemx_uniqueid_t default_flag_uid;
    std::memset(&default_flag_uid, 0, sizeof(default_flag_uid));

    aclshmemx_init_attr_t attr;
    std::memset(&attr, 0, sizeof(attr));
    int rc_set = test_set_attr(pe, pe_size, local_mem_size, ip_port, default_flag_uid, &attr);
    attr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;   // override test_set_attr's MTE
    std::printf("[probe] pe=%d test_set_attr rc=%d engine=UDMA, init_attr...\n", pe, rc_set);

    int rc_init = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    std::printf("[probe] pe=%d aclshmemx_init_attr rc=%d\n", pe, rc_init);
    if (rc_init != 0) {
        std::fprintf(stderr, "[probe] pe=%d init_attr FAILED rc=%d\n", pe, rc_init);
        return 1;
    }
    std::printf("[probe] pe=%d init OK  my_pe=%d n_pes=%d\n",
        pe, aclshmem_my_pe(), aclshmem_n_pes());

    // ================= P3: topo probe ==================================
    // small symmetric alloc just so the transport is fully wired before we read state
    void* p3win = aclshmem_malloc(4096);
    std::printf("[probe] pe=%d launching ShmemTopoProbe\n", pe);
    launch_shmem_topo_probe(1, stream, reinterpret_cast<uint8_t*>(p3win));
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
        std::fprintf(stderr, "[probe] pe=%d P3 stream sync FAILED\n", pe);
    aclshmem_free(p3win);

    // ================= P4: explicit-UDMA put + signal ==================
    if (run_p4) {
        const uint64_t win_bytes = (uint64_t)PROBE_SEG_BYTES * pe_size;
        void* gva = aclshmem_malloc(win_bytes);
        void* sig = aclshmem_malloc((uint64_t)pe_size * sizeof(uint64_t));
        if (!gva || !sig) { std::fprintf(stderr, "[probe] pe=%d malloc failed\n", pe); return 1; }

        // zero signal slots; write this PE's payload segment with the pattern
        std::vector<uint64_t> zeros(pe_size, 0);
        aclrtMemcpy(sig, pe_size * 8, zeros.data(), pe_size * 8, ACL_MEMCPY_HOST_TO_DEVICE);
        std::vector<int32_t> seg(PROBE_ELEMS);
        for (uint32_t i = 0; i < PROBE_ELEMS; ++i) seg[i] = pat(i);
        aclrtMemcpy(reinterpret_cast<uint8_t*>(gva) + (uint64_t)PROBE_SEG_BYTES * pe,
                    PROBE_SEG_BYTES, seg.data(), PROBE_SEG_BYTES, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtSynchronizeStream(stream);

        std::printf("[probe] pe=%d launching ShmemUdmaPutSignalProbe (seg=%dB signal=%llu mode=%d)\n",
            pe, PROBE_SEG_BYTES, (unsigned long long)PROBE_SIGNAL, p4_mode);
        launch_shmem_udma_put_signal_probe(1, stream,
            reinterpret_cast<uint8_t*>(gva), reinterpret_cast<uint8_t*>(sig),
            PROBE_SEG_BYTES, PROBE_SIGNAL, p4_mode);
        int rc_sync = aclrtSynchronizeStream(stream);
        std::printf("[probe] pe=%d P4 kernel stream sync rc=%d\n", pe, rc_sync);

        // validate: read every peer's segment that landed in our window, gated on
        // that peer's signal slot.
        int overall_pass = 1, first_bad = -1, bad_pe = -1;
        std::vector<int32_t> got(PROBE_ELEMS);
        for (int src = 0; src < pe_size; ++src) {
            if (src == pe) continue;
            // spin on sig[src] until == PROBE_SIGNAL (max ~30s)
            uint64_t sv = 0;
            int tries = 0;
            for (; tries < 30000; ++tries) {
                aclrtMemcpy(&sv, 8, reinterpret_cast<uint8_t*>(sig) + (uint64_t)src * 8, 8,
                            ACL_MEMCPY_DEVICE_TO_HOST);
                if (sv == PROBE_SIGNAL) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::printf("[probe] pe=%d sig[%d]=%llu after %d tries\n",
                pe, src, (unsigned long long)sv, tries);
            if (sv != PROBE_SIGNAL) { overall_pass = 0; bad_pe = src; first_bad = -2; continue; }

            aclrtMemcpy(got.data(), PROBE_SEG_BYTES,
                        reinterpret_cast<uint8_t*>(gva) + (uint64_t)PROBE_SEG_BYTES * src,
                        PROBE_SEG_BYTES, ACL_MEMCPY_DEVICE_TO_HOST);
            for (uint32_t i = 0; i < PROBE_ELEMS; ++i) {
                if (got[i] != pat(i)) { overall_pass = 0; first_bad = (int)i; bad_pe = src; break; }
            }
        }
        std::printf("[probe] pe=%d ===== P4 RESULT: pass=%d bad_pe=%d first_bad_idx=%d =====\n",
            pe, overall_pass, bad_pe, first_bad);

        aclshmem_free(gva);
        aclshmem_free(sig);
    }

    // ---- teardown -----------------------------------------------------
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    std::printf("[probe] pe=%d done\n", pe);
    return 0;
}
