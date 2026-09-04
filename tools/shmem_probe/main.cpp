/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SHMEM preflight P3/P4 probe host driver — AFD A5 (Ascend 950) SHMEM rewrite.
 * See docs/npu/A5_shmem_preflight.md.
 *
 * Built as the `main.cpp` of an in-tree CANN SHMEM example
 * ($SHMEM_SRC/examples/afd_probe/), via aclshmem_add_collective_example(afd_probe).
 * The <<<>>> launch wrappers live in afd_probe_kernel.cpp (-> libafd_probe_kernel.so).
 *
 * One process per PE (rank). Launch two, PE0 -> device 0, PE1 -> device 1.
 * Bootstrap = the same as the stock allgather/dispatch examples:
 *   test_set_attr(pe, n_pes, mem, "tcp://ip:port", uid, &attr)  (examples/utils/utils.h)
 *   aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr)
 * (ACLSHMEMX_INIT_WITH_UNIQUEID auto-picks a NIC/port and failed with rc=-2
 *  ACC_NEW_OBJECT_FAIL on the A5 node; the explicit tcp:// address works.)
 *
 * Build + run: see RUNBOOK.md.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "acl/acl.h"
#include "shmem.h"
#include "utils.h"        // examples/utils/ — on the include path via the example macro

// P4 kernel-side constants (keep in sync with afd_probe_kernel.cpp).
static constexpr uint32_t PROBE_PAYLOAD_BYTES = 4096u * 4u;
static constexpr uint32_t PROBE_SIGNAL_BYTES  = 64u;
static constexpr uint64_t PROBE_WINDOW_BYTES  = PROBE_PAYLOAD_BYTES + PROBE_SIGNAL_BYTES;

// Launch wrappers exported from libafd_probe_kernel.so (afd_probe_kernel.cpp).
extern void launch_shmem_topo_probe(uint32_t block_dim, void* stream, uint8_t* shmem_window);
extern void launch_shmem_put_signal_probe(
    uint32_t block_dim, void* stream, uint8_t* shmem_window, uint8_t* result_out);

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

} // namespace

int main(int argc, char** argv)
{
    (void)argc; (void)argv;

    const int pe        = env_int("SHMEM_PROBE_PE", -1);       // this process's rank
    const int pe_size   = env_int("SHMEM_PROBE_PE_SIZE", 2);
    const int device_id = env_int("SHMEM_PROBE_DEVICE", pe);   // pe0->dev0, pe1->dev1
    const int run_p4    = env_int("SHMEM_PROBE_RUN_P4", 1);
    const char* ip_port = env_str("SHMEM_PROBE_IPPORT", "tcp://127.0.0.1:8998");
    const uint64_t local_mem_size =
        (uint64_t)env_int("SHMEM_PROBE_HEAP_MB", 256) * 1024ull * 1024ull;

    if (pe < 0 || pe >= pe_size) {
        std::fprintf(stderr, "SHMEM_PROBE_PE must be in [0,%d); got %d\n", pe_size, pe);
        return 2;
    }
    std::printf("[probe] pe=%d/%d device=%d heap=%lluMB ipport=%s\n",
        pe, pe_size, device_id, (unsigned long long)(local_mem_size >> 20), ip_port);

    // ---- ACL init ---------------------------------------------------------
    if (aclInit(nullptr) != ACL_SUCCESS) { std::fprintf(stderr, "aclInit failed\n"); return 1; }
    if (aclrtSetDevice(device_id) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice(%d) failed\n", device_id); return 1;
    }
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // ---- SHMEM init (stock example bootstrap) ----------------------------
    aclshmemx_uniqueid_t default_flag_uid;
    std::memset(&default_flag_uid, 0, sizeof(default_flag_uid));

    aclshmemx_init_attr_t attr;
    std::memset(&attr, 0, sizeof(attr));
    int rc_set = test_set_attr(pe, pe_size, local_mem_size, ip_port, default_flag_uid, &attr);
    std::printf("[probe] pe=%d test_set_attr rc=%d, calling init_attr...\n", pe, rc_set);

    int rc_init = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
    std::printf("[probe] pe=%d aclshmemx_init_attr rc=%d\n", pe, rc_init);
    if (rc_init != 0) {
        std::fprintf(stderr, "[probe] pe=%d aclshmemx_init_attr FAILED rc=%d "
            "(SHMEM log: ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1)\n", pe, rc_init);
        return 1;
    }
    std::printf("[probe] pe=%d aclshmemx_init_attr OK  my_pe=%d n_pes=%d\n",
        pe, aclshmem_my_pe(), aclshmem_n_pes());

    // ---- symmetric window ----------------------------------------------
    // Must be called on every PE, same size, "in sync".
    void* win = aclshmem_malloc(PROBE_WINDOW_BYTES);
    if (!win) { std::fprintf(stderr, "[probe] pe=%d aclshmem_malloc failed\n", pe); return 1; }
    aclrtMemset(win, PROBE_WINDOW_BYTES, 0, PROBE_WINDOW_BYTES);
    aclshmem_barrier_all();

    // ---- P3: topo probe ----------------------------------------------
    std::printf("[probe] pe=%d launching ShmemTopoProbe\n", pe);
    launch_shmem_topo_probe(/*block_dim=*/1, stream, reinterpret_cast<uint8_t*>(win));
    aclrtSynchronizeStream(stream);
    aclshmem_barrier_all();

    // ---- P4: put + signal round-trip -------------------------------
    if (run_p4) {
        void* result = nullptr;
        aclrtMalloc(&result, 64, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMemset(result, 64, 0, 64);
        aclshmem_barrier_all();

        std::printf("[probe] pe=%d launching ShmemPutSignalProbe\n", pe);
        launch_shmem_put_signal_probe(
            /*block_dim=*/1, stream,
            reinterpret_cast<uint8_t*>(win), reinterpret_cast<uint8_t*>(result));
        aclrtSynchronizeStream(stream);
        aclshmem_barrier_all();

        if (pe == 1) {
            int32_t host_res[2] = {0, 0};
            aclrtMemcpy(host_res, sizeof(host_res), result, sizeof(host_res),
                        ACL_MEMCPY_DEVICE_TO_HOST);
            std::printf("[probe] ===== P4 RESULT: pass=%d first_bad_idx=%d =====\n",
                host_res[0], host_res[1]);
        }
        aclrtFree(result);
    }

    // ---- teardown ---------------------------------------------------
    aclshmem_free(win);
    aclshmem_finalize();
    aclrtDestroyStream(stream);
    aclrtResetDevice(device_id);
    aclFinalize();
    std::printf("[probe] pe=%d done\n", pe);
    return 0;
}
