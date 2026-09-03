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
 * UID bootstrap is file-based (no MPI, no torch): PE0 writes the uniqueid bytes
 * to $SHMEM_PROBE_UID_FILE, PE1 spin-reads it. run_probe.sh wires this up.
 *
 * Build + run: see RUNBOOK.md.
 *
 * VERIFY markers = check against install/shmem/include (SHMEM 1.7.0) before build.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>

#include "acl/acl.h"
#include "shmem.h"

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

// ---- file-based uniqueid exchange -----------------------------------------
void write_uid_file(const std::string& path, const aclshmemx_uniqueid_t& uid)
{
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&uid), sizeof(uid));
    f.close();
    std::rename(tmp.c_str(), path.c_str());   // atomic publish
}

bool read_uid_file(const std::string& path, aclshmemx_uniqueid_t& uid)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(reinterpret_cast<char*>(&uid), sizeof(uid));
    return f.gcount() == (std::streamsize)sizeof(uid);
}

} // namespace

int main(int argc, char** argv)
{
    (void)argc; (void)argv;

    const int pe        = env_int("SHMEM_PROBE_PE", -1);       // this process's rank
    const int pe_size   = env_int("SHMEM_PROBE_PE_SIZE", 2);
    const int device_id = env_int("SHMEM_PROBE_DEVICE", pe);   // pe0->dev0, pe1->dev1
    const std::string uid_file = env_str("SHMEM_PROBE_UID_FILE", "/tmp/shmem_probe_uid.bin");
    const int run_p4    = env_int("SHMEM_PROBE_RUN_P4", 1);
    const uint64_t local_mem_size =
        (uint64_t)env_int("SHMEM_PROBE_HEAP_MB", 256) * 1024ull * 1024ull;

    if (pe < 0 || pe >= pe_size) {
        std::fprintf(stderr, "SHMEM_PROBE_PE must be in [0,%d); got %d\n", pe_size, pe);
        return 2;
    }
    std::printf("[probe] pe=%d/%d device=%d heap=%luMB uid_file=%s\n",
        pe, pe_size, device_id, (unsigned long)(local_mem_size >> 20), uid_file.c_str());

    // ---- ACL init ---------------------------------------------------------
    if (aclInit(nullptr) != ACL_SUCCESS) { std::fprintf(stderr, "aclInit failed\n"); return 1; }
    if (aclrtSetDevice(device_id) != ACL_SUCCESS) {
        std::fprintf(stderr, "aclrtSetDevice(%d) failed\n", device_id); return 1;
    }
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    // ---- uniqueid bootstrap (file-based) --------------------------------
    // VERIFY exact names against shmem_host_init.h:
    //   aclshmemx_uniqueid_t, aclshmemx_get_uniqueid(&uid),
    //   aclshmemx_set_attr_uniqueid_args(pe, pe_size, local_mem_size, &uid, &attr),
    //   aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr)
    //
    // NOT using ACLSHMEM_UNIQUEID_INITIALIZER: its `{ 0 }` sub-initializer trips
    // -Wbraced-scalar-init, and the example CMake macro compiles main.cpp with
    // -Werror. memset is equivalent.
    //
    // FALLBACK if aclshmemx_set_attr_uniqueid_args does not exist: copy
    // test_set_attr() from examples/utils/utils.h (it takes an ip:port and calls
    // aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, ...)); pass
    // SHMEM_PROBE_IPPORT and drop the file-based exchange.
    aclshmemx_uniqueid_t uid;
    std::memset(&uid, 0, sizeof(uid));
    if (pe == 0) {
        std::remove(uid_file.c_str());
        if (aclshmemx_get_uniqueid(&uid) != 0) {
            std::fprintf(stderr, "aclshmemx_get_uniqueid failed\n"); return 1;
        }
        write_uid_file(uid_file, uid);
        std::printf("[probe] pe=0 published uniqueid\n");
    } else {
        for (int tries = 0; tries < 6000; ++tries) {   // ~60s
            if (read_uid_file(uid_file, uid)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::printf("[probe] pe=%d received uniqueid\n", pe);
    }

    aclshmemx_init_attr_t attr;
    std::memset(&attr, 0, sizeof(attr));
    aclshmemx_set_attr_uniqueid_args(pe, pe_size, local_mem_size, &uid, &attr);
    if (aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0) {
        std::fprintf(stderr, "[probe] pe=%d aclshmemx_init_attr FAILED "
            "(product_strategy may not recognise the A5 mainboard_id)\n", pe);
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
