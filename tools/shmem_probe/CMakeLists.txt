# SHMEM preflight P3/P4 probe — build skeleton.
#
# ⚠️ This is a BEST-EFFORT skeleton written on the Windows side without the real
#    install/shmem/include headers or the SHMEM examples' build files. Before
#    first build on the A5 node, reconcile it with:
#      <shmem>/examples/dispatch/dispatch_classic/CMakeLists.txt   (kernel + <<<>>> launch)
#      <shmem>/examples/init/CMakeLists.txt                        (host init link)
#    The two robust options:
#    (A) Copy probe_kernel.cpp / probe_launch.cpp / probe_host.cpp into a copy of
#        the dispatch_classic example dir and reuse ITS CMakeLists, just renaming
#        the target and source files. Fastest, least guesswork.
#    (B) Make this file work by filling in the ascendc/bisheng kernel-compile
#        bits from the example (npu_op_library / ascendc_library / bisheng flags).
#
# Layout assumed:
#   SHMEM_HOME = <shmem>/install/shmem   (has include/ and lib/libshmem.so)
#   ASCEND_HOME_PATH from CANN set_env.sh

cmake_minimum_required(VERSION 3.16)
project(shmem_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SHMEM_HOME "$ENV{SHMEM_HOME}" CACHE PATH "SHMEM install prefix (install/shmem)")
if(NOT SHMEM_HOME)
  set(SHMEM_HOME "/home/k00930897/shmem/install/shmem")
endif()
set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit path")

message(STATUS "SHMEM_HOME  = ${SHMEM_HOME}")
message(STATUS "ASCEND_HOME = ${ASCEND_HOME}")

# ---------------------------------------------------------------------------
# Device kernels (probe_kernel.cpp + probe_launch.cpp).
# VERIFY: replace this stub with the example's ascendc kernel target. The
# dispatch_classic example compiles its *_kernel.cpp with the bisheng compiler
# and links the resulting object into the host executable. Typical helpers:
#   ascendc_library(...) / add_ascendc_kernel(...) / bisheng flags:
#     --cce-aicore-arch=dav-c310  (A5 / __NPU_ARCH__==3510)
#     --cce-aicore-only  -mllvm -cce-aicore-function-stack-size=...
# ---------------------------------------------------------------------------
# add_ascendc_kernel(shmem_probe_kernels
#   probe_kernel.cpp probe_launch.cpp
#   SOC_VERSION ascend950
#   INCLUDES ${SHMEM_HOME}/include)

# ---------------------------------------------------------------------------
# Host driver.
# ---------------------------------------------------------------------------
add_executable(shmem_probe probe_host.cpp)
target_include_directories(shmem_probe PRIVATE
  ${SHMEM_HOME}/include
  ${ASCEND_HOME}/include)
target_link_directories(shmem_probe PRIVATE
  ${SHMEM_HOME}/lib
  ${ASCEND_HOME}/lib64)
target_link_libraries(shmem_probe PRIVATE
  shmem            # libshmem.so (P2 artifact)
  ascendcl)        # ACL runtime
# target_link_libraries(shmem_probe PRIVATE shmem_probe_kernels)

# Runtime: LD_LIBRARY_PATH must include ${SHMEM_HOME}/lib for libshmem_utils.so
# (see run_probe.sh).
