# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# A5 (Ascend 950) A2E/E2A kernel window ABI, forwarded from the environment to
# the AscendC kernel compile. Default: the A3 remoteRes-tree path (see
# csrc/npu/a2e/op_kernel/comm_args.h). AFD_A5_FLAT_WINDOW_ABI=1 restores the flat
# HcclCombinOpParam.windowsIn[] interpretation (PR #276/#295). AFD_A5_DUMP_WINDOWS=1
# prints both interpretations of the HCCL context on the first block.
set(_a2e_kernel_options --cce-auto-sync=off -Wno-deprecated-declarations -Werror)
if (DEFINED ENV{AFD_A5_FLAT_WINDOW_ABI} AND NOT "$ENV{AFD_A5_FLAT_WINDOW_ABI}" STREQUAL "0")
    list(APPEND _a2e_kernel_options -DAFD_A5_FLAT_WINDOW_ABI=1)
endif ()
if (DEFINED ENV{AFD_A5_DUMP_WINDOWS} AND NOT "$ENV{AFD_A5_DUMP_WINDOWS}" STREQUAL "0")
    list(APPEND _a2e_kernel_options -DAFD_A5_DUMP_WINDOWS=1)
endif ()

add_ops_compile_options(
        OP_NAME A2e
        OPTIONS ${_a2e_kernel_options}
)

target_sources(op_host_aclnnInner PRIVATE
        a2e_def.cpp
)

target_sources(opapi PRIVATE
        aclnn_a2e.cpp
)

target_sources(optiling PRIVATE
        a2e_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        a2e_proto.cpp)

file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_a2e.h")

install(FILES ${_GMM_Aclnn_header}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
