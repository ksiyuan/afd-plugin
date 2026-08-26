#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

ROOT_DIR=$1
SOC_VERSION=$2
CANN_HOME="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
OPP_CONFIG_DIR="${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe/config"

resolve_a5_compute_unit() {
    # CANN 9.1+ names Atlas A5 as ascend950. CANN 8.5 used the
    # transitional ascend910_95 alias for the same DAV_3510 arch.
    if [ -d "${OPP_CONFIG_DIR}/ascend950" ]; then
        echo "ascend950"
    else
        echo "ascend910_95"
    fi
}

case "$SOC_VERSION" in
  910c|ascend910_93*|ascend910_9392)
    SOC_ARG="ascend910_93"
    ;;
  950|a5|ascend950*|ascend910_95*)
    SOC_ARG="$(resolve_a5_compute_unit)"
    ;;
  *)
    echo "AFD A2E/E2A custom ACLNN ops are currently built for Ascend 910C or Ascend 950/A5; got ${SOC_VERSION}."
    exit 0
    ;;
esac

cd "${ROOT_DIR}/csrc/npu"
rm -rf build output
echo "building AFD ACLNN custom ops a2e;e2a for ${SOC_ARG}"
bash build.sh -n "a2e;e2a" -c "${SOC_ARG}"

INSTALL_PATH="${ROOT_DIR}/afd_plugin/_cann_ops_custom"
rm -rf "${INSTALL_PATH}"
mkdir -p "${INSTALL_PATH}"
./output/CANN-custom_ops*.run --install-path="${INSTALL_PATH}"
