#!/usr/bin/env bash
# SHMEM preflight P3/P4 probe launcher — 2 PEs, device 0 and 1, on the A5 node.
# See docs/npu/A5_shmem_preflight.md and ./README.md.
#
#   ./run_probe.sh /path/to/shmem_probe            # the built probe binary
#
# Env overrides:
#   SHMEM_INSTALL   default /home/k00930897/shmem/install/shmem  (P2 artifact)
#   PROBE_DEVICES   default "0 1"
#   PROBE_HEAP_MB   default 256
#   PROBE_RUN_P4    default 1   (0 = P3 only)
set -euo pipefail

PROBE_BIN="${1:-./shmem_probe}"
SHMEM_INSTALL="${SHMEM_INSTALL:-/home/k00930897/shmem/install/shmem}"
PROBE_DEVICES="${PROBE_DEVICES:-0 1}"
PROBE_HEAP_MB="${PROBE_HEAP_MB:-256}"
PROBE_RUN_P4="${PROBE_RUN_P4:-1}"
UID_FILE="$(mktemp -u /tmp/shmem_probe_uid.XXXXXX.bin)"

read -r -a DEVS <<< "$PROBE_DEVICES"
PE_SIZE="${#DEVS[@]}"

echo "== SHMEM probe =="
echo "  bin       : $PROBE_BIN"
echo "  shmem     : $SHMEM_INSTALL"
echo "  pe_size   : $PE_SIZE   devices: ${DEVS[*]}"
echo "  uid_file  : $UID_FILE"
echo

# libshmem_utils.so companion needs this (P2 note).
export LD_LIBRARY_PATH="$SHMEM_INSTALL/lib:${LD_LIBRARY_PATH:-}"
# CANN env (adjust if your toolkit lives elsewhere).
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

rm -f "$UID_FILE" "$UID_FILE.tmp"

pids=()
for pe in $(seq 0 $((PE_SIZE - 1))); do
  SHMEM_PROBE_PE="$pe" \
  SHMEM_PROBE_PE_SIZE="$PE_SIZE" \
  SHMEM_PROBE_DEVICE="${DEVS[$pe]}" \
  SHMEM_PROBE_UID_FILE="$UID_FILE" \
  SHMEM_PROBE_HEAP_MB="$PROBE_HEAP_MB" \
  SHMEM_PROBE_RUN_P4="$PROBE_RUN_P4" \
    "$PROBE_BIN" 2>&1 | sed "s/^/[pe$pe] /" &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=$?
done

rm -f "$UID_FILE" "$UID_FILE.tmp"
echo
echo "== exit rc=$rc =="
exit "$rc"
