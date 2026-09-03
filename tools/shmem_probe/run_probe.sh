#!/usr/bin/env bash
# SHMEM preflight P3/P4 probe launcher — 2 PEs, device 0 and 1, on the A5 node.
# See docs/npu/A5_shmem_preflight.md and ./RUNBOOK.md.
#
#   ./run_probe.sh /path/to/afd_probe            # the built probe binary
#
# Env overrides:
#   SHMEM_INSTALL   default /home/k00930897/shmem/install/shmem  (P2 artifact)
#   PROBE_DEVICES   default "0 1"
#   PROBE_HEAP_MB   default 256
#   PROBE_RUN_P4    default 1   (0 = P3 only)
#   PROBE_TIMEOUT   default 120 (seconds, per process)
#   PROBE_LOGDIR    default a fresh /tmp/shmem_probe.XXXX
set -uo pipefail

PROBE_BIN="${1:-./afd_probe}"
SHMEM_INSTALL="${SHMEM_INSTALL:-/home/k00930897/shmem/install/shmem}"
PROBE_DEVICES="${PROBE_DEVICES:-0 1}"
PROBE_HEAP_MB="${PROBE_HEAP_MB:-256}"
PROBE_RUN_P4="${PROBE_RUN_P4:-1}"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-120}"
LOGDIR="${PROBE_LOGDIR:-$(mktemp -d /tmp/shmem_probe.XXXXXX)}"
UID_FILE="$LOGDIR/uid.bin"

read -r -a DEVS <<< "$PROBE_DEVICES"
PE_SIZE="${#DEVS[@]}"

if [[ ! -x "$PROBE_BIN" ]]; then
  echo "ERROR: probe binary not found / not executable: $PROBE_BIN" >&2
  exit 2
fi

echo "== SHMEM probe =="
echo "  bin       : $PROBE_BIN"
echo "  shmem     : $SHMEM_INSTALL"
echo "  pe_size   : $PE_SIZE   devices: ${DEVS[*]}"
echo "  logdir    : $LOGDIR"
echo "  timeout   : ${PROBE_TIMEOUT}s/proc"
echo

export LD_LIBRARY_PATH="$SHMEM_INSTALL/lib:${LD_LIBRARY_PATH:-}"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

rm -f "$UID_FILE" "$UID_FILE.tmp"

pids=()
for pe in $(seq 0 $((PE_SIZE - 1))); do
  log="$LOGDIR/pe$pe.log"
  ( SHMEM_PROBE_PE="$pe" \
    SHMEM_PROBE_PE_SIZE="$PE_SIZE" \
    SHMEM_PROBE_DEVICE="${DEVS[$pe]}" \
    SHMEM_PROBE_UID_FILE="$UID_FILE" \
    SHMEM_PROBE_HEAP_MB="$PROBE_HEAP_MB" \
    SHMEM_PROBE_RUN_P4="$PROBE_RUN_P4" \
    timeout -k 5 "$PROBE_TIMEOUT" stdbuf -oL -eL "$PROBE_BIN" ) > "$log" 2>&1 &
  pids+=($!)
  echo "  pe$pe -> pid ${pids[-1]}, device ${DEVS[$pe]}, log $log"
done
echo

rc=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    ec=$?
    rc=$ec
    echo "!! pe$i exited rc=$ec (124 = timeout)"
  fi
done

echo
echo "================= pe logs ================="
for pe in $(seq 0 $((PE_SIZE - 1))); do
  echo "----- pe$pe -----"
  sed "s/^/[pe$pe] /" "$LOGDIR/pe$pe.log"
done
echo "=========================================="
echo
echo "logs kept in $LOGDIR"
echo "== exit rc=$rc =="
exit "$rc"
