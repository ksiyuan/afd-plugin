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
# TCP bootstrap address, same for every PE (rank0 binds, others connect).
# Random-ish port so a stale/hung previous run does not collide.
PROBE_IPPORT="${PROBE_IPPORT:-tcp://127.0.0.1:$((20000 + RANDOM % 20000))}"

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
echo "  ipport    : $PROBE_IPPORT"
echo "  timeout   : ${PROBE_TIMEOUT}s/proc"
echo

# libafd_probe_kernel.so lives next to / one level up from the built binary
# (e.g. build/bin/afd_probe -> build/lib/); libshmem_utils.so is in $SHMEM_INSTALL/lib.
BIN_LIB_DIR="$(cd "$(dirname "$PROBE_BIN")/../lib" 2>/dev/null && pwd || true)"
BIN_DIR="$(cd "$(dirname "$PROBE_BIN")" && pwd)"
export LD_LIBRARY_PATH="${BIN_LIB_DIR}:${BIN_DIR}:$SHMEM_INSTALL/lib:${LD_LIBRARY_PATH:-}"
echo "  LD_LIBRARY_PATH prepend: ${BIN_LIB_DIR} ${BIN_DIR} $SHMEM_INSTALL/lib"
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

pids=()
for pe in $(seq 0 $((PE_SIZE - 1))); do
  log="$LOGDIR/pe$pe.log"
  ( SHMEM_PROBE_PE="$pe" \
    SHMEM_PROBE_PE_SIZE="$PE_SIZE" \
    SHMEM_PROBE_DEVICE="${DEVS[$pe]}" \
    SHMEM_PROBE_IPPORT="$PROBE_IPPORT" \
    SHMEM_PROBE_HEAP_MB="$PROBE_HEAP_MB" \
    SHMEM_PROBE_RUN_P4="$PROBE_RUN_P4" \
    SHMEM_PROBE_P4_MODE="${PROBE_P4_MODE:-2}" \
    SHMEM_PROBE_TEST="${PROBE_TEST:-p4}" \
    SHMEM_PROBE_A2E_BATCH="${PROBE_A2E_BATCH:-8}" \
    SHMEM_PROBE_A2E_HIDDEN="${PROBE_A2E_HIDDEN:-512}" \
    SHMEM_PROBE_A2E_EXPERT_RANKS="${PROBE_A2E_EXPERT_RANKS:-1}" \
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
