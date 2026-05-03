#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 '<command>'"
  exit 2
fi

GPU_INDEX="${GPU_INDEX:-3}"
MEM_LIMIT_MB="${MEM_LIMIT_MB:-23000}"
POLL_SEC="${POLL_SEC:-3}"
LOG_FILE="${LOG_FILE:-/tmp/qwen_infer_watchdog.log}"

CMD="$1"

echo "[watchdog] gpu=${GPU_INDEX} mem_limit_mb=${MEM_LIMIT_MB} poll_sec=${POLL_SEC}" | tee -a "${LOG_FILE}"
echo "[watchdog] cmd=${CMD}" | tee -a "${LOG_FILE}"

bash -lc "${CMD}" >> "${LOG_FILE}" 2>&1 &
CHILD_PID=$!
echo "[watchdog] child_pid=${CHILD_PID}" | tee -a "${LOG_FILE}"

stop_child() {
  local reason="$1"
  echo "[watchdog] stop child due to: ${reason}" | tee -a "${LOG_FILE}"
  if kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -TERM "${CHILD_PID}" 2>/dev/null || true
    read -r -t 2 _ < /dev/null || true
  fi
  if kill -0 "${CHILD_PID}" 2>/dev/null; then
    kill -KILL "${CHILD_PID}" 2>/dev/null || true
  fi
}

while kill -0 "${CHILD_PID}" 2>/dev/null; do
  USED_MB="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ -n "${USED_MB}" ]] && [[ "${USED_MB}" =~ ^[0-9]+$ ]]; then
    echo "[watchdog] gpu${GPU_INDEX}_mem_used_mb=${USED_MB}" >> "${LOG_FILE}"
    if (( USED_MB >= MEM_LIMIT_MB )); then
      stop_child "gpu${GPU_INDEX} memory ${USED_MB}MB >= ${MEM_LIMIT_MB}MB"
      wait "${CHILD_PID}" || true
      exit 99
    fi
  fi
  read -r -t "${POLL_SEC}" _ < /dev/null || true

done

wait "${CHILD_PID}"
RC=$?
echo "[watchdog] child exited rc=${RC}" | tee -a "${LOG_FILE}"
exit "${RC}"
