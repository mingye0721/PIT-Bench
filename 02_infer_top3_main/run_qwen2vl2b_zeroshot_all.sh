#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"
GPU_INDEX="${GPU_INDEX:-3}"
MEM_LIMIT_MB="${MEM_LIMIT_MB:-22000}"
POLL_SEC="${POLL_SEC:-2}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/data/lixy/models/Qwen2-VL-2B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-1024}"

OUT_ROOT="${OUT_ROOT:-/data/lixy/pinyin/eval/qwen2vl2b_pitbench_outputs}"
LOG_ROOT="${LOG_ROOT:-${OUT_ROOT}/logs}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

run_watchdog() {
  local name="$1"
  local cmd="$2"
  echo "[run] ${name}"
  GPU_INDEX="${GPU_INDEX}" \
  MEM_LIMIT_MB="${MEM_LIMIT_MB}" \
  POLL_SEC="${POLL_SEC}" \
  LOG_FILE="${LOG_ROOT}/${name}.log" \
  /data/lixy/pinyin/infer/run_with_gpu_watchdog.sh "${cmd}"
}

COMMON_ENV="GPU_ID=${GPU_ID} BASE_MODEL_PATH=${BASE_MODEL_PATH} USE_ADAPTER=0 MAX_NEW_TOKENS=${MAX_NEW_TOKENS} MAX_IMAGE_SIDE=${MAX_IMAGE_SIDE}"

run_watchdog \
  task1_zeroshot \
  "${COMMON_ENV} TEST_DATA_PATH=/data/lixy/pinyin/data_construction/output_task1/task1_all.jsonl OUTPUT_WITH_IMG=${OUT_ROOT}/qwen2vl2b_zeroshot_task1_with_img.json OUTPUT_NO_IMG=${OUT_ROOT}/qwen2vl2b_zeroshot_task1_no_img.json WITH_IMG_PROGRESS_JSONL=${OUT_ROOT}/qwen2vl2b_zeroshot_task1_with_img.progress.jsonl NO_IMG_PROGRESS_JSONL=${OUT_ROOT}/qwen2vl2b_zeroshot_task1_no_img.progress.jsonl RESUME=1 python /data/lixy/pinyin/infer/infer_task1.py"

run_watchdog \
  task2_zeroshot \
  "${COMMON_ENV} TEST_DATA_PATH=/data/lixy/pinyin/data_construction/output_task2/task2_all_filled.jsonl OUTPUT_PATH=${OUT_ROOT}/qwen2vl2b_zeroshot_task2_three_conditions.jsonl python /data/lixy/pinyin/infer/infer_task2.py"

run_watchdog \
  task3_zeroshot \
  "${COMMON_ENV} TEST_DATA_PATH=/data/lixy/pinyin/data_construction/output_task3/task3_all.jsonl OUTPUT_DIR=${OUT_ROOT}/task3_zeroshot python /data/lixy/pinyin/infer/infer_task3.py"

echo "[done] zeroshot all infer finished"