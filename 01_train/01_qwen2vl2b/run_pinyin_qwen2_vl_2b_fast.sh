#!/usr/bin/env bash
set -e

MODEL_PATH="${MODEL_PATH:-/data/lixy/models/Qwen2-VL-2B-Instruct}"
GPU_ID="${GPU_ID:-2}"
DETACH="${DETACH:-1}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH not found: $MODEL_PATH"
  echo "Please download Qwen2-VL-2B-Instruct and set MODEL_PATH accordingly."
  exit 1
fi

cd /data/lixy

python /data/lixy/pinyin/data_construction/build_vlm_sft_data_fast.py \
  --input /data/lixy/pinyin/data_construction/output_train/train_pool_noleak_200k.jsonl \
  --output /data/lixy/LLaMA-Factory/data/pinyin_vlm_noleak_fast.json \
  --max-samples 120000

CFG=/data/lixy/LLaMA-Factory/examples/train_lora/pinyin_qwen2_vl_2b_fast.yaml
TMP_CFG=/tmp/pinyin_qwen2_vl_2b_fast_$$.yaml
cp "$CFG" "$TMP_CFG"
sed -i "s#^model_name_or_path:.*#model_name_or_path: $MODEL_PATH#" "$TMP_CFG"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

cd /data/lixy/LLaMA-Factory

if [[ "$DETACH" == "1" ]]; then
  mkdir -p /data/lixy/pinyin/logs
  LOG=/data/lixy/pinyin/logs/pinyin_qwen2_vl_2b_$(date +%Y%m%d_%H%M%S).log
  nohup setsid llamafactory-cli train "$TMP_CFG" > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > /data/lixy/pinyin/logs/pinyin_qwen2_vl_2b.pid
  echo "[OK] detached training started"
  echo "PID=$PID"
  echo "LOG=$LOG"
else
  llamafactory-cli train "$TMP_CFG"
fi
