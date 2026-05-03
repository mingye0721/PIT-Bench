#!/usr/bin/env bash
set -e

MODEL_PATH="${MODEL_PATH:-/data/lixy/models/InternVL2_5-2B}"
GPU_ID="${GPU_ID:-2}"
DETACH="${DETACH:-1}"
RESUME="${RESUME:-1}"
REBUILD_DATA="${REBUILD_DATA:-0}"
IGNORE_RNG_STATE="${IGNORE_RNG_STATE:-1}"

DATA_INPUT="/data/lixy/pinyin/data_construction/output_train/train_pool_noleak_200k.jsonl"
DATA_OUTPUT="/data/lixy/LLaMA-Factory/data/pinyin_vlm_noleak_fast.json"
OUTPUT_DIR="/data/lixy/pinyin/saves/internvl2_5_2b/lora/pinyin_fast"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[ERROR] MODEL_PATH not found: $MODEL_PATH"
  echo "Please download InternVL2.5-2B and set MODEL_PATH accordingly."
  exit 1
fi

MODEL_TYPE=$(python - "$MODEL_PATH" <<'PY'
import json
import os
import sys

model_path = sys.argv[1]
config_path = os.path.join(model_path, "config.json")
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(config.get("model_type", ""))
except Exception:
    print("")
PY
)

if [[ "$MODEL_TYPE" != "internvl_chat" ]]; then
  echo "[ERROR] Unsupported MODEL_PATH: $MODEL_PATH"
  echo "Expected config.model_type=internvl_chat, but got: ${MODEL_TYPE:-<empty>}"
  echo "Hint: use /data/lixy/models/InternVL2_5-2B instead of MPO-style checkpoints."
  exit 1
fi

cd /data/lixy

if [[ "$REBUILD_DATA" == "1" || ! -f "$DATA_OUTPUT" ]]; then
  python /data/lixy/pinyin/data_construction/build_vlm_sft_data_fast.py \
    --input "$DATA_INPUT" \
    --output "$DATA_OUTPUT" \
    --max-samples 120000
else
  echo "[INFO] Reuse existing dataset file: $DATA_OUTPUT"
fi

CFG=/data/lixy/LLaMA-Factory/examples/train_lora/pinyin_internvl2_5_2b_fast.yaml
TMP_CFG=/tmp/pinyin_internvl2_5_2b_fast_$$.yaml
cp "$CFG" "$TMP_CFG"
sed -i "s#^model_name_or_path:.*#model_name_or_path: $MODEL_PATH#" "$TMP_CFG"
sed -i "s#^overwrite_cache:.*#overwrite_cache: false#" "$TMP_CFG"

LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [[ "$RESUME" == "1" && -n "$LATEST_CKPT" ]]; then
  sed -i "s#^overwrite_output_dir:.*#overwrite_output_dir: false#" "$TMP_CFG"

  if [[ "$IGNORE_RNG_STATE" == "1" ]]; then
    for rng_file in "$LATEST_CKPT"/rng_state*.pth; do
      if [[ -f "$rng_file" ]]; then
        mv "$rng_file" "${rng_file}.bak_$(date +%Y%m%d_%H%M%S)"
        echo "[INFO] Ignore RNG state for resume: $rng_file"
      fi
    done
  fi

  if grep -q '^resume_from_checkpoint:' "$TMP_CFG"; then
    sed -i "s#^resume_from_checkpoint:.*#resume_from_checkpoint: $LATEST_CKPT#" "$TMP_CFG"
  else
    echo "resume_from_checkpoint: $LATEST_CKPT" >> "$TMP_CFG"
  fi
  echo "[INFO] Resume from checkpoint: $LATEST_CKPT"
else
  sed -i "s#^overwrite_output_dir:.*#overwrite_output_dir: true#" "$TMP_CFG"
  if [[ "$RESUME" == "1" ]]; then
    echo "[INFO] No checkpoint found, start a new run."
  else
    echo "[INFO] RESUME=0, start a new run."
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

cd /data/lixy/LLaMA-Factory

if [[ "$DETACH" == "1" ]]; then
  mkdir -p /data/lixy/pinyin/logs
  LOG=/data/lixy/pinyin/logs/pinyin_internvl2_5_2b_$(date +%Y%m%d_%H%M%S).log
  nohup setsid llamafactory-cli train "$TMP_CFG" > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > /data/lixy/pinyin/logs/pinyin_internvl2_5_2b.pid
  echo "[OK] detached training started"
  echo "PID=$PID"
  echo "LOG=$LOG"
else
  llamafactory-cli train "$TMP_CFG"
fi
