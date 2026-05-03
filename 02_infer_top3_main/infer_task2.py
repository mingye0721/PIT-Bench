#!/usr/bin/env python3

"""
Qwen2-VL-2B Task2 三种条件推理脚本
支持：
1. +I (匹配图像)
2. -I (无图像)
3. -I_wrong (错误图像)

输出格式保持与原 infer_task2.py 一致：JSONL
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ============================================================
# 配置
# ============================================================
BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH", "/data/lixy/models/Qwen2-VL-2B-Instruct")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "/data/lixy/pinyin/saves/qwen2_vl_2b/lora/pinyin_fast")
USE_ADAPTER = os.environ.get("USE_ADAPTER", "1") != "0"

TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "/data/lixy/pinyin/data_construction/output_task2/task2_all_filled.jsonl")
OUTPUT_PATH = os.environ.get(
    "OUTPUT_PATH", "/data/lixy/pinyin/eval/qwen2vl2b_pitbench_outputs/qwen2vl2b_task2_qwen_infer.jsonl"
)

GPU_ID = os.environ.get("GPU_ID", "2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))
RESUME = os.environ.get("RESUME", "1") != "0"


def safe_open_image(path: str) -> Optional[Image.Image]:
    try:
        with Image.open(path) as im:
            image = im.convert("RGB")
            if max(image.size) > MAX_IMAGE_SIDE:
                w, h = image.size
                scale = float(MAX_IMAGE_SIDE) / float(max(w, h))
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                image = image.resize((nw, nh), Image.Resampling.BICUBIC)
            return image
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def strip_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^assistant\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^输出[:：]\s*", "", text)
    return text.strip()


def build_prompt(pinyin: str) -> str:
    return (
        "你现在执行拼音转写任务，根据给定拼音和图像语义生成对应的汉字。\n"
        "输出格式必须为一个 JSON 对象：\n"
        '{"chars": "按拼音顺序生成的汉字序列"}\n'
        "现在请转写：\n"
        f"拼音：{(pinyin or '').strip()}"
    )


def load_model_and_processor():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print("=" * 70)
    print("Task2 三种条件推理 (Qwen2-VL-2B)")
    print("=" * 70)
    print("\n加载模型...")

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=dtype,
        device_map=None,
    )
    if USE_ADAPTER:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    print("模型加载完成\n")
    return model, processor, device


def generate_one(
    model,
    processor,
    device: str,
    prompt: str,
    image: Optional[Image.Image],
    max_new_tokens: int,
) -> str:
    content = [{"type": "text", "text": prompt}]
    images = None
    if image is not None:
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        images = [image]

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

    input_len = int(inputs["input_ids"].shape[1])
    trimmed = generated_ids[0][input_len:]
    out = processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return strip_output(out)


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_results(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def contains_target(pred: str, target: str) -> bool:
    try:
        if pred.startswith("{"):
            data = json.loads(pred)
            pred_text = data.get("chars", pred)
        else:
            pred_text = pred
        return target in pred_text
    except Exception:
        return target in pred


def main() -> None:
    torch.manual_seed(100)
    model, processor, device = load_model_and_processor()

    print(f"读取测试数据: {TEST_DATA_PATH}")
    test_data = read_jsonl(TEST_DATA_PATH)
    if not test_data:
        raise ValueError("未能加载任何测试数据")
    print(f"成功加载 {len(test_data)} 条测试样本\n")

    if not RESUME:
        Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_PATH).write_text("", encoding="utf-8")

    results = load_existing_results(OUTPUT_PATH) if RESUME else []
    completed_ids = {str(r.get("sample_id")) for r in results if r.get("sample_id") is not None}
    print(f"断点续跑: {'开启' if RESUME else '关闭'}，已完成 {len(completed_ids)} 条")
    print("开始推理...\n")

    for idx, sample in enumerate(tqdm(test_data, desc="Task2 推理")):
        sample_id = sample.get("sample_id", f"sample_{idx}")
        group_id = sample.get("group_id")
        target_word = sample.get("target_word", "")
        text_gt = sample.get("text", "")
        pinyin = sample.get("pinyin", "")

        image_matched = sample.get("image_matched", "")
        image_mismatched = sample.get("image_mismatched", "")

        if not pinyin:
            continue
        if str(sample_id) in completed_ids:
            continue

        prompt = build_prompt(pinyin)

        image_ok = safe_open_image(image_matched)
        image_wrong = safe_open_image(image_mismatched)

        try:
            pred_with_img = generate_one(model, processor, device, prompt, image_ok, MAX_NEW_TOKENS)
        except Exception as exc:
            pred_with_img = f"ERROR: {exc}"
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        try:
            pred_no_img = generate_one(model, processor, device, prompt, None, MAX_NEW_TOKENS)
        except Exception as exc:
            pred_no_img = f"ERROR: {exc}"
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        try:
            pred_wrong_img = generate_one(model, processor, device, prompt, image_wrong, MAX_NEW_TOKENS)
        except Exception as exc:
            pred_wrong_img = f"ERROR: {exc}"
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        result = {
            "sample_id": sample_id,
            "group_id": group_id,
            "target_word": target_word,
            "text_gt": text_gt,
            "pinyin": pinyin,
            "pred_with_img": pred_with_img,
            "pred_no_img": pred_no_img,
            "pred_wrong_img": pred_wrong_img,
            "image_matched": image_matched,
            "image_mismatched": image_mismatched,
            "source": sample.get("source", "unknown"),
        }
        results.append(result)
        append_jsonl(OUTPUT_PATH, result)
        completed_ids.add(str(sample_id))

    print(f"推理完成，总样本数: {len(results)}")
    print(f"结果已保存到: {OUTPUT_PATH}")

    count_with = sum(1 for r in results if contains_target(r["pred_with_img"], r["target_word"]))
    count_no = sum(1 for r in results if contains_target(r["pred_no_img"], r["target_word"]))
    count_wrong = sum(1 for r in results if contains_target(r["pred_wrong_img"], r["target_word"]))

    print("目标词命中率:")
    print(f"  +I: {count_with}/{len(results)} = {count_with / max(1, len(results)) * 100:.2f}%")
    print(f"  -I: {count_no}/{len(results)} = {count_no / max(1, len(results)) * 100:.2f}%")
    print(f"  -I_wrong: {count_wrong}/{len(results)} = {count_wrong / max(1, len(results)) * 100:.2f}%")


if __name__ == "__main__":
    main()
