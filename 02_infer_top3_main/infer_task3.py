#!/usr/bin/env python3

"""
Qwen2-VL-2B Task3 推理脚本
支持有图像和无图像两种条件，分别存储结果。

输出格式保持与原 infer_task3.py 一致：
- OUTPUT_WITH_IMG: JSONL，每条包含基础字段 + pred
- OUTPUT_NO_IMG: JSONL，每条包含基础字段 + pred
"""

import json
import os
import re
from collections import defaultdict
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

TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "/data/lixy/pinyin/data_construction/output_task3/task3_all.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/data/lixy/pinyin/eval/qwen2vl2b_pitbench_outputs/task3_qwen_infer")
OUTPUT_WITH_IMG = os.path.join(OUTPUT_DIR, "qwen2vl2b_task3_with_img.jsonl")
OUTPUT_NO_IMG = os.path.join(OUTPUT_DIR, "qwen2vl2b_task3_no_img.jsonl")

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


def build_prompt(error_pinyin: str) -> str:
    return (
        "你现在执行拼音转写任务，根据给定拼音生成对应的汉字，注意语义合理性。\n"
        "注意：输入拼音可能有错误（音近混淆、键盘误触、音节遗漏等）。\n"
        "请先纠错再输出最终中文。\n"
        "输出格式必须为一个 JSON 对象：\n"
        '{"chars": "按拼音顺序生成的汉字序列"}\n'
        "现在请转写：\n"
        f"拼音：{(error_pinyin or '').strip()}"
    )


def load_model_and_processor():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print("=" * 70)
    print("Task3 拼音错误纠正推理 (Qwen2-VL-2B)")
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


def read_task3(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        rows: List[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    return json.loads(content)


def load_jsonl(path: str) -> List[dict]:
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


def extract_text_from_pred(pred: str) -> str:
    try:
        if pred.startswith("{"):
            data = json.loads(pred)
            return data.get("chars", pred)
        m = re.search(r'"chars"\s*:\s*"([^"]*)"', pred)
        if m:
            return m.group(1)
        return pred
    except Exception:
        return pred


def main() -> None:
    torch.manual_seed(100)
    model, processor, device = load_model_and_processor()

    print(f"读取测试数据: {TEST_DATA_PATH}")
    test_data = read_task3(TEST_DATA_PATH)
    if not test_data:
        raise ValueError("未能加载任何测试数据")
    print(f"成功加载 {len(test_data)} 条测试样本\n")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if not RESUME:
        Path(OUTPUT_WITH_IMG).write_text("", encoding="utf-8")
        Path(OUTPUT_NO_IMG).write_text("", encoding="utf-8")

    results_with_img = load_jsonl(OUTPUT_WITH_IMG) if RESUME else []
    results_no_img = load_jsonl(OUTPUT_NO_IMG) if RESUME else []
    completed_with = {str(r.get("id")) for r in results_with_img if r.get("id") is not None}
    completed_no = {str(r.get("id")) for r in results_no_img if r.get("id") is not None}
    completed_ids = completed_with & completed_no
    print(f"断点续跑: {'开启' if RESUME else '关闭'}，已完成 {len(completed_ids)} 条")

    for idx, sample in enumerate(tqdm(test_data, desc="Task3 推理")):
        sample_id = sample.get("id", f"task3_{idx}")
        text_gt = sample.get("text_gt", "")
        pinyin_correct = sample.get("pinyin_correct", "")
        pinyin_error = sample.get("pinyin_error", "")
        error_type = sample.get("error_type", "")
        error_positions = sample.get("error_positions", [])
        num_errors = sample.get("num_errors", 0)
        error_intensity = sample.get("error_intensity", "")
        image_path = sample.get("image", "")
        source = sample.get("source", "unknown")

        if not pinyin_error:
            continue
        if str(sample_id) in completed_ids:
            continue

        prompt = build_prompt(pinyin_error)
        image = safe_open_image(image_path)

        try:
            pred_with_img = generate_one(model, processor, device, prompt, image, MAX_NEW_TOKENS)
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

        base_info = {
            "id": sample_id,
            "text_gt": text_gt,
            "pinyin_correct": pinyin_correct,
            "pinyin_error": pinyin_error,
            "error_type": error_type,
            "error_positions": error_positions,
            "num_errors": num_errors,
            "error_intensity": error_intensity,
            "source": source,
            "image": image_path,
        }

        with_img = base_info.copy()
        with_img["pred"] = pred_with_img
        results_with_img.append(with_img)
        append_jsonl(OUTPUT_WITH_IMG, with_img)

        no_img = base_info.copy()
        no_img["pred"] = pred_no_img
        results_no_img.append(no_img)
        append_jsonl(OUTPUT_NO_IMG, no_img)
        completed_ids.add(str(sample_id))

    print("推理完成")
    print(f"有图结果: {OUTPUT_WITH_IMG} ({len(results_with_img)} 样本)")
    print(f"无图结果: {OUTPUT_NO_IMG} ({len(results_no_img)} 样本)")

    correct_with = sum(
        1 for r in results_with_img if extract_text_from_pred(r["pred"]).strip() == (r.get("text_gt", "").strip())
    )
    correct_no = sum(
        1 for r in results_no_img if extract_text_from_pred(r["pred"]).strip() == (r.get("text_gt", "").strip())
    )

    print("句子准确率:")
    print(f"  +I: {correct_with}/{len(results_with_img)} = {correct_with / max(1, len(results_with_img)) * 100:.2f}%")
    print(f"  -I: {correct_no}/{len(results_no_img)} = {correct_no / max(1, len(results_no_img)) * 100:.2f}%")

    type_counts = defaultdict(int)
    for r in results_with_img:
        type_counts[r.get("error_type", "")] += 1
    print("错误类型分布:")
    for key in sorted(type_counts):
        cnt = type_counts[key]
        print(f"  {key}: {cnt}")


if __name__ == "__main__":
    main()
