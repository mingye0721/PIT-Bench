#!/usr/bin/env python3

"""
Qwen2-VL-2B Task1 Top-3 推理脚本
支持：
1. 带图像 (+I) Top-3
2. 不带图像 (-I) Top-3

输出格式：
- OUTPUT_WITH_IMG: JSON 数组，每条包含 id/pinyin/predictions/gt/img
- OUTPUT_NO_IMG: JSON 数组，每条包含 id/pinyin/predictions/gt
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

TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "/data/lixy/pinyin/data_construction/output_task1/task1_all.jsonl")
OUTPUT_WITH_IMG = os.environ.get(
    "OUTPUT_WITH_IMG",
    "/data/lixy/pinyin/eval/qwen2vl2b_pitbench_outputs/qwen2vl2b_task1_top3_with_img.json",
)
OUTPUT_NO_IMG = os.environ.get(
    "OUTPUT_NO_IMG",
    "/data/lixy/pinyin/eval/qwen2vl2b_pitbench_outputs/qwen2vl2b_task1_top3_no_img.json",
)

GPU_ID = os.environ.get("GPU_ID", "2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1024"))
RESUME = os.environ.get("RESUME", "1") != "0"
NUM_CANDIDATES = int(os.environ.get("NUM_CANDIDATES", "3"))

WITH_IMG_PROGRESS_JSONL = os.environ.get(
    "WITH_IMG_PROGRESS_JSONL", str(Path(OUTPUT_WITH_IMG).with_suffix(".progress.jsonl"))
)
NO_IMG_PROGRESS_JSONL = os.environ.get(
    "NO_IMG_PROGRESS_JSONL", str(Path(OUTPUT_NO_IMG).with_suffix(".progress.jsonl"))
)


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
        "你现在执行拼音转写任务，根据给定拼音生成对应的汉字，注意语义合理性。\n"
        "示例：拼音：shan po cao di liu gou zhen qie yi -> 汉字：山坡草地遛狗真惬意。\n"
        "输出格式必须为一个 JSON 对象：\n"
        '{"chars": "按拼音顺序生成的汉字序列"}\n'
        "现在请转写：\n"
        f"{(pinyin or '').strip()}"
    )


def load_model_and_processor():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print("=" * 70)
    print("Qwen2-VL-2B Task1 Top-3 推理")
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
    do_sample: bool,
    temperature: float,
    top_p: float,
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
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
        )

    input_len = int(inputs["input_ids"].shape[1])
    trimmed = generated_ids[0][input_len:]
    out = processor.batch_decode([trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return strip_output(out)


def generate_topk(
    model,
    processor,
    device: str,
    prompt: str,
    image: Optional[Image.Image],
    num_candidates: int,
) -> List[str]:
    # 第一个候选用 greedy，后续用 sampling，提高多样性。
    configs = [
        {"do_sample": False, "temperature": 1.0, "top_p": 1.0},
        {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
        {"do_sample": True, "temperature": 1.0, "top_p": 0.9},
    ]
    candidates: List[str] = []

    for i in range(num_candidates):
        cfg = configs[min(i, len(configs) - 1)]
        try:
            pred = generate_one(
                model=model,
                processor=processor,
                device=device,
                prompt=prompt,
                image=image,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=cfg["do_sample"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
            )
        except Exception as exc:
            pred = f"ERROR: {exc}"
        finally:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        if pred and pred not in candidates:
            candidates.append(pred)
        else:
            # 去重后数量不足时，保留当前结果占位，保证长度固定。
            candidates.append(pred if pred else "")

    while len(candidates) < num_candidates:
        candidates.append(candidates[-1] if candidates else "")

    return candidates[:num_candidates]


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_progress(path: str) -> List[dict]:
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


def append_progress(path: str, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    torch.manual_seed(100)
    model, processor, device = load_model_and_processor()

    print(f"读取测试数据: {TEST_DATA_PATH}")
    test_data = read_jsonl(TEST_DATA_PATH)
    if not test_data:
        raise ValueError("未能加载任何测试数据")
    print(f"成功加载 {len(test_data)} 条测试样本\n")

    # ============================================================
    # 带图像 (+I)
    # ============================================================
    print("=" * 70)
    print("【推理模式1：带图像 (+I) Top-3】")
    print("=" * 70)

    results_with_img = load_progress(WITH_IMG_PROGRESS_JSONL) if RESUME else []
    completed_with = {str(r.get("id")) for r in results_with_img if r.get("id") is not None}

    for idx, example in enumerate(tqdm(test_data, desc="推理(+I)Top-3")):
        pinyin = example.get("pinyin", "")
        image_path = example.get("img", "")
        gt = example.get("text", "")
        sample_id = example.get("id", f"sample_{idx}")

        if not pinyin or not gt or not image_path:
            continue
        if str(sample_id) in completed_with:
            continue

        image = safe_open_image(image_path)
        if image is None:
            preds = ["ERROR: 图像加载失败"] * NUM_CANDIDATES
        else:
            preds = generate_topk(
                model=model,
                processor=processor,
                device=device,
                prompt=build_prompt(pinyin),
                image=image,
                num_candidates=NUM_CANDIDATES,
            )

        record = {
            "id": sample_id,
            "pinyin": pinyin,
            "predictions": preds,
            "gt": gt,
            "img": image_path,
        }
        results_with_img.append(record)
        append_progress(WITH_IMG_PROGRESS_JSONL, record)
        completed_with.add(str(sample_id))

    Path(OUTPUT_WITH_IMG).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_WITH_IMG, "w", encoding="utf-8") as f:
        json.dump(results_with_img, f, ensure_ascii=False, indent=2)

    print(f"带图像推理完成，结果已保存到: {OUTPUT_WITH_IMG}")

    # ============================================================
    # 不带图像 (-I)
    # ============================================================
    print("\n" + "=" * 70)
    print("【推理模式2：不带图像 (-I) Top-3】")
    print("=" * 70)

    results_no_img = load_progress(NO_IMG_PROGRESS_JSONL) if RESUME else []
    completed_no = {str(r.get("id")) for r in results_no_img if r.get("id") is not None}

    for idx, example in enumerate(tqdm(test_data, desc="推理(-I)Top-3")):
        pinyin = example.get("pinyin", "")
        gt = example.get("text", "")
        sample_id = example.get("id", f"sample_{idx}")

        if not pinyin or not gt:
            continue
        if str(sample_id) in completed_no:
            continue

        preds = generate_topk(
            model=model,
            processor=processor,
            device=device,
            prompt=build_prompt(pinyin),
            image=None,
            num_candidates=NUM_CANDIDATES,
        )

        record = {
            "id": sample_id,
            "pinyin": pinyin,
            "predictions": preds,
            "gt": gt,
        }
        results_no_img.append(record)
        append_progress(NO_IMG_PROGRESS_JSONL, record)
        completed_no.add(str(sample_id))

    with open(OUTPUT_NO_IMG, "w", encoding="utf-8") as f:
        json.dump(results_no_img, f, ensure_ascii=False, indent=2)

    print(f"不带图像推理完成，结果已保存到: {OUTPUT_NO_IMG}")
    print("全部完成")


if __name__ == "__main__":
    main()
