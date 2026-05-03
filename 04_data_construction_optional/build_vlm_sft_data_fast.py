#!/usr/bin/env python3

import argparse
import json
import random
import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def image_readable(path: str, min_side: int = 16) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return False

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(p) as im:
                im.verify()

            with Image.open(p) as im:
                w, h = im.size
                if w < min_side or h < min_side:
                    return False
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fast VLM SFT data from no-leak train pool")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/data/lixy/pinyin/data_construction/output_train/train_pool_noleak_200k.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/lixy/LLaMA-Factory/data/pinyin_vlm_noleak_fast.json"),
    )
    parser.add_argument("--max-samples", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = list(iter_jsonl(args.input))
    rng.shuffle(rows)
    rows = rows[: args.max_samples]

    out = []
    total_rows = len(rows)
    skipped_empty = 0
    skipped_missing = 0
    skipped_bad_image = 0
    for r in rows:
        pinyin = (r.get("pinyin") or "").strip()
        text = (r.get("text") or "").strip()
        img = (r.get("img") or "").strip()
        if not pinyin or not text or not img:
            skipped_empty += 1
            continue
        if not Path(img).exists():
            skipped_missing += 1
            continue
        if not image_readable(img):
            skipped_bad_image += 1
            continue

        out.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请根据拼音输出对应中文，仅输出汉字结果，不要解释。<image>" + pinyin,
                    },
                    {
                        "role": "assistant",
                        "content": text,
                    },
                ],
                "images": [img],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"input rows: {total_rows}")
    print(f"built samples: {len(out)} -> {args.output}")
    if skipped_empty:
        print(f"skipped empty rows: {skipped_empty}")
    if skipped_missing:
        print(f"skipped missing images: {skipped_missing}")
    if skipped_bad_image:
        print(f"skipped unreadable images: {skipped_bad_image}")


if __name__ == "__main__":
    main()
