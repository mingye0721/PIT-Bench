#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common_dataset_utils import alloc_by_ratio, load_jsonl, rand_sample, safe_image_ok, write_jsonl


TOTAL = 3500
SOURCE_COUNTS = {
    "mmchat": 700,
    "wukong": 1400,
    "coco": 1400,
}

LEN_RATIOS = {
    "short": 0.20,   # 3-6
    "medium": 0.60,  # 7-12
    "long": 0.20,    # >12
}


def len_bucket(text: str) -> str | None:
    n = len(text)
    if 3 <= n <= 6:
        return "short"
    if 7 <= n <= 12:
        return "medium"
    if n > 12:
        return "long"
    return None


def source_logic(source: str, rows: List[dict], target_n: int, rng: random.Random) -> List[dict]:
    bucketed: Dict[str, List[dict]] = defaultdict(list)
    for i, r in enumerate(rows):
        text = r.get("text", "")
        img = r.get("img", "")
        py = r.get("pinyin", "")
        if not text or not img or not py:
            continue
        b = len_bucket(text)
        if b is None:
            continue
        rr = {
            "id": r.get("id", f"{source}_{i}"),
            "source": source,
            "text": text,
            "pinyin": py,
            "img": img,
            "length": len(text),
            "length_bucket": b,
            "task": "task1_pinyin_to_text",
        }
        bucketed[b].append(rr)

    need = alloc_by_ratio(target_n, LEN_RATIOS)
    selected: List[dict] = []
    for b in ["short", "medium", "long"]:
        pool = bucketed[b].copy()
        rng.shuffle(pool)
        kept = []
        for row in pool:
            if safe_image_ok(row["img"]):
                kept.append(row)
            if len(kept) >= need[b]:
                break
        selected.extend(kept)

    if len(selected) < target_n:
        remain_pool = []
        used_ids = {x["id"] for x in selected}
        for b in ["short", "medium", "long"]:
            remain_pool.extend([x for x in bucketed[b] if x["id"] not in used_ids])
        rng.shuffle(remain_pool)
        for row in remain_pool:
            if safe_image_ok(row["img"]):
                selected.append(row)
            if len(selected) >= target_n:
                break

    return selected[:target_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task1 benchmark with fixed source/length ratios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mmchat",
        type=Path,
        default=Path("/data/lixy/dataset_processed/mmchat_all_with_img.jsonl"),
    )
    parser.add_argument(
        "--wukong",
        type=Path,
        default=Path("/data/lixy/dataset_processed/wukong_all_with_img.jsonl"),
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=Path("/data/lixy/dataset_processed/coco_all_with_img.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/lixy/pinyin/data_construction/output_task1"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "mmchat": load_jsonl(args.mmchat),
        "wukong": load_jsonl(args.wukong),
        "coco": load_jsonl(args.coco),
    }

    all_rows: List[dict] = []
    stats = {}
    for source, target_n in SOURCE_COUNTS.items():
        rows = source_logic(source, inputs[source], target_n, rng)
        write_jsonl(args.out_dir / f"task1_{source}.jsonl", rows)
        all_rows.extend(rows)
        stats[source] = {
            "target": target_n,
            "actual": len(rows),
            "avg_len": (sum(len(x["text"]) for x in rows) / len(rows)) if rows else 0.0,
        }

    rng.shuffle(all_rows)
    write_jsonl(args.out_dir / "task1_all.jsonl", all_rows)

    with (args.out_dir / "task1_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "task1_pinyin_to_text",
                "total_target": TOTAL,
                "total_actual": len(all_rows),
                "source_counts": SOURCE_COUNTS,
                "length_ratios": LEN_RATIOS,
                "stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Task1 finished: {len(all_rows)} samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
