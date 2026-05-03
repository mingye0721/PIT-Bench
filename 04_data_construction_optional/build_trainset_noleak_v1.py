#!/usr/bin/env python3

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

from common_dataset_utils import load_jsonl, write_jsonl


def norm(s: str) -> str:
    return (s or "").strip()


def build_test_exclusion(
    task1_path: Path,
    task2_path: Path,
    task3_path: Path,
) -> Tuple[Set[Tuple[str, str, str]], Set[Tuple[str, str]], Set[str]]:
    triplets: Set[Tuple[str, str, str]] = set()
    text_py: Set[Tuple[str, str]] = set()
    images: Set[str] = set()

    for r in load_jsonl(task1_path):
        t = norm(r.get("text", ""))
        p = norm(r.get("pinyin", ""))
        i = norm(r.get("img", ""))
        if t and p:
            text_py.add((t, p))
        if t and p and i:
            triplets.add((t, p, i))
            images.add(i)

    for r in load_jsonl(task2_path):
        t = norm(r.get("text", ""))
        p = norm(r.get("pinyin", ""))
        i = norm(r.get("image_matched", ""))
        if t and p:
            text_py.add((t, p))
        if t and p and i:
            triplets.add((t, p, i))
            images.add(i)

    for r in load_jsonl(task3_path):
        t = norm(r.get("text_gt", ""))
        p = norm(r.get("pinyin_correct", ""))
        i = norm(r.get("image", ""))
        if t and p:
            text_py.add((t, p))
        if t and p and i:
            triplets.add((t, p, i))
            images.add(i)

    return triplets, text_py, images


def valid_source_row(r: dict) -> bool:
    t = norm(r.get("text", ""))
    p = norm(r.get("pinyin", ""))
    i = norm(r.get("img", ""))
    if not t or not p or not i:
        return False
    if len(t) <= 2:
        return False
    return Path(i).exists()


def sample_rows(rows: List[dict], n: int, rng: random.Random) -> List[dict]:
    if n <= 0:
        return []
    if len(rows) <= n:
        return rows.copy()
    return rng.sample(rows, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free trainset by excluding PIT-Bench tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmchat", type=Path, default=Path("/data/lixy/dataset_processed/mmchat_all_with_img.jsonl"))
    parser.add_argument("--wukong", type=Path, default=Path("/data/lixy/dataset_processed/wukong_all_with_img.jsonl"))
    parser.add_argument("--coco", type=Path, default=Path("/data/lixy/dataset_processed/coco_all_with_img.jsonl"))
    parser.add_argument("--task1-test", type=Path, default=Path("/data/lixy/pinyin/data_construction/output_task1/task1_all.jsonl"))
    parser.add_argument("--task2-test", type=Path, default=Path("/data/lixy/pinyin/data_construction/output_task2/task2_all_filled.jsonl"))
    parser.add_argument("--task3-test", type=Path, default=Path("/data/lixy/pinyin/data_construction/output_task3/task3_all.jsonl"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/lixy/pinyin/data_construction/output_train"),
    )
    parser.add_argument("--exclude-test-images", action="store_true")
    parser.add_argument("--target-400k", type=int, default=400000)
    parser.add_argument("--target-200k", type=int, default=200000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_triplets, test_text_py, test_images = build_test_exclusion(args.task1_test, args.task2_test, args.task3_test)
    print(
        f"test triplets: {len(test_triplets)}, "
        f"test text-pinyin pairs: {len(test_text_py)}, "
        f"test images: {len(test_images)}"
    )

    source_paths = {
        "mmchat": args.mmchat,
        "wukong": args.wukong,
        "coco": args.coco,
    }

    by_source: Dict[str, List[dict]] = {"mmchat": [], "wukong": [], "coco": []}
    dropped = {"invalid": 0, "test_triplet": 0, "test_text_py": 0, "test_image": 0, "dup": 0}
    seen_triplets: Set[Tuple[str, str, str]] = set()

    for source, p in source_paths.items():
        rows = load_jsonl(p)
        kept = []
        for idx, r in enumerate(rows):
            if not valid_source_row(r):
                dropped["invalid"] += 1
                continue

            t = norm(r.get("text", ""))
            py = norm(r.get("pinyin", ""))
            img = norm(r.get("img", ""))
            tri = (t, py, img)
            tp = (t, py)

            if tri in test_triplets:
                dropped["test_triplet"] += 1
                continue
            if tp in test_text_py:
                dropped["test_text_py"] += 1
                continue
            if args.exclude_test_images and img in test_images:
                dropped["test_image"] += 1
                continue
            if tri in seen_triplets:
                dropped["dup"] += 1
                continue
            seen_triplets.add(tri)

            kept.append(
                {
                    "id": f"{source}_{idx}",
                    "source": source,
                    "text": t,
                    "pinyin": py,
                    "img": img,
                    "length": len(t),
                }
            )
        by_source[source] = kept
        print(f"[{source}] kept: {len(kept)}")

    all_rows = by_source["mmchat"] + by_source["wukong"] + by_source["coco"]
    rng.shuffle(all_rows)
    write_jsonl(args.out_dir / "train_pool_noleak_all.jsonl", all_rows)

    # Build practical subsets for 3-day training.
    # Desired 20/40/40 may be impossible due to coco size, so we keep all coco first and redistribute remainder.
    coco_cap_400k = min(len(by_source["coco"]), int(args.target_400k * 0.40))
    remain_400k = max(0, args.target_400k - coco_cap_400k)
    mm_400k = min(len(by_source["mmchat"]), remain_400k // 3)
    wk_400k = min(len(by_source["wukong"]), remain_400k - mm_400k)
    # If still short due to caps, top up from whoever has more.
    cur_400k = mm_400k + wk_400k + coco_cap_400k
    short_400k = args.target_400k - cur_400k
    if short_400k > 0:
        add_mm = min(short_400k, max(0, len(by_source["mmchat"]) - mm_400k))
        mm_400k += add_mm
        short_400k -= add_mm
    if short_400k > 0:
        add_wk = min(short_400k, max(0, len(by_source["wukong"]) - wk_400k))
        wk_400k += add_wk
        short_400k -= add_wk

    train_400k = (
        sample_rows(by_source["mmchat"], mm_400k, rng)
        + sample_rows(by_source["wukong"], wk_400k, rng)
        + sample_rows(by_source["coco"], coco_cap_400k, rng)
    )
    rng.shuffle(train_400k)
    write_jsonl(args.out_dir / "train_pool_noleak_400k.jsonl", train_400k)

    coco_cap_200k = min(len(by_source["coco"]), int(args.target_200k * 0.20))
    remain_200k = max(0, args.target_200k - coco_cap_200k)
    mm_200k = min(len(by_source["mmchat"]), remain_200k // 3)
    wk_200k = min(len(by_source["wukong"]), remain_200k - mm_200k)
    cur_200k = mm_200k + wk_200k + coco_cap_200k
    short_200k = args.target_200k - cur_200k
    if short_200k > 0:
        add_mm = min(short_200k, max(0, len(by_source["mmchat"]) - mm_200k))
        mm_200k += add_mm
        short_200k -= add_mm
    if short_200k > 0:
        add_wk = min(short_200k, max(0, len(by_source["wukong"]) - wk_200k))
        wk_200k += add_wk

    train_200k = (
        sample_rows(by_source["mmchat"], mm_200k, rng)
        + sample_rows(by_source["wukong"], wk_200k, rng)
        + sample_rows(by_source["coco"], coco_cap_200k, rng)
    )
    rng.shuffle(train_200k)
    write_jsonl(args.out_dir / "train_pool_noleak_200k.jsonl", train_200k)

    meta = {
        "seed": args.seed,
        "dropped": dropped,
        "kept_by_source": {k: len(v) for k, v in by_source.items()},
        "total_kept": len(all_rows),
        "subset_400k": {
            "target": args.target_400k,
            "actual": len(train_400k),
            "mmchat": mm_400k,
            "wukong": wk_400k,
            "coco": coco_cap_400k,
        },
        "subset_200k": {
            "target": args.target_200k,
            "actual": len(train_200k),
            "mmchat": mm_200k,
            "wukong": wk_200k,
            "coco": coco_cap_200k,
        },
        "exclusion": {
            "test_triplets": len(test_triplets),
            "test_text_pinyin": len(test_text_py),
            "test_images": len(test_images),
            "exclude_test_images": args.exclude_test_images,
            "policy": "drop if (text,pinyin,img) in test OR (text,pinyin) in test" + (" OR image in test" if args.exclude_test_images else ""),
        },
    }

    import json

    with (args.out_dir / "train_pool_noleak_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
