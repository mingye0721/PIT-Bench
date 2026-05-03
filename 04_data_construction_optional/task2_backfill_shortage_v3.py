#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from common_dataset_utils import load_jsonl, write_jsonl


def alloc_by_weight(total: int, weights: Dict[str, int]) -> Dict[str, int]:
    names = list(weights.keys())
    s = float(sum(weights.values()))
    if s <= 0:
        return {k: 0 for k in names}
    raw = {k: total * (weights[k] / s) for k in names}
    out = {k: int(raw[k]) for k in names}
    rem = total - sum(out.values())
    frac = sorted([(raw[k] - out[k], k) for k in names], reverse=True)
    for i in range(rem):
        out[frac[i % len(frac)][1]] += 1
    return out


def build_extra_rows(
    source: str,
    candidates: List[dict],
    used_sample_ids: set,
    need: int,
    min_coarse_score: float,
    rng: random.Random,
) -> List[dict]:
    # Use high-coarse-score candidates first.
    filtered = [
        r
        for r in candidates
        if str(r.get("id")) not in used_sample_ids
        and float(r.get("coarse_score", 0.0)) >= min_coarse_score
    ]
    filtered.sort(key=lambda x: float(x.get("coarse_score", 0.0)), reverse=True)

    by_gid = defaultdict(list)
    for r in filtered:
        by_gid[r["group_id"]].append(r)

    out = []
    for gid, rows in by_gid.items():
        by_word = defaultdict(list)
        for r in rows:
            by_word[r["target_word"]].append(r)
        if len(by_word) < 2:
            continue

        for _, wrs in by_word.items():
            others = []
            for ow, ors in by_word.items():
                if ow != wrs[0]["target_word"]:
                    others.extend(ors)
            if not others:
                continue

            for r in wrs:
                rid = str(r.get("id"))
                if rid in used_sample_ids:
                    continue
                wrong_pool = [x for x in others if x["image"] != r["image"]] or others
                wrong = rng.choice(wrong_pool)
                out.append(
                    {
                        "sample_id": r["id"],
                        "group_id": gid,
                        "target_word": r["target_word"],
                        "text": r["text"],
                        "pinyin": r["pinyin"],
                        "image_matched": r["image"],
                        "image_mismatched": wrong["image"],
                        "image_none": None,
                        "source": source,
                        "coarse_score": r["coarse_score"],
                        "orig_image": r.get("orig_image", r["image"]),
                        "wrong_from_word": wrong["target_word"],
                        "wrong_from_id": wrong["id"],
                    }
                )
                used_sample_ids.add(rid)
                if len(out) >= need:
                    return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Task2 shortage from existing candidate pools")
    parser.add_argument("--out-dir", type=Path, default=Path("/data/lixy/pinyin/data_construction/output_task2"))
    parser.add_argument("--target-total", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fallback-order",
        type=str,
        default="wukong,mmchat",
        help="Sources used for backfill, comma-separated",
    )
    parser.add_argument(
        "--min-coarse-score",
        type=float,
        default=0.0,
        help="Hard lower-bound for CLIP coarse_score in final dataset",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = args.out_dir

    rows_by_source_raw = {
        "mmchat": load_jsonl(out_dir / "task2_mmchat.jsonl"),
        "wukong": load_jsonl(out_dir / "task2_wukong.jsonl"),
        "coco": load_jsonl(out_dir / "task2_coco.jsonl"),
    }
    cands_by_source = {
        "mmchat": load_jsonl(out_dir / "task2_candidates_mmchat.jsonl"),
        "wukong": load_jsonl(out_dir / "task2_candidates_wukong.jsonl"),
        "coco": load_jsonl(out_dir / "task2_candidates_coco.jsonl"),
    }

    rows_by_source = {
        s: [r for r in rows_by_source_raw[s] if float(r.get("coarse_score", 0.0)) >= args.min_coarse_score]
        for s in rows_by_source_raw
    }

    total_now = sum(len(v) for v in rows_by_source.values())
    if total_now >= args.target_total:
        print(f"No backfill needed: total={total_now} >= target={args.target_total}")
        return

    shortage = args.target_total - total_now
    fallback = [x.strip() for x in args.fallback_order.split(",") if x.strip()]
    fallback = [x for x in fallback if x in rows_by_source]
    if not fallback:
        raise ValueError("fallback-order has no valid sources")

    weights = {"mmchat": 1, "wukong": 2, "coco": 2}
    alloc = alloc_by_weight(shortage, {s: weights.get(s, 1) for s in fallback})

    used_ids = {
        s: set(str(x["sample_id"]) for x in rows_by_source[s]) for s in rows_by_source
    }

    extras = {s: [] for s in rows_by_source}
    for s in fallback:
        extra = build_extra_rows(
            source=s,
            candidates=cands_by_source[s],
            used_sample_ids=used_ids[s],
            need=alloc[s],
            min_coarse_score=args.min_coarse_score,
            rng=rng,
        )
        extras[s].extend(extra)

    got = sum(len(extras[s]) for s in fallback)
    if got < shortage:
        remain = shortage - got
        for s in fallback:
            if remain <= 0:
                break
            more = build_extra_rows(
                source=s,
                candidates=cands_by_source[s],
                used_sample_ids=used_ids[s],
                need=remain,
                min_coarse_score=args.min_coarse_score,
                rng=rng,
            )
            extras[s].extend(more)
            remain -= len(more)

    filled_by_source = {}
    for s in rows_by_source:
        filled = list(rows_by_source[s]) + extras[s]
        rng.shuffle(filled)
        filled_by_source[s] = filled

    all_filled = []
    for s in ["mmchat", "wukong", "coco"]:
        all_filled.extend(filled_by_source[s])
    rng.shuffle(all_filled)

    write_jsonl(out_dir / "task2_mmchat_filled.jsonl", filled_by_source["mmchat"])
    write_jsonl(out_dir / "task2_wukong_filled.jsonl", filled_by_source["wukong"])
    write_jsonl(out_dir / "task2_coco_filled.jsonl", filled_by_source["coco"])
    write_jsonl(out_dir / "task2_all_filled.jsonl", all_filled)

    meta = {
        "target_total": args.target_total,
        "base_total": total_now,
        "filled_total": len(all_filled),
        "shortage": shortage,
        "min_coarse_score": args.min_coarse_score,
        "fallback_order": fallback,
        "added_by_source": {s: len(extras[s]) for s in extras},
        "final_by_source": {s: len(filled_by_source[s]) for s in filled_by_source},
    }
    with (out_dir / "task2_meta_filled.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
