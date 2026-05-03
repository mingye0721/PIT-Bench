#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from common_dataset_utils import alloc_by_ratio, load_jsonl, rand_sample, safe_image_ok, write_jsonl


TOTAL = 3500
SOURCE_COUNTS = {
    "mmchat": 700,
    "wukong": 1400,
    "coco": 1400,
}

ERROR_FAMILY_RATIO = {
    "sound": 0.40,
    "keyboard": 0.30,
    "others": 0.30,
}

OTHERS_TYPES = ["delete", "insert", "swap"]

KEY_NEIGHBOR = {
    "q": ["w", "a"], "w": ["q", "e", "s"], "e": ["w", "r", "d"], "r": ["e", "t", "f"],
    "t": ["r", "y", "g"], "y": ["t", "u", "h"], "u": ["y", "i", "j"], "i": ["u", "o", "k"],
    "o": ["i", "p", "l"], "p": ["o", "l"], "a": ["q", "s", "z"], "s": ["a", "w", "d", "x"],
    "d": ["s", "e", "f", "c"], "f": ["d", "r", "g", "v"], "g": ["f", "t", "h", "b"],
    "h": ["g", "y", "j", "n"], "j": ["h", "u", "k", "m"], "k": ["j", "i", "l"], "l": ["k", "o", "p"],
    "z": ["a", "x"], "x": ["z", "s", "c"], "c": ["x", "d", "v"], "v": ["c", "f", "b"],
    "b": ["v", "g", "n"], "n": ["b", "h", "m"], "m": ["n", "j"],
}

SOUND_PAIRS = [
    ("z", "zh"), ("c", "ch"), ("s", "sh"),
    ("l", "n"), ("f", "h"),
    ("an", "ang"), ("en", "eng"), ("in", "ing"),
]


def replace_sound(tok: str) -> str:
    for a, b in SOUND_PAIRS:
        if tok.startswith(a):
            return tok.replace(a, b, 1)
        if tok.startswith(b):
            return tok.replace(b, a, 1)
        if tok.endswith(a):
            return tok[: -len(a)] + b
        if tok.endswith(b):
            return tok[: -len(b)] + a
    return tok


def replace_keyboard(tok: str, rng: random.Random) -> str:
    if not tok:
        return tok
    idx = rng.randrange(len(tok))
    ch = tok[idx]
    if ch in KEY_NEIGHBOR:
        nch = rng.choice(KEY_NEIGHBOR[ch])
        return tok[:idx] + nch + tok[idx + 1:]
    return tok


def make_error(pinyin_text: str, etype: str, rng: random.Random) -> Tuple[str, List[int], int]:
    toks = pinyin_text.split()
    if not toks:
        return pinyin_text, [], 0

    pos = []
    if etype == "sound":
        i = rng.randrange(len(toks))
        new = replace_sound(toks[i])
        if new != toks[i]:
            toks[i] = new
            pos.append(i)
    elif etype == "keyboard":
        i = rng.randrange(len(toks))
        new = replace_keyboard(toks[i], rng)
        if new != toks[i]:
            toks[i] = new
            pos.append(i)
    elif etype == "delete":
        if len(toks) > 1:
            i = rng.randrange(len(toks))
            del toks[i]
            pos.append(i)
    elif etype == "insert":
        i = rng.randrange(len(toks))
        toks.insert(i, toks[i])
        pos.append(i)
    elif etype == "swap":
        if len(toks) > 1:
            i = rng.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
            pos.extend([i, i + 1])

    return " ".join(toks), sorted(set(pos)), len(set(pos))


def source_logic(source: str, rows: List[dict], target_n: int, rng: random.Random) -> List[dict]:
    pool = []
    for i, r in enumerate(rows):
        text = r.get("text", "")
        py = r.get("pinyin", "")
        img = r.get("img", "")
        if not text or not py or not img:
            continue
        pool.append(
            {
                "id": r.get("id", f"{source}_{i}"),
                "source": source,
                "image": img,
                "text_gt": text,
                "pinyin_correct": py,
            }
        )

    # 先抽样，再做图片有效性过滤，避免全量扫描图片。
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    base = []
    for row in shuffled:
        if safe_image_ok(row["image"]):
            base.append(row)
        if len(base) >= target_n:
            break

    family_counts = alloc_by_ratio(len(base), ERROR_FAMILY_RATIO)
    families = []
    for k, c in family_counts.items():
        families.extend([k] * c)
    rng.shuffle(families)

    out = []
    for i, row in enumerate(base):
        fam = families[i]
        if fam == "others":
            etype = rng.choice(OTHERS_TYPES)
        else:
            etype = fam

        ep, epos, nerr = make_error(row["pinyin_correct"], etype, rng)
        new_row = {
            **row,
            "pinyin_error": ep,
            "error_family": fam,
            "error_type": etype,
            "error_positions": epos,
            "num_errors": nerr,
            "task": "task3_pinyin_error_correction",
        }
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task3 pinyin-error benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmchat", type=Path, default=Path("/data/lixy/dataset_processed/mmchat_all_with_img.jsonl"))
    parser.add_argument("--wukong", type=Path, default=Path("/data/lixy/dataset_processed/wukong_all_with_img.jsonl"))
    parser.add_argument("--coco", type=Path, default=Path("/data/lixy/dataset_processed/coco_all_with_img.jsonl"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/lixy/pinyin/data_construction/output_task3"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = {
        "mmchat": load_jsonl(args.mmchat),
        "wukong": load_jsonl(args.wukong),
        "coco": load_jsonl(args.coco),
    }

    all_rows = []
    stats = {}
    for source, target_n in SOURCE_COUNTS.items():
        rows = source_logic(source, source_rows[source], target_n, rng)
        write_jsonl(args.out_dir / f"task3_{source}.jsonl", rows)
        all_rows.extend(rows)
        fam = defaultdict(int)
        for r in rows:
            fam[r["error_family"]] += 1
        stats[source] = {
            "target": target_n,
            "actual": len(rows),
            "error_family": dict(fam),
        }

    rng.shuffle(all_rows)
    write_jsonl(args.out_dir / "task3_all.jsonl", all_rows)

    with (args.out_dir / "task3_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "task3_pinyin_error_correction",
                "total_target": TOTAL,
                "total_actual": len(all_rows),
                "source_counts": SOURCE_COUNTS,
                "error_family_ratio": ERROR_FAMILY_RATIO,
                "stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Task3 finished: {len(all_rows)} samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
