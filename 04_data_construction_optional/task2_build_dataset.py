#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task2 测试集构建（统一版）

支持多来源输入（mmchat / wukong / coco），从原始文本+图片数据直接构建：
1) 中间候选集（group_id/target_word 等字段，兼容 task2_construction 思路）
2) 最终测试集（三种条件：matched / none / mismatched）

默认流程：
- jieba 分词提取中文词
- 按词拼音分组，保留同拼音且至少 2 个不同词的组
- 按组内候选词数进行比例抽样（2词组/3词组/4+词组）
- 每词保留 Top-K 候选句（按粗分）
- 组内互换图像生成 image_mismatched
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jieba
from pypinyin import Style, pinyin


CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def to_pinyin(text: str) -> str:
    py = pinyin(text, style=Style.NORMAL, heteronym=False)
    return " ".join(item[0] for item in py)


def coarse_quality_score(text: str, ideal_min: int, ideal_max: int) -> float:
    length = len(text)
    if ideal_min <= length <= ideal_max:
        return 100.0
    if length < ideal_min:
        return (length / ideal_min) * 100.0 if ideal_min > 0 else 0.0
    return (ideal_max / length) * 100.0 if length > 0 else 0.0


def extract_words(text: str, min_len: int, max_len: int) -> List[str]:
    words: List[str] = []
    for token in jieba.lcut(text):
        token = token.strip()
        if not token:
            continue
        if not CHINESE_RE.match(token):
            continue
        if min_len <= len(token) <= max_len:
            words.append(token)
    return words


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_input_specs(items: List[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"输入格式错误: {item}，应为 source=/path/to/file.jsonl")
        src, path = item.split("=", 1)
        src = src.strip().lower()
        p = Path(path.strip())
        specs.append((src, p))
    return specs


def normalize_row(row: Dict[str, Any], source_name: str, row_idx: int) -> Dict[str, Any] | None:
    text = row.get("text") or row.get("caption") or row.get("sentence") or ""
    text = str(text).strip()
    if not text:
        return None

    image = (
        row.get("image")
        or row.get("img")
        or row.get("img_path")
        or row.get("image_path")
        or ""
    )
    image = str(image).strip()
    if not image:
        return None

    sent_pinyin = row.get("pinyin")
    if not sent_pinyin:
        sent_pinyin = to_pinyin(text)

    rid = row.get("id")
    if rid is None or str(rid).strip() == "":
        rid = f"{source_name}_{row_idx}"

    source = row.get("source", source_name)

    return {
        "id": str(rid),
        "source": str(source),
        "text": text,
        "pinyin": str(sent_pinyin),
        "image": image,
        "orig_image": image,
    }


def load_and_normalize_inputs(input_specs: List[Tuple[str, Path]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for source_name, path in input_specs:
        rows = load_jsonl(path)
        kept = 0
        for i, row in enumerate(rows):
            norm = normalize_row(row, source_name=source_name, row_idx=i)
            if norm is None:
                continue
            merged.append(norm)
            kept += 1
        print(f"[{source_name}] 输入 {len(rows)} 条, 有效 {kept} 条: {path}")
    print(f"合并后总样本: {len(merged)}")
    return merged


def build_homophone_groups(
    samples: List[Dict[str, Any]],
    min_word_len: int,
    max_word_len: int,
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, int]]:
    # group_pinyin -> {word -> [sample_idx...]}
    groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    word_freq: Dict[str, int] = defaultdict(int)

    for idx, s in enumerate(samples):
        words = set(extract_words(s["text"], min_word_len, max_word_len))
        if not words:
            continue

        for w in words:
            py = to_pinyin(w)
            groups[py][w].append(idx)
            word_freq[w] += 1

    return groups, word_freq


def select_group_ids(
    groups: Dict[str, Dict[str, List[int]]],
    num_groups: int,
    ratio_2: float,
    ratio_3: float,
    ratio_4p: float,
) -> Dict[str, int]:
    stats: List[Tuple[str, int, int]] = []
    for py, word_map in groups.items():
        num_words = len(word_map)
        if num_words < 2:
            continue
        group_score = sum(len(v) for v in word_map.values())
        stats.append((py, num_words, group_score))

    bucket2 = [x for x in stats if x[1] == 2]
    bucket3 = [x for x in stats if x[1] == 3]
    bucket4p = [x for x in stats if x[1] >= 4]

    bucket2.sort(key=lambda x: x[2], reverse=True)
    bucket3.sort(key=lambda x: x[2], reverse=True)
    bucket4p.sort(key=lambda x: x[2], reverse=True)

    n2 = int(round(num_groups * ratio_2))
    n3 = int(round(num_groups * ratio_3))
    n4p = int(round(num_groups * ratio_4p))
    delta = num_groups - (n2 + n3 + n4p)
    n2 += delta

    selected: List[Tuple[str, int, int]] = []
    selected.extend(bucket2[:n2])
    selected.extend(bucket3[:n3])
    selected.extend(bucket4p[:n4p])

    if len(selected) < num_groups:
        used = {x[0] for x in selected}
        rest = [x for x in stats if x[0] not in used]
        rest.sort(key=lambda x: x[2], reverse=True)
        selected.extend(rest[: max(0, num_groups - len(selected))])

    selected = selected[:num_groups]
    selected.sort(key=lambda x: x[2], reverse=True)
    return {py: i + 1 for i, (py, _, _) in enumerate(selected)}


def build_candidate_rows(
    samples: List[Dict[str, Any]],
    groups: Dict[str, Dict[str, List[int]]],
    selected_group_ids: Dict[str, int],
    max_candidates_per_word: int,
    ideal_text_min: int,
    ideal_text_max: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for py, gid in selected_group_ids.items():
        word_map = groups[py]
        for word, idx_list in word_map.items():
            uniq_idx = sorted(set(idx_list))
            scored: List[Tuple[float, int]] = []
            for i in uniq_idx:
                text = samples[i]["text"]
                score = coarse_quality_score(text, ideal_text_min, ideal_text_max)
                scored.append((score, i))
            scored.sort(key=lambda x: x[0], reverse=True)

            keep = scored[:max_candidates_per_word]
            for score, i in keep:
                s = samples[i]
                rows.append(
                    {
                        "id": s["id"],
                        "group_id": gid,
                        "target_word": word,
                        "text": s["text"],
                        "image": s["image"],
                        "pinyin": s["pinyin"],
                        "source": s["source"],
                        "coarse_score": round(score, 4),
                        "orig_image": s["orig_image"],
                        "group_pinyin": py,
                    }
                )

    rows.sort(key=lambda x: (x["group_id"], x["target_word"], -x["coarse_score"]))
    return rows


def create_task2_set(candidate_rows: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    by_gid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_gid[row["group_id"]].append(row)

    out: List[Dict[str, Any]] = []
    for gid, rows in by_gid.items():
        by_word: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_word[r["target_word"]].append(r)

        if len(by_word) < 2:
            continue

        for target_word, word_rows in by_word.items():
            others: List[Dict[str, Any]] = []
            for w, rs in by_word.items():
                if w != target_word:
                    others.extend(rs)
            if not others:
                continue

            for row in word_rows:
                # 优先选与 matched 不同路径的错图
                diff_img = [x for x in others if x["image"] != row["image"]]
                wrong = rng.choice(diff_img if diff_img else others)

                out.append(
                    {
                        "sample_id": row["id"],
                        "group_id": gid,
                        "target_word": target_word,
                        "text": row["text"],
                        "pinyin": row["pinyin"],
                        "image_matched": row["image"],
                        "image_mismatched": wrong["image"],
                        "image_none": None,
                        "source": row.get("source", "unknown"),
                        "coarse_score": row.get("coarse_score", 0),
                        "orig_image": row.get("orig_image", row["image"]),
                        "wrong_from_word": wrong["target_word"],
                        "wrong_from_id": wrong["id"],
                    }
                )

    return out


def summarize_candidates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_gid_words: Dict[int, set] = defaultdict(set)
    by_gid_cnt: Dict[int, int] = defaultdict(int)
    src_cnt: Counter = Counter()

    for r in rows:
        gid = r["group_id"]
        by_gid_words[gid].add(r["target_word"])
        by_gid_cnt[gid] += 1
        src_cnt[r["source"]] += 1

    cand_dist: Counter = Counter()
    for gid, words in by_gid_words.items():
        cand_dist[len(words)] += by_gid_cnt[gid]

    return {
        "total_rows": len(rows),
        "total_groups": len(by_gid_words),
        "avg_rows_per_group": (len(rows) / len(by_gid_words)) if by_gid_words else 0.0,
        "candidate_distribution": dict(sorted(cand_dist.items())),
        "source_distribution": dict(src_cnt),
    }


def summarize_task2(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_gid_words: Dict[int, set] = defaultdict(set)
    by_gid_cnt: Dict[int, int] = defaultdict(int)
    src_cnt: Counter = Counter()
    len_dist = {"short": 0, "medium": 0, "long": 0}

    for r in rows:
        gid = r["group_id"]
        by_gid_words[gid].add(r["target_word"])
        by_gid_cnt[gid] += 1
        src_cnt[r["source"]] += 1
        l = len(r["text"])
        if l <= 8:
            len_dist["short"] += 1
        elif l <= 15:
            len_dist["medium"] += 1
        else:
            len_dist["long"] += 1

    cand_dist: Counter = Counter()
    for gid, words in by_gid_words.items():
        cand_dist[len(words)] += by_gid_cnt[gid]

    return {
        "total_samples": len(rows),
        "total_groups": len(by_gid_words),
        "avg_samples_per_group": (len(rows) / len(by_gid_words)) if by_gid_words else 0.0,
        "candidate_distribution": dict(sorted(cand_dist.items())),
        "source_distribution": dict(src_cnt),
        "length_distribution": len_dist,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task2 dataset from multi-source JSONL")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="输入，格式 source=/abs/path/file.jsonl，可重复指定",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/lixy/pinyin/homophone_data/task2_build"),
    )
    parser.add_argument("--num-groups", type=int, default=1200)
    parser.add_argument("--ratio-2", type=float, default=0.60)
    parser.add_argument("--ratio-3", type=float, default=0.25)
    parser.add_argument("--ratio-4p", type=float, default=0.15)
    parser.add_argument("--min-word-len", type=int, default=2)
    parser.add_argument("--max-word-len", type=int, default=6)
    parser.add_argument("--max-candidates-per-word", type=int, default=3)
    parser.add_argument("--ideal-text-min", type=int, default=10)
    parser.add_argument("--ideal-text-max", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratio_sum = args.ratio_2 + args.ratio_3 + args.ratio_4p
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-8):
        raise ValueError(f"比例和必须为1.0，当前为 {ratio_sum}")

    input_specs = parse_input_specs(args.input)
    all_samples = load_and_normalize_inputs(input_specs)

    groups, _ = build_homophone_groups(
        all_samples,
        min_word_len=args.min_word_len,
        max_word_len=args.max_word_len,
    )

    selected_group_ids = select_group_ids(
        groups,
        num_groups=args.num_groups,
        ratio_2=args.ratio_2,
        ratio_3=args.ratio_3,
        ratio_4p=args.ratio_4p,
    )
    print(f"筛选同音组: {len(selected_group_ids)}")

    candidate_rows = build_candidate_rows(
        all_samples,
        groups,
        selected_group_ids=selected_group_ids,
        max_candidates_per_word=args.max_candidates_per_word,
        ideal_text_min=args.ideal_text_min,
        ideal_text_max=args.ideal_text_max,
    )

    rng = random.Random(args.seed)
    task2_rows = create_task2_set(candidate_rows, rng)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = output_dir / "task2_candidates_top.jsonl"
    task2_path = output_dir / "task2_set.jsonl"
    meta_path = output_dir / "task2_metadata.json"

    write_jsonl(candidates_path, candidate_rows)
    write_jsonl(task2_path, task2_rows)

    metadata = {
        "config": {
            "inputs": [{"source": s, "path": str(p)} for s, p in input_specs],
            "num_groups": args.num_groups,
            "ratio_2": args.ratio_2,
            "ratio_3": args.ratio_3,
            "ratio_4p": args.ratio_4p,
            "min_word_len": args.min_word_len,
            "max_word_len": args.max_word_len,
            "max_candidates_per_word": args.max_candidates_per_word,
            "ideal_text_min": args.ideal_text_min,
            "ideal_text_max": args.ideal_text_max,
            "seed": args.seed,
        },
        "candidate_summary": summarize_candidates(candidate_rows),
        "task2_summary": summarize_task2(task2_rows),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Task2 构建完成")
    print("=" * 70)
    print(f"候选输出: {candidates_path} ({len(candidate_rows)} 条)")
    print(f"测试集输出: {task2_path} ({len(task2_rows)} 条)")
    print(f"元数据: {meta_path}")


if __name__ == "__main__":
    main()
