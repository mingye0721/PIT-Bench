#!/usr/bin/env python3

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
from pypinyin import Style, pinyin


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_pinyin(text: str) -> str:
    py = pinyin(text, style=Style.NORMAL, heteronym=False)
    return " ".join(item[0] for item in py)


def valid_row(row: dict) -> bool:
    text = (row.get("text") or "").strip()
    py = (row.get("pinyin") or "").strip()
    img = (row.get("img") or "").strip()
    return bool(text and py and img and Path(img).exists())


def extract_words(text: str, min_len: int = 2, max_len: int = 6) -> List[str]:
    words = []
    for token in jieba.lcut(text):
        token = token.strip()
        if not token:
            continue
        if not all("\u4e00" <= ch <= "\u9fff" for ch in token):
            continue
        if min_len <= len(token) <= max_len:
            words.append(token)
    return words


def prompt_task1(py: str) -> str:
    return "请根据拼音输出对应中文，仅输出汉字结果，不要解释。<image>" + py


def prompt_task2(py: str) -> str:
    return "你现在执行同音词消歧任务。根据给定拼音和图片语义，输出对应的中文句子。只输出汉字结果，不要解释。<image>" + py


def prompt_task3(py: str) -> str:
    return "你现在执行拼音纠错转写任务。给定的拼音中可能有错误，请结合图片语义恢复正确中文。只输出汉字结果，不要解释。<image>" + py


SOUND_PAIRS = [
    ("z", "zh"), ("c", "ch"), ("s", "sh"),
    ("l", "n"), ("f", "h"),
    ("an", "ang"), ("en", "eng"), ("in", "ing"),
]

KEY_NEIGHBOR = {
    "q": ["w", "a"], "w": ["q", "e", "s"], "e": ["w", "r", "d"], "r": ["e", "t", "f"],
    "t": ["r", "y", "g"], "y": ["t", "u", "h"], "u": ["y", "i", "j"], "i": ["u", "o", "k"],
    "o": ["i", "p", "l"], "p": ["o", "l"], "a": ["q", "s", "z"], "s": ["a", "w", "d", "x"],
    "d": ["s", "e", "f", "c"], "f": ["d", "r", "g", "v"], "g": ["f", "t", "h", "b"],
    "h": ["g", "y", "j", "n"], "j": ["h", "u", "k", "m"], "k": ["j", "i", "l"], "l": ["k", "o", "p"],
    "z": ["a", "x"], "x": ["z", "s", "c"], "c": ["x", "d", "v"], "v": ["c", "f", "b"],
    "b": ["v", "g", "n"], "n": ["b", "h", "m"], "m": ["n", "j"],
}


def replace_sound(token: str) -> str:
    for a, b in SOUND_PAIRS:
        if token.startswith(a):
            return token.replace(a, b, 1)
        if token.startswith(b):
            return token.replace(b, a, 1)
        if token.endswith(a):
            return token[: -len(a)] + b
        if token.endswith(b):
            return token[: -len(b)] + a
    return token


def replace_keyboard(token: str, rng: random.Random) -> str:
    if not token:
        return token
    idx = rng.randrange(len(token))
    ch = token[idx]
    if ch in KEY_NEIGHBOR:
        return token[:idx] + rng.choice(KEY_NEIGHBOR[ch]) + token[idx + 1 :]
    return token


def make_error(pinyin_text: str, error_type: str, rng: random.Random) -> str:
    tokens = pinyin_text.split()
    if not tokens:
        return pinyin_text
    if error_type == "sound":
        idx = rng.randrange(len(tokens))
        tokens[idx] = replace_sound(tokens[idx])
    elif error_type == "keyboard":
        idx = rng.randrange(len(tokens))
        tokens[idx] = replace_keyboard(tokens[idx], rng)
    elif error_type == "delete" and len(tokens) > 1:
        del tokens[rng.randrange(len(tokens))]
    elif error_type == "insert":
        idx = rng.randrange(len(tokens))
        tokens.insert(idx, tokens[idx])
    elif error_type == "swap" and len(tokens) > 1:
        idx = rng.randrange(len(tokens) - 1)
        tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
    return " ".join(tokens)


def build_task2_candidates(rows: List[dict]) -> List[dict]:
    groups: Dict[str, Counter] = defaultdict(Counter)
    row_words: List[List[str]] = []
    for row in rows:
        words = list(set(extract_words((row.get("text") or "").strip())))
        row_words.append(words)
        for word in words:
            groups[to_pinyin(word)][word] += 1

    ambiguous_groups = {
        py: {word for word, count in counter.items() if count >= 2}
        for py, counter in groups.items()
        if sum(1 for _, count in counter.items() if count >= 2) >= 2
    }

    candidates = []
    for row, words in zip(rows, row_words):
        matched = []
        for word in words:
            py = to_pinyin(word)
            vocab = ambiguous_groups.get(py)
            if vocab and word in vocab:
                matched.append((word, py, len(vocab)))
        if not matched:
            continue
        matched.sort(key=lambda item: item[2], reverse=True)
        target_word, group_py, group_size = matched[0]
        candidates.append(
            {
                "row": row,
                "target_word": target_word,
                "group_pinyin": group_py,
                "group_size": group_size,
            }
        )
    candidates.sort(key=lambda item: (item["group_size"], len(item["row"].get("text", ""))), reverse=True)
    return candidates


def build_record(prompt: str, answer: str, image_path: str, task_tag: str, extra: dict | None = None) -> dict:
    payload = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "images": [image_path],
        "task_tag": task_tag,
    }
    if extra:
        payload["meta"] = extra
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multitask no-leak VLM SFT data for Track-B enhanced training")
    parser.add_argument("--input", type=Path, default=Path("/data/lixy/pinyin/data_construction/output_train/train_pool_noleak_200k.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("/data/lixy/LLaMA-Factory/data/pinyin_vlm_noleak_multitask_120k.json"))
    parser.add_argument("--meta-out", type=Path, default=Path("/data/lixy/LLaMA-Factory/data/pinyin_vlm_noleak_multitask_120k_meta.json"))
    parser.add_argument("--target-total", type=int, default=120000)
    parser.add_argument("--task1-ratio", type=float, default=0.70)
    parser.add_argument("--task2-ratio", type=float, default=0.20)
    parser.add_argument("--task3-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [row for row in iter_jsonl(args.input) if valid_row(row)]
    rng.shuffle(rows)
    if not rows:
        raise RuntimeError("No valid rows found in train pool.")

    task1_target = int(args.target_total * args.task1_ratio)
    task2_target = int(args.target_total * args.task2_ratio)
    task3_target = args.target_total - task1_target - task2_target

    task1_rows = rows[: min(task1_target, len(rows))]
    task1_records = [
        build_record(
            prompt=prompt_task1((row.get("pinyin") or "").strip()),
            answer=(row.get("text") or "").strip(),
            image_path=(row.get("img") or "").strip(),
            task_tag="task1",
            extra={"source": row.get("source"), "id": row.get("id")},
        )
        for row in task1_rows
    ]

    task2_candidates = build_task2_candidates(rows)
    if not task2_candidates:
        raise RuntimeError("No Task2-style candidates mined from no-leak pool.")
    task2_records = []
    for idx in range(task2_target):
        item = task2_candidates[idx % len(task2_candidates)]
        row = item["row"]
        task2_records.append(
            build_record(
                prompt=prompt_task2((row.get("pinyin") or "").strip()),
                answer=(row.get("text") or "").strip(),
                image_path=(row.get("img") or "").strip(),
                task_tag="task2",
                extra={
                    "source": row.get("source"),
                    "id": row.get("id"),
                    "target_word": item["target_word"],
                    "group_pinyin": item["group_pinyin"],
                    "group_size": item["group_size"],
                },
            )
        )

    error_types = ["sound", "keyboard", "delete", "insert", "swap"]
    task3_base_rows = rows[:]
    rng.shuffle(task3_base_rows)
    task3_records = []
    for idx in range(task3_target):
        row = task3_base_rows[idx % len(task3_base_rows)]
        error_type = error_types[idx % len(error_types)]
        corrupted = make_error((row.get("pinyin") or "").strip(), error_type, rng)
        task3_records.append(
            build_record(
                prompt=prompt_task3(corrupted),
                answer=(row.get("text") or "").strip(),
                image_path=(row.get("img") or "").strip(),
                task_tag="task3",
                extra={"source": row.get("source"), "id": row.get("id"), "error_type": error_type},
            )
        )

    records = task1_records + task2_records + task3_records
    rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False)

    meta = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "source_valid_rows": len(rows),
        "target_total": args.target_total,
        "actual_total": len(records),
        "task_counts": dict(Counter(record["task_tag"] for record in records)),
        "task2_candidate_rows": len(task2_candidates),
    }
    with args.meta_out.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()