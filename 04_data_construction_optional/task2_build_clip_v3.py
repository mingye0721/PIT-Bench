#!/usr/bin/env python3

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
import torch
from opencc import OpenCC
from PIL import Image
from pypinyin import Style, pinyin
from transformers import CLIPModel, CLIPProcessor

from common_dataset_utils import alloc_by_ratio, load_jsonl, safe_image_ok, write_jsonl


SOURCES = {
    "mmchat": {"samples": 600, "groups": 300},
    "wukong": {"samples": 1200, "groups": 600},
    "coco": {"samples": 1200, "groups": 600},
}

GROUP_RATIO = {
    "2": 0.60,
    "3": 0.25,
    "4p": 0.15,
}

CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def to_pinyin(text: str) -> str:
    py = pinyin(text, style=Style.NORMAL, heteronym=False)
    return " ".join(x[0] for x in py)


def extract_words(text: str, min_len: int = 2, max_len: int = 6) -> List[str]:
    words = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok:
            continue
        if not CHINESE_RE.match(tok):
            continue
        if min_len <= len(tok) <= max_len:
            words.append(tok)
    return words


def basic_image_ok(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def task2_text_score(text: str, source: str) -> float:
    length = len(text)
    if length < 5:
        return 0.0
    if 8 <= length <= 24:
        base = 100.0
    elif 5 <= length < 8:
        base = 80.0
    elif 25 <= length <= 40:
        base = 70.0
    else:
        base = 45.0

    # 轻微偏向更稳定的来源，避免同源过拟合。
    source_bonus = {"wukong": 6.0, "mmchat": 4.0, "coco": 3.0}.get(source, 0.0)
    return base + source_bonus


class ClipScorer:
    def __init__(self, model_name: str, device: str) -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.cache: Dict[Tuple[str, str], float] = {}

    @torch.no_grad()
    def score(self, image_path: str, text: str) -> float:
        k = (image_path, text)
        if k in self.cache:
            return self.cache[k]

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            inputs = self.processor(text=[text], images=[im], return_tensors="pt", padding=True)
        inputs = {kk: vv.to(self.device) for kk, vv in inputs.items()}
        out = self.model(**inputs)
        img = out.image_embeds
        txt = out.text_embeds
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        s = float((img * txt).sum().item())
        self.cache[k] = s
        return s


def remove_trad_simp_duplicates(word_count: Dict[str, int]) -> List[str]:
    cc = OpenCC("t2s")
    by_simp: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for w, c in word_count.items():
        by_simp[cc.convert(w)].append((w, c))

    keep = []
    for _, items in by_simp.items():
        items.sort(key=lambda x: x[1], reverse=True)
        keep.append(items[0][0])
    return keep


def mine_groups(rows: List[dict]) -> Dict[str, Dict[str, List[int]]]:
    groups: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, r in enumerate(rows):
        text = r["text"]
        for w in set(extract_words(text)):
            groups[to_pinyin(w)][w].append(idx)

    filtered: Dict[str, Dict[str, List[int]]] = {}
    for py, wm in groups.items():
        wc = {w: len(idxs) for w, idxs in wm.items()}
        kept_words = remove_trad_simp_duplicates(wc)
        kept_map = {w: wm[w] for w in kept_words if wc[w] >= 2}
        if len(kept_map) >= 2:
            filtered[py] = kept_map
    return filtered


def build_text_reserved_rows(
    rows: List[dict],
    group_map: Dict[str, Dict[str, List[int]]],
    source: str,
    reserve_n: int,
    per_word_cap: int,
) -> List[dict]:
    return build_text_reserved_pool(rows, group_map, source, per_word_cap)[:reserve_n]


def build_text_reserved_pool(
    rows: List[dict],
    group_map: Dict[str, Dict[str, List[int]]],
    source: str,
    per_word_cap: int,
) -> List[dict]:
    reserved = []
    seen = set()

    for py, wm in group_map.items():
        for word, idxs in wm.items():
            scored = []
            for i in set(idxs):
                row = rows[i]
                score = task2_text_score(row["text"], source)
                scored.append((score, i))
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, i in scored[:per_word_cap]:
                row = dict(rows[i])
                row["text_score"] = score
                row["group_pinyin"] = py
                row["target_word"] = word
                key = row.get("id", str(i))
                if key in seen:
                    continue
                seen.add(key)
                reserved.append(row)

    reserved.sort(key=lambda x: x.get("text_score", 0.0), reverse=True)
    return reserved


def select_groups(group_map: Dict[str, Dict[str, List[int]]], n_groups: int) -> List[str]:
    stats = []
    for py, wm in group_map.items():
        n = len(wm)
        score = sum(len(v) for v in wm.values())
        stats.append((py, n, score))

    b2 = sorted([x for x in stats if x[1] == 2], key=lambda x: x[2], reverse=True)
    b3 = sorted([x for x in stats if x[1] == 3], key=lambda x: x[2], reverse=True)
    b4 = sorted([x for x in stats if x[1] >= 4], key=lambda x: x[2], reverse=True)

    need = alloc_by_ratio(n_groups, GROUP_RATIO)
    picked = b2[: need["2"]] + b3[: need["3"]] + b4[: need["4p"]]
    if len(picked) < n_groups:
        used = {x[0] for x in picked}
        rest = [x for x in sorted(stats, key=lambda x: x[2], reverse=True) if x[0] not in used]
        picked.extend(rest[: n_groups - len(picked)])

    return [x[0] for x in picked[:n_groups]]


def assemble_task2_outputs(
    scored_valid: List[dict],
    group_map: Dict[str, Dict[str, List[int]]],
    selected_pys: List[str],
    source: str,
    samples_target: int,
    rng: random.Random,
) -> Tuple[List[dict], List[dict]]:
    print(f"[{source}] step6 组内候选构建...")
    candidate_rows = []
    gid = 1
    for py in selected_pys:
        wm = group_map[py]
        for word, idxs in wm.items():
            scored = []
            for i in set(idxs):
                s = scored_valid[i]["clip_score"]
                scored.append((s, i))
            scored.sort(key=lambda x: x[0], reverse=True)
            for s, i in scored[:3]:
                v = scored_valid[i]
                candidate_rows.append(
                    {
                        "id": v["id"],
                        "group_id": gid,
                        "target_word": word,
                        "text": v["text"],
                        "pinyin": v["pinyin"],
                        "image": v["image"],
                        "source": source,
                        "coarse_score": round(s, 6),
                        "orig_image": v["orig_image"],
                        "group_pinyin": py,
                    }
                )
        gid += 1

    by_gid = defaultdict(list)
    for r in candidate_rows:
        by_gid[r["group_id"]].append(r)

    out = []
    for g, rs in by_gid.items():
        by_word = defaultdict(list)
        for r in rs:
            by_word[r["target_word"]].append(r)
        if len(by_word) < 2:
            continue
        for w, wrs in by_word.items():
            others = []
            for ow, ors in by_word.items():
                if ow != w:
                    others.extend(ors)
            if not others:
                continue
            for r in wrs:
                wrong = rng.choice([x for x in others if x["image"] != r["image"]] or others)
                out.append(
                    {
                        "sample_id": r["id"],
                        "group_id": g,
                        "target_word": w,
                        "text": r["text"],
                        "pinyin": r["pinyin"],
                        "image_matched": r["image"],
                        "image_mismatched": wrong["image"],
                        "image_none": None,
                        "source": source,
                        "coarse_score": r["coarse_score"],
                        "orig_image": r["orig_image"],
                        "wrong_from_word": wrong["target_word"],
                        "wrong_from_id": wrong["id"],
                    }
                )

    if len(out) > samples_target:
        out = rng.sample(out, samples_target)
    print(f"[{source}] task2 samples: {len(out)}")
    return candidate_rows, out


def build_task2_for_source(
    source: str,
    rows: List[dict],
    groups_target: int,
    samples_target: int,
    scorer: ClipScorer,
    rng: random.Random,
    text_reserve_multiplier: float,
    text_reserve_min_keep: int,
    text_per_word_cap: int,
) -> Tuple[List[dict], List[dict], dict]:
    print(f"\n[{source}] step1 过滤有效样本...")
    valid = []
    for i, r in enumerate(rows):
        text = r.get("text", "")
        img = r.get("img", "")
        py = r.get("pinyin", "")
        if not text or len(text) <= 4 or not img or not py:
            continue
        if not basic_image_ok(img):
            continue
        valid.append({
            "id": r.get("id", f"{source}_{i}"),
            "source": source,
            "text": text,
            "pinyin": py,
            "image": img,
            "orig_image": img,
        })
    print(f"[{source}] valid rows: {len(valid)}")

    print(f"[{source}] step2 同音组挖掘（全量有效文本）...")
    group_map = mine_groups(valid)
    print(f"[{source}] groups mined: {len(group_map)}")

    reserve_n = max(text_reserve_min_keep, int(samples_target * text_reserve_multiplier))
    print(f"[{source}] step3 文本优质候选预留: reserve_n={reserve_n}, per_word_cap={text_per_word_cap}")
    reserved_pool = build_text_reserved_pool(
        rows=valid,
        group_map=group_map,
        source=source,
        per_word_cap=text_per_word_cap,
    )
    print(f"[{source}] after text reserve pool: {len(reserved_pool)}")

    scored_valid: List[dict] = []
    best_candidate_rows: List[dict] = []
    best_out: List[dict] = []
    best_group_map: Dict[str, Dict[str, List[int]]] = {}
    best_selected_pys: List[str] = []
    best_checked = 0
    reserve_step = max(250, samples_target // 2)
    checked = 0
    reserve_cap = min(reserve_n, len(reserved_pool))
    round_idx = 0

    while True:
        round_idx += 1
        next_cap = min(len(reserved_pool), reserve_cap)
        if next_cap <= checked:
            if checked >= len(reserved_pool):
                break
            reserve_cap = min(len(reserved_pool), checked + reserve_step)
            continue

        batch = reserved_pool[checked:next_cap]
        print(f"[{source}] step4 round {round_idx}: 图像有效性 + CLIP 查验 {checked + 1}-{next_cap}/{len(reserved_pool)}")
        for idx, v in enumerate(batch, checked + 1):
            if not safe_image_ok(v["image"]):
                continue
            s = scorer.score(v["image"], v["text"])
            vv = dict(v)
            vv["clip_score"] = round(s, 6)
            scored_valid.append(vv)
            if idx % 500 == 0:
                print(f"[{source}] CLIP checked: {idx}/{len(reserved_pool)}")

        checked = next_cap
        scored_valid.sort(key=lambda x: x["clip_score"], reverse=True)
        print(f"[{source}] after clip check: {len(scored_valid)}")

        print(f"[{source}] step5 重新挖掘同音组（仅高分候选）...")
        group_map = mine_groups(scored_valid)
        if len(group_map) == 0:
            print(f"[{source}] WARNING: no groups after reservation")

        selected_pys = select_groups(group_map, groups_target) if group_map else []
        print(f"[{source}] selected groups: {len(selected_pys)}")

        candidate_rows, out = assemble_task2_outputs(
            scored_valid=scored_valid,
            group_map=group_map,
            selected_pys=selected_pys,
            source=source,
            samples_target=samples_target,
            rng=rng,
        )

        if len(out) > len(best_out):
            best_candidate_rows = candidate_rows
            best_out = out
            best_group_map = group_map
            best_selected_pys = selected_pys
            best_checked = checked

        if len(out) >= samples_target and len(selected_pys) >= groups_target:
            break
        if checked >= len(reserved_pool):
            break

        reserve_cap = min(len(reserved_pool), checked + reserve_step)

    candidate_rows = best_candidate_rows
    out = best_out
    group_map = best_group_map
    selected_pys = best_selected_pys
    clip_stats = {
        "valid_before_clip": len(valid),
        "after_text_reserve": len(reserved_pool),
        "after_clip_check": len(scored_valid),
        "text_reserve_multiplier": text_reserve_multiplier,
        "text_reserve_min_keep": text_reserve_min_keep,
        "text_per_word_cap": text_per_word_cap,
        "clip_rounds": round_idx,
        "clip_checked_until": best_checked,
        "groups_mined": len(group_map),
        "selected_groups": len(selected_pys),
    }
    return candidate_rows, out, clip_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Task2 homophone benchmark with CLIP ranking")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--mmchat", type=Path, default=Path("/data/lixy/dataset_processed/mmchat_all_with_img.jsonl"))
    parser.add_argument("--wukong", type=Path, default=Path("/data/lixy/dataset_processed/wukong_all_with_img.jsonl"))
    parser.add_argument("--coco", type=Path, default=Path("/data/lixy/dataset_processed/coco_all_with_img.jsonl"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/lixy/pinyin/data_construction/output_task2"),
    )
    parser.add_argument(
        "--text-reserve-multiplier",
        type=float,
        default=4.0,
        help="按测试集目标样本数的多少倍预留文本优质候选",
    )
    parser.add_argument(
        "--text-reserve-min-keep",
        type=int,
        default=2000,
        help="文本预留阶段至少保留的样本数",
    )
    parser.add_argument(
        "--text-per-word-cap",
        type=int,
        default=5,
        help="每个同音词在文本预留阶段最多保留多少条候选",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scorer = ClipScorer(args.clip_model, args.device)
    source_rows = {
        "mmchat": load_jsonl(args.mmchat),
        "wukong": load_jsonl(args.wukong),
        "coco": load_jsonl(args.coco),
    }

    all_candidates = []
    all_task2 = []
    stats = {}

    for source, cfg in SOURCES.items():
        print(f"\n========== build source: {source} ==========")
        cands, t2, clip_stats = build_task2_for_source(
            source=source,
            rows=source_rows[source],
            groups_target=cfg["groups"],
            samples_target=cfg["samples"],
            scorer=scorer,
            rng=rng,
            text_reserve_multiplier=args.text_reserve_multiplier,
            text_reserve_min_keep=args.text_reserve_min_keep,
            text_per_word_cap=args.text_per_word_cap,
        )
        write_jsonl(args.out_dir / f"task2_candidates_{source}.jsonl", cands)
        write_jsonl(args.out_dir / f"task2_{source}.jsonl", t2)
        all_candidates.extend(cands)
        all_task2.extend(t2)
        stats[source] = {
            "groups_target": cfg["groups"],
            "samples_target": cfg["samples"],
            "candidates": len(cands),
            "task2_samples": len(t2),
            "clip_prefilter": clip_stats,
        }

    rng.shuffle(all_task2)
    write_jsonl(args.out_dir / "task2_candidates_all.jsonl", all_candidates)
    write_jsonl(args.out_dir / "task2_all.jsonl", all_task2)

    with (args.out_dir / "task2_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "task2_homophone_disambiguation",
                "sources": SOURCES,
                "group_ratio": GROUP_RATIO,
                "clip_model": args.clip_model,
                "device": args.device,
                "text_reserve_multiplier": args.text_reserve_multiplier,
                "text_reserve_min_keep": args.text_reserve_min_keep,
                "text_per_word_cap": args.text_per_word_cap,
                "stats": stats,
                "total_task2": len(all_task2),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Task2 finished: {len(all_task2)} samples -> {args.out_dir}")


if __name__ == "__main__":
    main()
