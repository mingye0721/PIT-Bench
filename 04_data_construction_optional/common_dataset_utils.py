#!/usr/bin/env python3

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image, ImageStat, UnidentifiedImageError


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_image_ok(path: str, min_side: int = 16, min_var: float = 5.0) -> bool:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return False

    try:
        with Image.open(p) as im:
            im.load()
            w, h = im.size
            if w < min_side or h < min_side:
                return False

            gray = im.convert("L")
            stat = ImageStat.Stat(gray)
            if not stat.var:
                return False
            if stat.var[0] < min_var:
                return False
    except (UnidentifiedImageError, OSError, ValueError):
        return False

    return True


def alloc_by_ratio(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    items = list(ratios.items())
    raw = [(k, total * r) for k, r in items]
    base = {k: int(v) for k, v in raw}
    remain = total - sum(base.values())
    frac_sorted = sorted(raw, key=lambda x: x[1] - int(x[1]), reverse=True)
    i = 0
    while remain > 0 and frac_sorted:
        k = frac_sorted[i % len(frac_sorted)][0]
        base[k] += 1
        remain -= 1
        i += 1
    return base


def rand_sample(rows: List[dict], n: int, rng: random.Random) -> List[dict]:
    if n <= 0:
        return []
    if len(rows) <= n:
        return rows.copy()
    return rng.sample(rows, n)
