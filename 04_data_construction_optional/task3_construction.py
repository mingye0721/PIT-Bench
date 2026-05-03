在task1的基础上运行这个代码：#!/usr/bin/env python3
# generate_task3_errors.py

"""
Task 3: 拼音错误纠正 - 错误数据生成
"""

import json
import random
import copy
from typing import List, Dict, Any

# ============================================================
# 配置
# ============================================================

INPUT_FILE = "/root/benchmark/task1/task1_test_final.jsonl"
OUTPUT_FILE = "/root/benchmark/task1/task3_error.json"
TOTAL_SAMPLES = 3500

# 错误类型分配（方案A:  真实场景优先）
ERROR_DISTRIBUTION = {
    "sound": 1400,  # 40% - 音近混淆
    "keyboard": 1050,  # 30% - 键盘误触
    "delete": 525,  # 15% - 音节遗漏
    "insert": 350,  # 10% - 音节冗余
    "swap": 175,  # 5% - 音节顺序错误
}

# 错误强度配置
ERROR_INTENSITY = {
    "light": 0.7,  # 70%样本：轻度错误（改1-2个音节）
    "medium": 0.25,  # 25%样本：中度错误（改3-5个音节）
    "heavy": 0.05,  # 5%样本：重度错误（改6+个音节）
}

# ============================================================
# 音近/键盘邻近规则
# ============================================================

# 音近混淆规则
SOUND_GROUPS = [
    ("z", "zh"), ("c", "ch"), ("s", "sh"),  # 平翘舌
    ("l", "n"),  # 鼻边音
    ("f", "h"),  # 唇齿音
    ("an", "ang"), ("en", "eng"),  # 前后鼻音
    ("in", "ing"),
]

# 键盘邻近规则
KEYBOARD_NEIGHBORS = {
    'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'],
    'r': ['e', 't', 'f'], 't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'],
    'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'],
    'p': ['o', 'l'],
    'a': ['q', 's', 'z'], 's': ['a', 'w', 'd', 'x'], 'd': ['s', 'e', 'f', 'c'],
    'f': ['d', 'r', 'g', 'v'], 'g': ['f', 't', 'h', 'b'], 'h': ['g', 'y', 'j', 'n'],
    'j': ['h', 'u', 'k', 'm'], 'k': ['j', 'i', 'l'], 'l': ['k', 'o', 'p'],
    'z': ['a', 'x'], 'x': ['z', 's', 'c'], 'c': ['x', 'd', 'v'],
    'v': ['c', 'f', 'b'], 'b': ['v', 'g', 'n'], 'n': ['b', 'h', 'm'],
    'm': ['n', 'j'],
}


# ============================================================
# 错误生成函数
# ============================================================


def replace_sound(token: str) -> str:
    """音近替换"""
    for a, b in SOUND_GROUPS:
        if token.startswith(a):
            return token.replace(a, b, 1)
        if token.startswith(b):
            return token.replace(b, a, 1)
        if token.endswith(a):
            return token[:-len(a)] + b
        if token.endswith(b):
            return token[:-len(b)] + a
    return token


def replace_keyboard(token: str) -> str:
    """键盘误触"""
    if not token:
        return token

    # 随机选一个字符替换
    idx = random.randrange(len(token))
    char = token[idx]

    if char in KEYBOARD_NEIGHBORS:
        new_char = random.choice(KEYBOARD_NEIGHBORS[char])
        return token[: idx] + new_char + token[idx + 1:]

    return token


def determine_max_changes(n_tokens: int, intensity: str) -> int:
    """根据强度确定最大修改音节数"""
    if intensity == "light":
        return max(1, min(2, n_tokens // 4))
    elif intensity == "medium":
        return max(2, min(5, n_tokens // 2))
    else:  # heavy
        return max(3, n_tokens // 2)


def make_error(pinyin: str, mode: str, intensity: str = "light") -> Dict[str, Any]:
    """
    生成拼音错误

    Returns:
        {
            "error_pinyin": "错误拼音",
            "error_positions": [索引列表],
            "num_errors": 错误数量
        }
    """
    tokens = pinyin.split()
    n_tokens = len(tokens)

    if n_tokens == 0:
        return {"error_pinyin": pinyin, "error_positions": [], "num_errors": 0}

    max_changes = determine_max_changes(n_tokens, intensity)
    error_positions = []
    num_errors = 0

    # ========================================
    # 音节级错误（delete/insert/swap）
    # ========================================
    if mode in ["delete", "insert", "swap"]:
        changes = 0
        while changes < max_changes and tokens:
            idx = random.randrange(len(tokens))

            if mode == "delete":
                if len(tokens) > 1:
                    del tokens[idx]
                    error_positions.append(idx)
                    changes += 1
                    num_errors += 1

            elif mode == "insert":
                # 插入重复音节
                tokens.insert(idx, tokens[idx])
                error_positions.append(idx)
                changes += 1
                num_errors += 1

            elif mode == "swap":
                if len(tokens) > 1 and idx < len(tokens) - 1:
                    tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]
                    error_positions.extend([idx, idx + 1])
                    changes += 1
                    num_errors += 1

    # ========================================
    # 音节内错误（sound/keyboard）
    # ========================================
    else:
        indices = list(range(len(tokens)))
        random.shuffle(indices)

        changes = 0
        for idx in indices:
            if changes >= max_changes:
                break

            original = tokens[idx]

            if mode == "sound":
                tokens[idx] = replace_sound(tokens[idx])
            elif mode == "keyboard":
                tokens[idx] = replace_keyboard(tokens[idx])

            if tokens[idx] != original:
                error_positions.append(idx)
                changes += 1
                num_errors += 1

    return {
        "error_pinyin": " ".join(tokens),
        "error_positions": sorted(list(set(error_positions))),
        "num_errors": num_errors
    }


# ============================================================
# 主函数
# ============================================================


def main():
    print("=" * 70)
    print("Task 3: 拼音错误纠正 - 数据生成")
    print("=" * 70)

    # 读取数据
    print(f"\n📖 读取数据:  {INPUT_FILE}")
    data=[]
    with open(INPUT_FILE, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"   共 {len(data)} 条原始数据")

    # 检查数量
    if len(data) < TOTAL_SAMPLES:
        print(f"⚠️  警告: 原始数据不足 {TOTAL_SAMPLES} 条，将复用部分数据")
        data = data * (TOTAL_SAMPLES // len(data) + 1)

    # 随机打乱
    random.seed(42)
    random.shuffle(data)
    data = data[:TOTAL_SAMPLES]

    print(f"\n🎲 错误类型分配:")
    for error_type, count in ERROR_DISTRIBUTION.items():
        percentage = count / TOTAL_SAMPLES * 100
        print(f"   {error_type:12s}:{count:4d} ({percentage:5.1f}%)")

    print(f"\n🎯 错误强度分布:")
    for intensity, ratio in ERROR_INTENSITY.items():
        count = int(TOTAL_SAMPLES * ratio)
        print(f"   {intensity:8s}:{count:4d} ({ratio * 100:5.1f}%)")

    # 分配错误类型
    print(f"\n🔧 生成错误数据...")

    error_types = []
    for error_type, count in ERROR_DISTRIBUTION.items():
        error_types.extend([error_type] * count)

    random.shuffle(error_types)

    # 生成错误数据
    noisy_dataset = []

    for i, entry in enumerate(data):
        # 确定错误强度
        rand = random.random()
        if rand < ERROR_INTENSITY["light"]:
            intensity = "light"
        elif rand < ERROR_INTENSITY["light"] + ERROR_INTENSITY["medium"]:
            intensity = "medium"
        else:
            intensity = "heavy"

        # 生成错误
        error_type = error_types[i]
        error_result = make_error(entry["pinyin"], error_type, intensity)

        # 构建输出
        new_entry = {
            "id": entry.get("id", f"task3_{i}"),
            "image": entry["img"],
            "text_gt": entry["text"],
            "pinyin_correct": entry["pinyin"],
            "pinyin_error": error_result["error_pinyin"],
            "error_type": error_type,
            "error_intensity": intensity,
            "error_positions": error_result["error_positions"],
            "num_errors": error_result["num_errors"],
        }

        noisy_dataset.append(new_entry)

    # 保存
    print(f"\n💾 保存到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf8") as f:
        json.dump(noisy_dataset, f, ensure_ascii=False, indent=2)

    # 统计
    print(f"\n📊 统计信息:")

    # 按错误类型
    type_counts = {}
    for entry in noisy_dataset:
        t = entry["error_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  错误类型分布:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t:12s}: {count:4d}")

    # 按强度
    intensity_counts = {}
    for entry in noisy_dataset:
        inten = entry["error_intensity"]
        intensity_counts[inten] = intensity_counts.get(inten, 0) + 1

    print(f"\n  错误强度分布:")
    for inten, count in sorted(intensity_counts.items()):
        print(f"    {inten:8s}: {count:4d}")

    # 平均错误数
    avg_errors = sum(e["num_errors"] for e in noisy_dataset) / len(noisy_dataset)
    print(f"\n  平均错误数:{avg_errors:.2f} 个音节/样本")

    print("\n" + "=" * 70)
    print("✅ Task 3 错误数据生成完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()，构建出task3的数据集