在已经清洗好的数据集的基础上，通过jieba分词后进行pypinyin转换，找到出现频率最高的多音节拼音组，然后提取前1200个频率高的同音词组，其中2candidates的占60%，3占25，4+占15%。在每个同音词组内进行图像的互换来达到image——mismatched,构建代码为：#!/usr/bin/env python3
# convert_task2_dataset.py

"""
Task 2 数据转换脚本
将现有的同音词组数据转换为支持三种推理条件的测试集:
  1. 匹配图像 (+I): 每个样本用自己的原始图像
  2. 无图像 (-I): 不提供图像
  3. 错误图像 (-I_wrong): 用同组其他词的图像
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

# ============================================================
# 配置
# ============================================================

INPUT_FILE = "/root/autodl-tmp/bench/task2/task2_top1500.jsonl"
OUTPUT_DIR = "/root/autodl-tmp/bench/task2"

# 输出文件
OUTPUT_BASE = Path(OUTPUT_DIR) / "task2_set. jsonl"
OUTPUT_METADATA = Path(OUTPUT_DIR) / "task2_metadata.json"

# 随机种子
RANDOM_SEED = 42


# ============================================================
# 数据转换
# ============================================================


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """读取原始数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def group_by_group_id(data: List[Dict]) -> Dict[int, List[Dict]]:
    """按group_id分组"""
    groups = defaultdict(list)
    for item in data:
        gid = item['group_id']
        groups[gid].append(item)
    return groups


def create_test_samples(groups: Dict[int, List[Dict]]) -> List[Dict[str, Any]]:
    """
    为每个样本创建三种测试条件

    返回格式:
    {
        "sample_id": "word_816",
        "group_id":  2,
        "target_word":  "地方",
        "text": "这个破地方的三文鱼超级不新鲜",
        "pinyin": "zhe ge po di fang de san wen yu.. .",

        # 三种条件的图像路径
        "image_matched": "/path/to/matched. jpg",      # ← 原始图像
        "image_mismatched": "/path/to/wrong.jpg",     # ← 同组其他词的图像
        "image_none": null,                            # ← 无图像

        # 元数据
        "source": "mmchat",
        "coarse_score": 100,
        "orig_image":  "/path/to/orig.jpg"
    }
    """

    test_samples = []

    for group_id, samples in groups.items():
        # 按target_word分组
        samples_by_word = defaultdict(list)
        for s in samples:
            samples_by_word[s['target_word']].append(s)

        # 确保至少有2个不同的词
        if len(samples_by_word) < 2:
            print(f"⚠️  跳过 group {group_id}: 只有 {len(samples_by_word)} 个词")
            continue

        # 为每个样本分配错误图像
        for target_word, word_samples in samples_by_word.items():
            # 获取同组其他词的所有样本
            other_words_samples = []
            for other_word, other_samples in samples_by_word.items():
                if other_word != target_word:
                    other_words_samples.extend(other_samples)

            if not other_words_samples:
                continue

            # 为当前词的每个样本创建测试条件
            for sample in word_samples:
                # 随机选择一个错误图像
                wrong_sample = random.choice(other_words_samples)

                test_sample = {
                    # 基本信息
                    "sample_id": sample['id'],
                    "group_id": group_id,
                    "target_word": target_word,
                    "text": sample['text'],
                    "pinyin": sample['pinyin'],

                    # 三种条件的图像
                    "image_matched": sample['image'],  # ✅ 正确图像
                    "image_mismatched": wrong_sample['image'],  # ❌ 错误图像
                    "image_none": None,  # ⭕ 无图像

                    # 元数据
                    "source": sample.get('source', 'unknown'),
                    "coarse_score": sample.get('coarse_score', 0),
                    "orig_image": sample.get('orig_image', sample['image']),

                    # 调试信息
                    "wrong_from_word": wrong_sample['target_word'],
                    "wrong_from_id": wrong_sample['id']
                }

                test_samples.append(test_sample)

    return test_samples


def generate_metadata(test_samples: List[Dict]) -> Dict[str, Any]:
    """生成元数据统计"""

    # 按group统计
    groups = defaultdict(set)
    samples_per_group = defaultdict(int)

    for s in test_samples:
        gid = s['group_id']
        groups[gid].add(s['target_word'])
        samples_per_group[gid] += 1

    # 按候选词数量分布
    candidate_distribution = defaultdict(int)
    for gid, words in groups.items():
        num_candidates = len(words)
        candidate_distribution[num_candidates] += samples_per_group[gid]

    # 按来源统计
    source_distribution = defaultdict(int)
    for s in test_samples:
        source_distribution[s['source']] += 1

    # 文本长度分布
    length_distribution = {"short": 0, "medium": 0, "long": 0}
    for s in test_samples:
        length = len(s['text'])
        if length <= 8:
            length_distribution["short"] += 1
        elif length <= 15:
            length_distribution["medium"] += 1
        else:
            length_distribution["long"] += 1

    metadata = {
        "total_samples": len(test_samples),
        "total_groups": len(groups),
        "avg_samples_per_group": len(test_samples) / len(groups) if groups else 0,

        "candidate_distribution": {
            f"{k}_candidates": {
                "count": v,
                "percentage": v / len(test_samples) * 100
            }
            for k, v in sorted(candidate_distribution.items())
        },

        "source_distribution": {
            k: {"count": v, "percentage": v / len(test_samples) * 100}
            for k, v in source_distribution.items()
        },

        "length_distribution": {
            k: {"count": v, "percentage": v / len(test_samples) * 100}
            for k, v in length_distribution.items()
        },

        "group_details": [
            {
                "group_id": gid,
                "words": sorted(list(words)),
                "num_words": len(words),
                "num_samples": samples_per_group[gid]
            }
            for gid, words in sorted(groups.items())
        ]
    }

    return metadata


# ============================================================
# 主函数
# ============================================================


def main():
    print("=" * 70)
    print("Task 2 数据转换脚本")
    print("=" * 70)
    print(f"\n输入文件: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIR}\n")

    # 设置随机种子
    random.seed(RANDOM_SEED)

    # Step 1: 读取数据
    print("📖 读取原始数据...")
    data = load_data(INPUT_FILE)
    print(f"   读取 {len(data)} 条样本")

    # Step 2: 按group_id分组
    print("\n📊 按同音词组分组...")
    groups = group_by_group_id(data)
    print(f"   共 {len(groups)} 个同音词组")

    # 统计每组的词数
    group_sizes = defaultdict(int)
    for gid, samples in groups.items():
        words = set(s['target_word'] for s in samples)
        group_sizes[len(words)] += 1

    print(f"\n   词数分布:")
    for num_words in sorted(group_sizes.keys()):
        print(f"     {num_words} 个候选词:  {group_sizes[num_words]} 组")

    # Step 3: 创建测试样本
    print("\n🔧 生成测试样本...")
    test_samples = create_test_samples(groups)
    print(f"   生成 {len(test_samples)} 个测试样本")

    # Step 4: 保存测试集
    print(f"\n💾 保存测试集到 {OUTPUT_BASE}...")
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_BASE, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"   ✅ 已保存 {len(test_samples)} 个样本")

    # Step 5: 生成元数据
    print(f"\n📊 生成元数据...")
    metadata = generate_metadata(test_samples)

    with open(OUTPUT_METADATA, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 元数据已保存到 {OUTPUT_METADATA}")

    # Step 6: 打印统计信息
    print("\n" + "=" * 70)
    print("数据集统计")
    print("=" * 70)

    print(f"\n总样本数: {metadata['total_samples']}")
    print(f"同音词组数: {metadata['total_groups']}")
    print(f"平均每组样本数: {metadata['avg_samples_per_group']:.2f}")

    print(f"\n【候选词数量分布】")
    for k, v in metadata['candidate_distribution'].items():
        print(f"  {k}: {v['count']} 样本 ({v['percentage']:.1f}%)")

    print(f"\n【来源分布】")
    for k, v in metadata['source_distribution'].items():
        print(f"  {k}: {v['count']} 样本 ({v['percentage']:.1f}%)")

    print(f"\n【文本长度分布】")
    for k, v in metadata['length_distribution'].items():
        print(f"  {k}: {v['count']} 样本 ({v['percentage']:.1f}%)")

    print("\n" + "=" * 70)
    print("✅ 数据转换完成！")
    print("=" * 70)

    print(f"\n下一步:")
    print(f"  1. 查看测试集:  cat {OUTPUT_BASE} | head")
    print(f"  2. 运行推理:  python inference_task2.py")


if __name__ == '__main__':
    main()