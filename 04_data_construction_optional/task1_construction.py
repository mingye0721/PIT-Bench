在已有的清洗过的数据集：#!/usr/bin/env python3
# merge_with_images.py

"""
完整流程：
1. 从原始数据集按比例采样
2. 从补充数据集补充长句
3. 检查图片是否存在
4. 复制图片到新目录
5. 更新样本中的图片路径
"""

import json
import os
import shutil
from tqdm import tqdm
from collections import Counter
import random
from pathlib import Path
import hashlib

# ============================================================
# 配置
# ============================================================

# 阶段1：原始数据集
ORIGINAL_FILES = {
    'coco': {
        'path': "/root/benchmark/raw_bench/coco_benchmark.jsonl",
        'count': 1400,
        'start': 0
    },
    'mmchat': {
        'path': "/root/benchmark/raw_bench/mmchat_benchmark.jsonl",
        'count': 700,
        'start': 0
    },
    'wukong': {
        'path': "/root/benchmark/raw_bench/wukong_benchmark.jsonl",
        'count': 1400,
        'start': 600
    }
}

# 阶段2：补充长句
SUPPLEMENT_FILES = {
    'coco': {
        'path': "/root/autodl-tmp/bench/coco_all_with_img.jsonl",
        'target_long': 160
    },
    'mmchat': {
        'path': "/root/autodl-tmp/bench/mmchat_all_with_img.jsonl",
        'target_long': 79
    },
    'wukong': {
        'path': "/root/autodl-tmp/bench/wukong_all_with_img.jsonl",
        'target_long': 160
    }
}

# 输出配置
OUTPUT_FILE = "/root/benchmark/task1_test_final.jsonl"
TARGET_IMAGE_DIR = "/root/benchmark/task1_images"

# 随机种子
RANDOM_SEED = 42


# ============================================================


def get_file_hash(file_path):
    """计算文件MD5哈希"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None


def classify_length(text):
    """长度分类"""
    length = len(text)
    if 3 <= length <= 6:
        return "3-6"
    elif 7 <= length <= 10:
        return "7-10"
    elif length > 10:
        return ">10"
    else:
        return "other"


def copy_image(original_path, target_dir, copied_images):
    """
    复制图片到目标目录

    Args:
        original_path: 原始图片路径
        target_dir: 目标目录
        copied_images: 已复制图片的映射 {原路径: 新路径}

    Returns:
        新的图片路径，如果失败返回None
    """
    # 如果已经复制过，直接返回
    if original_path in copied_images:
        return copied_images[original_path]

    original_path_obj = Path(original_path)

    # 检查原图片是否存在
    if not original_path_obj.exists():
        return None

    # 保持原文件名
    filename = original_path_obj.name
    new_path = target_dir / filename

    # 如果目标文件已存在
    if new_path.exists():
        # 检查是否是同一文件（通过hash）
        original_hash = get_file_hash(original_path_obj)
        existing_hash = get_file_hash(new_path)

        if original_hash == existing_hash:
            # 同一文件，直接返回
            copied_images[original_path] = str(new_path)
            return str(new_path)
        else:
            # 文件名冲突，添加编号
            stem = original_path_obj.stem
            suffix = original_path_obj.suffix
            counter = 1
            while new_path.exists():
                new_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

    # 复制文件
    try:
        shutil.copy2(original_path_obj, new_path)
        copied_images[original_path] = str(new_path)
        return str(new_path)
    except Exception as e:
        return None


def sample_from_original(file_config, dataset_name):
    """从原始数据集采样"""
    print(f"\n{'=' * 70}")
    print(f"阶段1 - 采样: {dataset_name.upper()}")
    print(f"{'=' * 70}")

    file_path = file_config['path']
    target_count = file_config['count']
    start_idx = file_config['start']

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return []

    print(f"文件: {file_path}")
    print(f"目标:  {target_count} 条")
    if start_idx > 0:
        print(f"起始: 第 {start_idx} 条")

    samples = []
    length_stats = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, desc=f"  读取{dataset_name}")):

            if line_idx < start_idx:
                continue

            if len(samples) >= target_count:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                text = sample.get('text', '')
                category = classify_length(text)

                samples.append(sample)
                length_stats[category] += 1

            except json.JSONDecodeError:
                continue

    print(f"\n✅ 采样完成: {len(samples)} 条")
    print(f"   长度分布:")
    for cat in ['3-6', '7-10', '>10']:
        count = length_stats[cat]
        pct = count / len(samples) * 100 if samples else 0
        print(f"     {cat}字:  {count:4d} ({pct:5.1f}%)")

    return samples


def extract_long_sentences(file_path, dataset_name, target_count, existing_ids=None):
    """从补充数据集中提取长句"""
    print(f"\n{'=' * 70}")
    print(f"阶段2 - 补充长句: {dataset_name.upper()}")
    print(f"{'=' * 70}")

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return []

    print(f"文件: {file_path}")
    print(f"目标: {target_count} 条 (>10字)")

    if existing_ids is None:
        existing_ids = set()

    # 收集候选
    candidates = []

    print(f"\n扫描候选...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"  扫描{dataset_name}"):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                text = sample.get('text', '')
                sample_id = sample.get('id', '')

                if len(text) > 10 and sample_id not in existing_ids:
                    candidates.append(sample)

            except json.JSONDecodeError:
                continue

    print(f"   找到 {len(candidates)} 条候选")

    # 采样
    if len(candidates) < target_count:
        print(f"   ⚠️  候选不足，只能采样 {len(candidates)} 条")
        sampled = candidates
    else:
        random.seed(RANDOM_SEED)
        sampled = random.sample(candidates, target_count)

    # 统计
    length_dist = Counter()
    for sample in sampled:
        length = len(sample['text'])
        if 11 <= length <= 15:
            length_dist['11-15'] += 1
        elif 16 <= length <= 20:
            length_dist['16-20'] += 1
        elif length > 20:
            length_dist['>20'] += 1

    print(f"\n✅ 补充完成: {len(sampled)} 条")
    print(f"   长度分布:")
    for cat in ['11-15', '16-20', '>20']:
        count = length_dist[cat]
        pct = count / len(sampled) * 100 if sampled else 0
        print(f"     {cat}字: {count: 4d} ({pct:5.1f}%)")

    return sampled


def process_images(samples, target_image_dir):
    """
    处理图片：检查、复制、更新路径

    Args:
        samples: 样本列表
        target_image_dir: 目标图片目录

    Returns:
        processed_samples: 处理后的样本列表
        stats: 统计信息
    """
    print(f"\n{'=' * 70}")
    print("处理图片")
    print(f"{'=' * 70}")

    target_dir = Path(target_image_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"目标目录:  {target_dir}")
    print(f"处理样本数: {len(samples)}")

    copied_images = {}
    processed_samples = []

    stats = {
        'total': len(samples),
        'success': 0,
        'image_not_found': 0,
        'copy_failed': 0,
        'unique_images': 0
    }

    print(f"\n复制图片...")
    for sample in tqdm(samples, desc="  处理"):
        # 获取图片路径
        original_image = sample.get('img', sample.get('image', ''))

        if not original_image:
            stats['image_not_found'] += 1
            continue

        # 检查并复制图片
        new_image_path = copy_image(original_image, target_dir, copied_images)

        if new_image_path:
            # 更新样本中的图片路径
            sample_copy = sample.copy()

            # 统一使用'img'字段
            if 'image' in sample_copy:
                del sample_copy['image']
            sample_copy['img'] = new_image_path

            processed_samples.append(sample_copy)
            stats['success'] += 1
        else:
            # 图片不存在或复制失败
            if not Path(original_image).exists():
                stats['image_not_found'] += 1
            else:
                stats['copy_failed'] += 1

    stats['unique_images'] = len(copied_images)

    # 打印统计
    print(f"\n✅ 图片处理完成")
    print(f"   成功:  {stats['success']} 条")
    print(f"   图片不存在: {stats['image_not_found']} 条")
    print(f"   复制失败: {stats['copy_failed']} 条")
    print(f"   实际复制图片数: {stats['unique_images']} 张")

    # 计算磁盘使用
    total_size = 0
    for img_path in target_dir.glob('*'):
        if img_path.is_file():
            total_size += img_path.stat().st_size

    print(f"   图片目录大小: {total_size / 1024 / 1024:.2f} MB")

    return processed_samples, stats


def merge_datasets():
    """主函数：合并数据集并处理图片"""
    print("=" * 70)
    print("数据集合并 + 图片处理")
    print("=" * 70)

    all_samples = []
    all_ids = set()

    # ============================================================
    # 阶段1：基础采样
    # ============================================================

    print("\n" + "=" * 70)
    print("【阶段1：基础采样】")
    print("=" * 70)

    for dataset_name, config in ORIGINAL_FILES.items():
        samples = sample_from_original(config, dataset_name)

        for sample in samples:
            sample_id = sample.get('id', '')
            if sample_id:
                all_ids.add(sample_id)

        all_samples.extend(samples)

    print(f"\n阶段1完成: 共 {len(all_samples)} 条")

    # ============================================================
    # 阶段2：补充长句
    # ============================================================

    print("\n" + "=" * 70)
    print("【阶段2：补充长句】")
    print("=" * 70)

    stage2_samples = []

    for dataset_name, config in SUPPLEMENT_FILES.items():
        long_samples = extract_long_sentences(
            config['path'],
            dataset_name,
            config['target_long'],
            existing_ids=all_ids
        )

        stage2_samples.extend(long_samples)

        for sample in long_samples:
            sample_id = sample.get('id', '')
            if sample_id:
                all_ids.add(sample_id)

    print(f"\n阶段2完成: 共补充 {len(stage2_samples)} 条长句")

    all_samples.extend(stage2_samples)

    print(f"\n合并完成: 共 {len(all_samples)} 条待处理样本")

    # ============================================================
    # 阶段3：处理图片
    # ============================================================

    print("\n" + "=" * 70)
    print("【阶段3：处理图片】")
    print("=" * 70)

    processed_samples, image_stats = process_images(all_samples, TARGET_IMAGE_DIR)

    # ============================================================
    # 最终统计
    # ============================================================

    print("\n" + "=" * 70)
    print("【最终统计】")
    print("=" * 70)

    total = len(processed_samples)
    print(f"\n有效样本数: {total}")

    # 长度分布
    final_length = Counter()
    for sample in processed_samples:
        text = sample.get('text', '')
        category = classify_length(text)
        final_length[category] += 1

    print(f"\n长度分布:")
    print(f"  {'类别':<10} {'数量': >8} {'占比':>8}")
    print(f"  {'-' * 10} {'-' * 8} {'-' * 8}")

    for cat in ['3-6', '7-10', '>10']:
        count = final_length[cat]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {cat: <10} {count:8d} {pct:7.1f}%")

    # 来源分布
    print(f"\n来源分布:")
    source_dist = Counter()
    for sample in processed_samples:
        img = sample.get('img', '')
        if 'coco' in img.lower():
            source_dist['coco'] += 1
        elif 'mmchat' in img.lower():
            source_dist['mmchat'] += 1
        elif 'wukong' in img.lower():
            source_dist['wukong'] += 1
        else:
            source_dist['unknown'] += 1

    for source in ['coco', 'mmchat', 'wukong', 'unknown']:
        count = source_dist[source]
        pct = count / total * 100 if total > 0 else 0
        print(f"  {source:<10}:  {count:5d} ({pct:5.1f}%)")

    # ============================================================
    # 保存
    # ============================================================

    print(f"\n{'=' * 70}")
    print("保存数据")
    print(f"{'=' * 70}")

    # 随机打乱
    random.seed(RANDOM_SEED)
    random.shuffle(processed_samples)

    # 确保输出目录存在
    output_dir = Path(OUTPUT_FILE).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for sample in processed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n✅ 数据已保存到: {OUTPUT_FILE}")
    print(f"   有效样本:  {total} 条")

    # 保存统计报告
    report_file = OUTPUT_FILE.replace('.jsonl', '_report.json')
    report = {
        'total_samples': total,
        'length_distribution': dict(final_length),
        'source_distribution': dict(source_dist),
        'image_stats': image_stats,
        'stage1_samples': len(all_samples) - len(stage2_samples),
        'stage2_samples': len(stage2_samples),
        'image_directory': str(TARGET_IMAGE_DIR)
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ 统计报告:  {report_file}")

    # 展示几个样本
    print(f"\n{'=' * 70}")
    print("样本示例（前3个）")
    print(f"{'=' * 70}")

    for i, sample in enumerate(processed_samples[:3], 1):
        print(f"\n{i}.ID: {sample.get('id', 'N/A')}")
        print(f"   文本: {sample.get('text', 'N/A')}")
        print(f"   长度: {len(sample.get('text', ''))} 字")
        print(f"   图片: {sample.get('img', 'N/A')}")

    print(f"\n{'=' * 70}")
    print("🎉 全部完成！")
    print(f"{'=' * 70}")

    print(f"\n输出文件:")
    print(f"  数据文件: {OUTPUT_FILE}")，通过长度和数据集的比例来创建
    print(f"  图片目录: {TARGET_IMAGE_DIR}")
    print(f"  统计报告: {report_file}")


if __name__ == '__main__':
    merge_datasets()