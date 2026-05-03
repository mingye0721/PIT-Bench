#!/usr/bin/env python3
# eval_task1_top3.py

"""
Task1 评估脚本（Top-3采样）
支持 JSON 和 JSONL 格式
"""

import json
from typing import List, Dict, Any
from difflib import SequenceMatcher
import os
import contextlib
from collections import defaultdict

# ============ 配置 ============

INPUT_FILE = "/root/autodl-tmp/bench/infer_baichuan/baichuan_top3_results.json"
WITH_IMG_FILE = "/root/autodl-tmp/bench/infer_llava/llava_top3_img_pro.json"
WITHOUT_IMG_FILE = "/root/autodl-tmp/bench/infer_llava/llava_top3_noimg_pro.json"
LOG_FILE = "/root/autodl-tmp/bench/infer_llava/llava_task1_metrics.txt"


# ============ 文件读取函数 ============

def load_json_or_jsonl(file_path: str) -> List[Dict]:
    """
    自动识别并加载 JSON 或 JSONL 文件

    Args:
        file_path: 文件路径

    Returns:
        包含所有数据的列表
    """
    data = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

        if not content:
            return data

        # 尝试作为 JSON 数组读取
        try:
            loaded = json.loads(content)
            if isinstance(loaded, list):
                data = loaded
                print(f"✓ 检测到 JSON 格式:  {file_path}")
                return data
            elif isinstance(loaded, dict):
                # 单个 JSON 对象
                data = [loaded]
                print(f"✓ 检测到单个 JSON 对象: {file_path}")
                return data
        except json.JSONDecodeError:
            pass

        # 尝试作为 JSONL 读取
        try:
            f.seek(0)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"⚠ 警告:  第 {line_num} 行解析失败: {e}")
                    continue

            if data:
                print(f"✓ 检测到 JSONL 格式: {file_path} ({len(data)} 行)")
                return data
        except Exception as e:
            print(f"✗ 读取文件失败: {file_path}")
            print(f"  错误: {e}")
            raise

    if not data:
        print(f"⚠ 警告: 文件为空或格式无法识别:  {file_path}")

    return data


# ============ 评估指标 ============


def exact_match(pred: str, gt: str) -> int:
    """完全匹配"""
    return 1 if pred == gt else 0


def char_accuracy(pred: str, gt: str) -> float:
    """字符级准确率"""
    if not gt:
        return 0.0

    matcher = SequenceMatcher(None, pred, gt)
    matches = sum(block.size for block in matcher.get_matching_blocks())

    return matches / len(gt)


def edit_distance(pred: str, gt: str) -> int:
    """编辑距离"""
    if not gt:
        return len(pred)
    if not pred:
        return len(gt)

    m, n = len(pred), len(gt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                )

    return dp[m][n]


def normalized_edit_distance(pred: str, gt: str) -> float:
    """标准化编辑距离"""
    if not gt:
        return 0.0

    dist = edit_distance(pred, gt)
    max_len = max(len(pred), len(gt))

    return 1.0 - (dist / max_len) if max_len > 0 else 0.0


def bleu_2gram(pred: str, gt: str) -> float:
    """2-gram BLEU"""
    if len(gt) < 2:
        return 1.0 if pred == gt else 0.0

    pred_bigrams = [pred[i:i + 2] for i in range(len(pred) - 1)]
    gt_bigrams = [gt[i:i + 2] for i in range(len(gt) - 1)]

    if not gt_bigrams:
        return 0.0

    matches = 0
    gt_bigrams_copy = gt_bigrams.copy()
    for bigram in pred_bigrams:
        if bigram in gt_bigrams_copy:
            matches += 1
            gt_bigrams_copy.remove(bigram)

    precision = matches / len(pred_bigrams) if pred_bigrams else 0.0
    recall = matches / len(gt_bigrams) if gt_bigrams else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============ Top-K 评估 ============


def top_k_metrics(preds: List[str], gt: str, k: int) -> Dict[str, float]:
    """Top-K评估"""
    if not preds:
        return {
            "exact_match": 0,
            "char_acc": 0.0,
            "edit_dist": len(gt),
            "norm_edit_dist": 0.0,
            "bleu_2gram": 0.0
        }

    top_k_preds = preds[:k]

    # 完全匹配：Top-K中是否命中
    em = 1 if any(pred == gt for pred in top_k_preds) else 0

    # 其他指标：取最好的
    char_accs = [char_accuracy(pred, gt) for pred in top_k_preds]
    edit_dists = [edit_distance(pred, gt) for pred in top_k_preds]
    norm_edit_dists = [normalized_edit_distance(pred, gt) for pred in top_k_preds]
    bleu_scores = [bleu_2gram(pred, gt) for pred in top_k_preds]

    return {
        "exact_match": em,
        "char_acc": max(char_accs),
        "edit_dist": min(edit_dists),
        "norm_edit_dist": max(norm_edit_dists),
        "bleu_2gram": max(bleu_scores)
    }


# ============ 多样性分析 ============


def diversity_metrics(preds: List[str]) -> Dict[str, float]:
    """分析Top-K生成的多样性"""
    if not preds:
        return {"unique_ratio": 0.0, "avg_pairwise_ed": 0.0}

    # 去重率
    unique_ratio = len(set(preds)) / len(preds)

    # 两两编辑距离
    if len(preds) < 2:
        avg_ed = 0.0
    else:
        total_ed = 0
        count = 0
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                total_ed += edit_distance(preds[i], preds[j])
                count += 1
        avg_ed = total_ed / count if count > 0 else 0.0

    return {
        "unique_ratio": unique_ratio,
        "avg_pairwise_ed": avg_ed
    }


# ============ 细粒度分析 ============


def analyze_by_length(samples: List[Dict]) -> Dict:
    """按文本长度分组分析"""

    grouped = {"short": [], "medium": [], "long": []}

    for s in samples:
        length = len(s['gt'])
        if length <= 6:  # 短文本 4-6字符
            grouped["short"].append(s)
        elif length <= 12:  # 中等 7-12字符
            grouped["medium"].append(s)
        else:  # 长文本 13-20字符
            grouped["long"].append(s)

    analysis = {}
    for category, samples_list in grouped.items():
        if not samples_list:
            continue

        # 计算Top-1指标
        top1_em = sum(top_k_metrics(s['predictions'], s['gt'], 1)['exact_match']
                      for s in samples_list) / len(samples_list) * 100

        top1_ca = sum(top_k_metrics(s['predictions'], s['gt'], 1)['char_acc']
                      for s in samples_list) / len(samples_list) * 100

        # 计算Top-3指标
        top3_hit = sum(top_k_metrics(s['predictions'], s['gt'], 3)['exact_match']
                       for s in samples_list) / len(samples_list) * 100

        top3_ca = sum(top_k_metrics(s['predictions'], s['gt'], 3)['char_acc']
                      for s in samples_list) / len(samples_list) * 100

        analysis[category] = {
            "count": len(samples_list),
            "top1_exact_match": top1_em,
            "top1_char_accuracy": top1_ca,
            "top3_hit_rate": top3_hit,
            "top3_best_char_accuracy": top3_ca
        }

    return analysis


def analyze_by_source(samples: List[Dict]) -> Dict:
    """按数据来源分组分析"""

    grouped = defaultdict(list)

    for s in samples:
        source = s.get('source', 'unknown')
        grouped[source].append(s)

    analysis = {}
    for source, samples_list in grouped.items():
        if not samples_list:
            continue

        # 计算Top-1指标
        top1_em = sum(top_k_metrics(s['predictions'], s['gt'], 1)['exact_match']
                      for s in samples_list) / len(samples_list) * 100

        top1_ca = sum(top_k_metrics(s['predictions'], s['gt'], 1)['char_acc']
                      for s in samples_list) / len(samples_list) * 100

        # 计算Top-3指标
        top3_hit = sum(top_k_metrics(s['predictions'], s['gt'], 3)['exact_match']
                       for s in samples_list) / len(samples_list) * 100

        top3_ca = sum(top_k_metrics(s['predictions'], s['gt'], 3)['char_acc']
                      for s in samples_list) / len(samples_list) * 100

        analysis[source] = {
            "count": len(samples_list),
            "top1_exact_match": top1_em,
            "top1_char_accuracy": top1_ca,
            "top3_hit_rate": top3_hit,
            "top3_best_char_accuracy": top3_ca
        }

    return analysis


def compute_vcs(acc_with: float, acc_without: float) -> float:
    """计算VCS"""
    if acc_without == 0:
        return 0.0
    return (acc_with - acc_without) / acc_without * 100


# ============ 评估单文件（支持细粒度分析）============


def eval_single(file: str, enable_fine_grained: bool = False) -> Dict[str, Any]:
    """评估单个文件（支持 JSON 和 JSONL）"""

    # 使用新的加载函数
    data_list = load_json_or_jsonl(file)

    if not data_list:
        print(f"⚠ 警告: 文件为空，无法评估:  {file}")
        return {
            "total_samples": 0,
            "error": "No data found in file"
        }

    n = 0
    samples = []  # 存储所有样本用于细粒度分析

    # Top-1指标
    top1_em, top1_ca, top1_ed, top1_ned, top1_bleu = 0, 0.0, 0.0, 0.0, 0.0

    # Top-3指标
    top3_em, top3_ca, top3_ed, top3_ned, top3_bleu = 0, 0.0, 0.0, 0.0, 0.0

    # 多样性
    total_unique_ratio = 0.0
    total_avg_ed = 0.0

    for d in data_list:
        gt = d.get("gt", "")
        preds = d.get("predictions", [])

        if not gt or not preds:
            print(f"⚠ 警告: 跳过无效数据项 (gt: {bool(gt)}, predictions: {len(preds)})")
            continue

        # 存储样本
        if enable_fine_grained:
            samples.append({
                'gt': gt,
                'predictions': preds,
                'source': d.get('source', 'unknown'),
                'img': d.get('img', '')
            })

        # Top-1
        m1 = top_k_metrics(preds, gt, 1)
        top1_em += m1["exact_match"]
        top1_ca += m1["char_acc"]
        top1_ed += m1["edit_dist"]
        top1_ned += m1["norm_edit_dist"]
        top1_bleu += m1["bleu_2gram"]

        # Top-3
        m3 = top_k_metrics(preds, gt, 3)
        top3_em += m3["exact_match"]
        top3_ca += m3["char_acc"]
        top3_ed += m3["edit_dist"]
        top3_ned += m3["norm_edit_dist"]
        top3_bleu += m3["bleu_2gram"]

        # 多样性
        div = diversity_metrics(preds[: 3])  # 只看Top-3
        total_unique_ratio += div["unique_ratio"]
        total_avg_ed += div["avg_pairwise_ed"]

        n += 1

    if n == 0:
        print(f"⚠ 警告: 没有有效样本可以评估")
        return {
            "total_samples": 0,
            "error": "No valid samples found"
        }

    result = {
        "total_samples": n,
        "top1": {
            "exact_match": top1_em / n * 100,
            "char_accuracy": top1_ca / n * 100,
            "avg_edit_distance": top1_ed / n,
            "norm_edit_distance": top1_ned / n * 100,
            "bleu_2gram": top1_bleu / n * 100,
        },
        "top3": {
            "hit_rate": top3_em / n * 100,
            "best_char_accuracy": top3_ca / n * 100,
            "best_edit_distance": top3_ed / n,
            "best_norm_edit_distance": top3_ned / n * 100,
            "best_bleu_2gram": top3_bleu / n * 100,
        },
        "diversity": {
            "unique_ratio": total_unique_ratio / n * 100,
            "avg_pairwise_edit_distance": total_avg_ed / n,
        }
    }

    # 添加细粒度分析
    if enable_fine_grained:
        result["by_length"] = analyze_by_length(samples)
        result["by_source"] = analyze_by_source(samples)

    return result


# ============ VGS计算（支持细粒度VCS）============


def eval_vgs(f_with: str, f_without: str) -> Dict[str, Any]:
    """计算VGS（包含细粒度VCS）"""
    m_with = eval_single(f_with, enable_fine_grained=True)
    m_without = eval_single(f_without, enable_fine_grained=True)

    def calc_vgs(val_with, val_without):
        if val_without == 0:
            return 0.0
        return (val_with - val_without) / val_without * 100

    vgs = {
        "top1": {
            "exact_match": calc_vgs(m_with["top1"]["exact_match"],
                                    m_without["top1"]["exact_match"]),
            "char_accuracy": calc_vgs(m_with["top1"]["char_accuracy"],
                                      m_without["top1"]["char_accuracy"]),
            "norm_edit_distance": calc_vgs(m_with["top1"]["norm_edit_distance"],
                                           m_without["top1"]["norm_edit_distance"]),
        },
        "top3": {
            "hit_rate": calc_vgs(m_with["top3"]["hit_rate"],
                                 m_without["top3"]["hit_rate"]),
            "best_char_accuracy": calc_vgs(m_with["top3"]["best_char_accuracy"],
                                           m_without["top3"]["best_char_accuracy"]),
        },
    }

    # 计算细粒度VCS
    fine_grained_vcs = {}

    # 按长度的VCS
    if "by_length" in m_with and "by_length" in m_without:
        fine_grained_vcs["by_length"] = {}
        for category in m_with["by_length"]:
            if category in m_without["by_length"]:
                fine_grained_vcs["by_length"][category] = {
                    "vcs_top1_em": calc_vgs(
                        m_with["by_length"][category]["top1_exact_match"],
                        m_without["by_length"][category]["top1_exact_match"]
                    ),
                    "vcs_top1_ca": calc_vgs(
                        m_with["by_length"][category]["top1_char_accuracy"],
                        m_without["by_length"][category]["top1_char_accuracy"]
                    ),
                    "vcs_top3_hit": calc_vgs(
                        m_with["by_length"][category]["top3_hit_rate"],
                        m_without["by_length"][category]["top3_hit_rate"]
                    ),
                    "vcs_top3_ca": calc_vgs(
                        m_with["by_length"][category]["top3_best_char_accuracy"],
                        m_without["by_length"][category]["top3_best_char_accuracy"]
                    )
                }

    # 按来源的VCS
    if "by_source" in m_with and "by_source" in m_without:
        fine_grained_vcs["by_source"] = {}
        for source in m_with["by_source"]:
            if source in m_without["by_source"]:
                fine_grained_vcs["by_source"][source] = {
                    "vcs_top1_em": calc_vgs(
                        m_with["by_source"][source]["top1_exact_match"],
                        m_without["by_source"][source]["top1_exact_match"]
                    ),
                    "vcs_top1_ca": calc_vgs(
                        m_with["by_source"][source]["top1_char_accuracy"],
                        m_without["by_source"][source]["top1_char_accuracy"]
                    ),
                    "vcs_top3_hit": calc_vgs(
                        m_with["by_source"][source]["top3_hit_rate"],
                        m_without["by_source"][source]["top3_hit_rate"]
                    ),
                    "vcs_top3_ca": calc_vgs(
                        m_with["by_source"][source]["top3_best_char_accuracy"],
                        m_without["by_source"][source]["top3_best_char_accuracy"]
                    )
                }

    return {
        "with_image": m_with,
        "without_image": m_without,
        "vgs": vgs,
        "fine_grained_vcs": fine_grained_vcs
    }


# ============ 打印结果（支持细粒度）============


def print_res(res: Dict[str, Any]):
    """打印结果"""

    if "with_image" in res:  # VGS模式
        w = res["with_image"]
        o = res["without_image"]
        v = res["vgs"]
        fv = res.get("fine_grained_vcs", {})

        print("\n" + "=" * 70)
        print("Task 1 评估结果（有图 vs 无图）- Top-3 Sampling")
        print("=" * 70)

        print("\n【有图像 (+I)】")
        print(f"  样本数: {w['total_samples']}")
        print(f"\n  Top-1 (单次生成):")
        print(f"    完全匹配 (Exact Match)     : {w['top1']['exact_match']: 6.2f}%")
        print(f"    字符准确率 (Char Acc)       : {w['top1']['char_accuracy']:6.2f}%")
        print(f"    标准化编辑距离 (Norm ED)    : {w['top1']['norm_edit_distance']:6.2f}%")
        print(f"    平均编辑距离 (Avg ED)       : {w['top1']['avg_edit_distance']:6.2f}")
        print(f"    BLEU-2gram                 : {w['top1']['bleu_2gram']:6.2f}%")

        print(f"\n  Top-3 (3次采样，取最佳):")
        print(f"    命中率 (Hit@3)             : {w['top3']['hit_rate']:6.2f}%")
        print(f"    最佳字符准确率             : {w['top3']['best_char_accuracy']:6.2f}%")
        print(f"    最佳标准化编辑距离         :  {w['top3']['best_norm_edit_distance']:6.2f}%")
        print(f"    最佳编辑距离               : {w['top3']['best_edit_distance']:6.2f}")

        print(f"\n  多样性 (Top-3):")
        print(f"    去重率 (Unique Ratio)      : {w['diversity']['unique_ratio']:6.2f}%")
        print(f"    平均两两编辑距离           : {w['diversity']['avg_pairwise_edit_distance']: 6.2f}")

        print("\n【无图像 (-I)】")
        print(f"  样本数: {o['total_samples']}")
        print(f"\n  Top-1:")
        print(f"    完全匹配                   : {o['top1']['exact_match']:6.2f}%")
        print(f"    字符准确率                 : {o['top1']['char_accuracy']:6.2f}%")
        print(f"\n  Top-3:")
        print(f"    命中率 (Hit@3)             : {o['top3']['hit_rate']:6.2f}%")
        print(f"    最佳字符准确率             : {o['top3']['best_char_accuracy']:6.2f}%")

        print("\n【VGS (Visual Grounding Score)】")
        print(f"  Top-1:")
        print(f"    完全匹配提升               :{v['top1']['exact_match']:+.2f}%")
        print(f"    字符准确率提升             :{v['top1']['char_accuracy']:+.2f}%")
        print(f"    标准化编辑距离提升         :{v['top1']['norm_edit_distance']:+.2f}%")
        print(f"  Top-3:")
        print(f"    命中率提升 (Hit@3)         :{v['top3']['hit_rate']:+.2f}%")
        print(f"    最佳字符准确率提升         :{v['top3']['best_char_accuracy']:+.2f}%")

        # 打印细粒度分析（按文本长度）
        if "by_length" in w:
            print(f"\n" + "━" * 70)
            print("【细粒度分析 - 按文本长度】")
            print("━" * 70)

            for category, data in sorted(w["by_length"].items()):
                print(f"\n  {category} ({data['count']} 样本):")

                # 有图 (+I)
                print(f"    有图 (+I):")
                print(f"      Top-1 EM: {data['top1_exact_match']:5.2f}%  |  "
                      f"Top-1 CA: {data['top1_char_accuracy']: 5.2f}%")
                print(f"      Top-3 Hit: {data['top3_hit_rate']:5.2f}%  |  "
                      f"Top-3 CA:  {data['top3_best_char_accuracy']:5.2f}%")

                # 无图 (-I)
                if category in o.get("by_length", {}):
                    o_data = o["by_length"][category]
                    print(f"    无图 (-I):")
                    print(f"      Top-1 EM: {o_data['top1_exact_match']:5.2f}%  |  "
                          f"Top-1 CA: {o_data['top1_char_accuracy']:5.2f}%")
                    print(f"      Top-3 Hit: {o_data['top3_hit_rate']:5.2f}%  |  "
                          f"Top-3 CA: {o_data['top3_best_char_accuracy']:5.2f}%")

                    # VCS
                    if "by_length" in fv and category in fv["by_length"]:
                        vcs_data = fv["by_length"][category]
                        print(f"    VCS:")
                        print(f"      Top-1 EM: {vcs_data['vcs_top1_em']:+6.2f}%  |  "
                              f"Top-1 CA: {vcs_data['vcs_top1_ca']:+6.2f}%")
                        print(f"      Top-3 Hit:  {vcs_data['vcs_top3_hit']:+6.2f}%  |  "
                              f"Top-3 CA: {vcs_data['vcs_top3_ca']:+6.2f}%")

    else:  # 单文件模式
        print("\n" + "=" * 70)
        print("Task 1 评估结果 - Top-3 Sampling")
        print("=" * 70)

        print(f"\n样本数: {res['total_samples']}")

        print(f"\n【Top-1 指标（单次生成）】")
        print(f"  完全匹配 (Exact Match)     : {res['top1']['exact_match']:6.2f}%")
        print(f"  字符准确率 (Char Acc)       : {res['top1']['char_accuracy']:6.2f}%")
        print(f"  标准化编辑距离 (Norm ED)    : {res['top1']['norm_edit_distance']:6.2f}%")
        print(f"  平均编辑距离 (Avg ED)       : {res['top1']['avg_edit_distance']:6.2f}")
        print(f"  BLEU-2gram                 : {res['top1']['bleu_2gram']:6.2f}%")

        print(f"\n【Top-3 指标（3次采样，取最佳）】")
        print(f"  命中率 (Hit@3)             : {res['top3']['hit_rate']:6.2f}%")
        print(f"  最佳字符准确率             : {res['top3']['best_char_accuracy']: 6.2f}%")
        print(f"  最佳标准化编辑距离         : {res['top3']['best_norm_edit_distance']:6.2f}%")
        print(f"  最佳编辑距离               : {res['top3']['best_edit_distance']:6.2f}")
        print(f"  最佳BLEU-2gram             : {res['top3']['best_bleu_2gram']:6.2f}%")

        print(f"\n【多样性分析 (Top-3)】")
        print(f"  去重率 (Unique Ratio)      : {res['diversity']['unique_ratio']:6.2f}%")
        print(f"  平均两两编辑距离           : {res['diversity']['avg_pairwise_edit_distance']:6.2f}")

        # 单文件也支持细粒度分析
        if "by_length" in res:
            print(f"\n" + "━" * 70)
            print("【细粒度分析 - 按文本长度】")
            print("━" * 70)

            for category, data in sorted(res["by_length"].items()):
                print(f"\n  {category} ({data['count']} 样本):")
                print(f"    Top-1 EM: {data['top1_exact_match']: 5.2f}%  |  "
                      f"Top-1 CA: {data['top1_char_accuracy']:5.2f}%")
                print(f"    Top-3 Hit: {data['top3_hit_rate']:5.2f}%  |  "
                      f"Top-3 CA: {data['top3_best_char_accuracy']:5.2f}%")

        if "by_source" in res:
            print(f"\n" + "━" * 70)
            print("【细粒度分析 - 按数据来源】")
            print("━" * 70)

            for source, data in sorted(res["by_source"].items()):
                print(f"\n  {source} ({data['count']} 样本):")
                print(f"    Top-1 EM: {data['top1_exact_match']:5.2f}%  |  "
                      f"Top-1 CA: {data['top1_char_accuracy']:5.2f}%")
                print(f"    Top-3 Hit: {data['top3_hit_rate']:5.2f}%  |  "
                      f"Top-3 CA: {data['top3_best_char_accuracy']:5.2f}%")

    print("=" * 70)


# ============ 主函数 ============


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Task1 评估脚本（支持 JSON 和 JSONL 格式）")
    print("=" * 70)

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if WITH_IMG_FILE and WITHOUT_IMG_FILE:
        print(f"\n模式: VGS 评估（有图 vs 无图）")
        print(f"有图文件: {WITH_IMG_FILE}")
        print(f"无图文件: {WITHOUT_IMG_FILE}")
        result = eval_vgs(WITH_IMG_FILE, WITHOUT_IMG_FILE)
    else:
        print(f"\n模式: 单文件评估")
        print(f"输入文件: {INPUT_FILE}")
        result = eval_single(INPUT_FILE, enable_fine_grained=True)

    # 打印到屏幕
    print_res(result)

    # 保存到文件
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        import sys

        old_stdout = sys.stdout
        sys.stdout = f
        print_res(result)
        sys.stdout = old_stdout

    # 保存JSON数据
    json_output = LOG_FILE.replace('.txt', '_data.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评估完成")
    print(f"   日志文件: {LOG_FILE}")
    print(f"   数据文件: {json_output}")