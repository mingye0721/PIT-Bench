#!/usr/bin/env python3
# eval_task3.py

"""
Task 3 评估脚本：拼音错误纠正（输出文本）
支持有图和无图的对比评估 + 按错误类型计算VCS
"""

import json
from typing import List, Dict, Any
from difflib import SequenceMatcher
from collections import defaultdict
import os

# ============================================================
# 配置
# ============================================================

# ✅ 分别读取有图和无图的预处理结果
INPUT_WITH_IMG = "/root/autodl-tmp/bench/infer_llava/task3/llava_task3_with_img.jsonl"
INPUT_NO_IMG = "/root/autodl-tmp/bench/infer_llava/task3/llava_task3_no_img.jsonl"

OUTPUT_REPORT = "/root/autodl-tmp/bench/infer_llava/llama/task3_eval.txt"
OUTPUT_JSON = "/root/autodl-tmp/bench/infer_llava/task3_evaluation_data.json"


# ============================================================
# 评估指标（与Task 1相同）
# ============================================================


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


# ============================================================
# 读取数据
# ============================================================


def load_results(file_path: str) -> List[Dict]:
    """读取预处理后的结果"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


# ============================================================
# 评估单个条件
# ============================================================


def evaluate_single_condition(results: List[Dict]) -> Dict[str, Any]:
    """评估单个条件（有图或无图）"""

    n = len(results)

    # 累加指标
    total_em = 0
    total_ca = 0.0
    total_ed = 0.0
    total_ned = 0.0
    total_bleu = 0.0

    # 按错误类型分组
    by_error_type = defaultdict(lambda: {
        'count': 0,
        'em': 0,
        'ca': 0.0,
        'ed': 0.0,
        'ned': 0.0
    })

    # 按错误强度分组
    by_intensity = defaultdict(lambda: {
        'count': 0,
        'em': 0,
        'ca': 0.0,
        'ed': 0.0,
        'ned': 0.0
    })

    for r in results:
        text_gt = r['text_gt']
        pred = r.get('pred', '')

        # 主要指标
        em = exact_match(pred, text_gt)
        ca = char_accuracy(pred, text_gt)
        ed = edit_distance(pred, text_gt)
        ned = normalized_edit_distance(pred, text_gt)
        bleu = bleu_2gram(pred, text_gt)

        total_em += em
        total_ca += ca
        total_ed += ed
        total_ned += ned
        total_bleu += bleu

        # 按错误类型
        error_type = r.get('error_type', 'unknown')
        by_error_type[error_type]['count'] += 1
        by_error_type[error_type]['em'] += em
        by_error_type[error_type]['ca'] += ca
        by_error_type[error_type]['ed'] += ed
        by_error_type[error_type]['ned'] += ned

        # 按错误强度
        intensity = r.get('error_intensity', 'unknown')
        by_intensity[intensity]['count'] += 1
        by_intensity[intensity]['em'] += em
        by_intensity[intensity]['ca'] += ca
        by_intensity[intensity]['ed'] += ed
        by_intensity[intensity]['ned'] += ned

    # 计算平均值
    def finalize_group(group_dict):
        result = {}
        for key, data in group_dict.items():
            count = data['count']
            if count == 0:
                continue
            result[key] = {
                'count': count,
                'sentence_accuracy': data['em'] / count * 100,
                'char_accuracy': data['ca'] / count * 100,
                'avg_edit_distance': data['ed'] / count,
                'norm_edit_distance': data['ned'] / count * 100
            }
        return result

    return {
        'total_samples': n,
        'sentence_accuracy': total_em / n * 100,
        'char_accuracy': total_ca / n * 100,
        'avg_edit_distance': total_ed / n,
        'norm_edit_distance': total_ned / n * 100,
        'bleu_2gram': total_bleu / n * 100,
        'by_error_type': finalize_group(by_error_type),
        'by_intensity': finalize_group(by_intensity)
    }


# ============================================================
# VCS计算
# ============================================================


def compute_vcs(acc_with: float, acc_without: float) -> float:
    """计算VCS (Visual Contribution Score)

    VCS = (有图准确率 - 无图准确率) / 无图准确率 × 100%
    """
    if acc_without == 0:
        return 0.0
    return (acc_with - acc_without) / acc_without * 100


def compute_vcs_by_group(eval_with: Dict, eval_no: Dict, group_key: str) -> Dict[str, Dict]:
    """计算按错误类型或强度的VCS

    Args:
        eval_with: 有图的评估结果
        eval_no: 无图的评估结果
        group_key: 'by_error_type' 或 'by_intensity'

    Returns:
        {
            'sound':  {'sa_vcs': +58.4, 'ca_vcs':  +21.2, ...  },
            'keyboard': {...  },
            ...
        }
    """
    vcs_result = {}

    groups_with = eval_with.get(group_key, {})
    groups_no = eval_no.get(group_key, {})

    # 找到所有类型（有图和无图的并集）
    all_types = set(groups_with.keys()) | set(groups_no.keys())

    for error_type in all_types:
        data_with = groups_with.get(error_type, {})
        data_no = groups_no.get(error_type, {})

        # 如果某一组缺失数据，跳过
        if not data_with or not data_no:
            continue

        vcs_result[error_type] = {
            'count_with': data_with['count'],
            'count_no': data_no['count'],
            'sa_with': data_with['sentence_accuracy'],
            'sa_no': data_no['sentence_accuracy'],
            'sa_vcs': compute_vcs(
                data_with['sentence_accuracy'],
                data_no['sentence_accuracy']
            ),
            'ca_with': data_with['char_accuracy'],
            'ca_no': data_no['char_accuracy'],
            'ca_vcs': compute_vcs(
                data_with['char_accuracy'],
                data_no['char_accuracy']
            ),
            'ed_with': data_with['avg_edit_distance'],
            'ed_no': data_no['avg_edit_distance'],
            # 编辑距离的VCS是相反的（越小越好）
            'ed_vcs': compute_vcs(
                data_no['avg_edit_distance'],  # 注意：这里反过来
                data_with['avg_edit_distance']
            ) * -1,  # 然后取负
        }

    return vcs_result


# ============================================================
# 主评估函数
# ============================================================


def evaluate_task3() -> Dict[str, Any]:
    """完整评估流程"""

    print("\n" + "=" * 70)
    print("Task 3 评估（拼音纠错转写）")
    print("=" * 70)

    # 读取结果
    print(f"\n📖 读取推理结果...")
    print(f"   有图: {INPUT_WITH_IMG}")
    print(f"   无图: {INPUT_NO_IMG}")

    results_with = load_results(INPUT_WITH_IMG)
    results_no = load_results(INPUT_NO_IMG)

    print(f"   ✅ 有图样本数: {len(results_with)}")
    print(f"   ✅ 无图样本数: {len(results_no)}")

    if len(results_with) != len(results_no):
        print(f"   ⚠️ 警告：有图和无图样本数不一致！")

    # 评估两个条件
    print(f"\n📊 计算指标...")

    eval_with = evaluate_single_condition(results_with)
    eval_no = evaluate_single_condition(results_no)

    # ✅ 计算整体VCS（包含多个指标）
    vcs_overall = {
        'sentence_accuracy': {
            'relative': compute_vcs(
                eval_with['sentence_accuracy'],
                eval_no['sentence_accuracy']
            ),
            'absolute': eval_with['sentence_accuracy'] - eval_no['sentence_accuracy']
        },
        'char_accuracy': {
            'relative': compute_vcs(
                eval_with['char_accuracy'],
                eval_no['char_accuracy']
            ),
            'absolute': eval_with['char_accuracy'] - eval_no['char_accuracy']
        }
    }

    # ✅ 计算按错误类型的VCS
    vcs_by_error_type = compute_vcs_by_group(eval_with, eval_no, 'by_error_type')

    # ✅ 计算按错误强度的VCS
    vcs_by_intensity = compute_vcs_by_group(eval_with, eval_no, 'by_intensity')

    return {
        'with_image': eval_with,
        'without_image': eval_no,
        'vcs_overall': vcs_overall,
        'vcs_by_error_type': vcs_by_error_type,
        'vcs_by_intensity': vcs_by_intensity
    }


# ============================================================
# 打印报告
# ============================================================


def print_report(evaluation: Dict[str, Any]):
    """打印评估报告"""

    print("\n" + "=" * 70)
    print("Task 3: 拼音错误纠正评估结果")
    print("=" * 70)

    w = evaluation['with_image']
    o = evaluation['without_image']
    vcs = evaluation['vcs_overall']

    print(f"\n总样本数: {w['total_samples']}")

    print(f"\n【句子准确率 (Sentence Accuracy)】")
    print(f"  匹配图像 (+I)              : {w['sentence_accuracy']: 6.2f}%")
    print(f"  无图像 (-I)                : {o['sentence_accuracy']:6.2f}%")

    print(f"\n【字符准确率 (Character Accuracy)】")
    print(f"  匹配图像 (+I)              : {w['char_accuracy']:6.2f}%")
    print(f"  无图像 (-I)                : {o['char_accuracy']: 6.2f}%")

    print(f"\n【其他指标 (有图)】")
    print(f"  平均编辑距离 (Avg ED)      : {w['avg_edit_distance']:6.2f}")
    print(f"  标准化编辑距离 (NED)       : {w['norm_edit_distance']:6.2f}%")
    print(f"  BLEU-2gram                 : {w['bleu_2gram']:6.2f}%")

    print(f"\n【视觉贡献分数 (VCS)】")
    print(f"  句子准确率:")
    print(f"    相对提升                 : {vcs['sentence_accuracy']['relative']:+.2f}%")
    print(f"    绝对提升                 : {vcs['sentence_accuracy']['absolute']:+.2f} pp")
    print(f"  字符准确率:")
    print(f"    相对提升                 : {vcs['char_accuracy']['relative']:+.2f}%")
    print(f"    绝对提升                 : {vcs['char_accuracy']['absolute']:+.2f} pp")

    # 按错误类型
    if w['by_error_type']:
        print(f"\n" + "━" * 70)
        print("【细粒度分析 - 按错误类型】")
        print("━" * 70)

        for error_type, data in sorted(w['by_error_type'].items()):
            print(f"\n  {error_type} ({data['count']} 样本):")
            print(f"    有图 (+I):")
            print(f"      SA: {data['sentence_accuracy']:5.2f}%  |  "
                  f"CA: {data['char_accuracy']:5.2f}%  |  "
                  f"Avg ED: {data['avg_edit_distance']:5.2f}")

    # ✅ 按错误类型的VCS
    if 'vcs_by_error_type' in evaluation and evaluation['vcs_by_error_type']:
        print(f"\n" + "━" * 70)
        print("【VCS分析 - 按错误类型】")
        print("━" * 70)

        vcs_types = evaluation['vcs_by_error_type']

        # 按SA的VCS排序（从高到低）
        sorted_types = sorted(
            vcs_types.items(),
            key=lambda x: x[1]['sa_vcs'],
            reverse=True
        )

        for error_type, vcs_data in sorted_types:
            print(f"\n  {error_type} ({vcs_data['count_with']} 样本):")
            print(f"    +I: {vcs_data['sa_with']:5.2f}%  |  "
                  f"-I: {vcs_data['sa_no']:5.2f}%  |  "
                  f"VCS (SA): {vcs_data['sa_vcs']:+6.2f}%")
            print(f"    CA VCS: {vcs_data['ca_vcs']:+6.2f}%")

    # 按错误强度
    if w['by_intensity']:
        print(f"\n" + "━" * 70)
        print("【细粒度分析 - 按错误强度】")
        print("━" * 70)

        for intensity, data in sorted(w['by_intensity'].items()):
            print(f"\n  {intensity} ({data['count']} 样本):")
            print(f"    SA: {data['sentence_accuracy']:5.2f}%  |  "
                  f"CA: {data['char_accuracy']:5.2f}%  |  "
                  f"NED: {data['norm_edit_distance']:5.2f}%")

    # ✅ 按错误强度的VCS
    if 'vcs_by_intensity' in evaluation and evaluation['vcs_by_intensity']:
        print(f"\n" + "━" * 70)
        print("【VCS分析 - 按错误强度】")
        print("━" * 70)

        vcs_intensities = evaluation['vcs_by_intensity']

        # 按强度名称排序（light < medium < heavy）
        intensity_order = {'light': 1, 'medium': 2, 'heavy': 3}
        sorted_intensities = sorted(
            vcs_intensities.items(),
            key=lambda x: intensity_order.get(x[0], 99)
        )

        for intensity, vcs_data in sorted_intensities:
            print(f"\n  {intensity} ({vcs_data['count_with']} 样本):")
            print(f"    +I: {vcs_data['sa_with']:5.2f}%  |  "
                  f"-I: {vcs_data['sa_no']:5.2f}%  |  "
                  f"VCS (SA): {vcs_data['sa_vcs']:+6.2f}%")
            print(f"    CA VCS: {vcs_data['ca_vcs']:+6.2f}%")

    print("\n" + "=" * 70)


# ============================================================
# 主函数
# ============================================================


def main():
    # 评估
    evaluation = evaluate_task3()

    # 打印报告
    print_report(evaluation)

    # 保存结果
    print(f"\n💾 保存评估结果...")

    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)

    # 保存文本报告
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        print_report(evaluation)
        sys.stdout = old_stdout

    # 保存JSON数据
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(f"   ✅ 报告:  {OUTPUT_REPORT}")
    print(f"   ✅ 数据: {OUTPUT_JSON}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()