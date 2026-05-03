#!/usr/bin/env python3
# eval_task2_clean.py

"""
Task 2 评估脚本（使用预处理后的数据）
"""

import json
from typing import List, Dict, Any
from difflib import SequenceMatcher
from collections import defaultdict
import os

# ============================================================
# 配置
# ============================================================

INPUT_FILE = "/root/autodl-tmp/bench/llava/llava_task2_results.jsonl"
OUTPUT_REPORT = "/root/autodl-tmp/bench/infer_llava/task2_evaluation_report.txt"
OUTPUT_JSON = "/root/autodl-tmp/bench/infer_llava/task2_evaluation_data.json"


# ============================================================
# 评估指标
# ============================================================


def contains_target_word(pred_text: str, target_word: str) -> bool:
    """判断预测文本中是否包含目标词"""
    return target_word in pred_text


def exact_match(pred_text: str, gt_text: str) -> bool:
    """完全匹配"""
    return pred_text == gt_text


def char_accuracy(pred_text: str, gt_text: str) -> float:
    """字符级准确率"""
    if not gt_text:
        return 0.0

    matcher = SequenceMatcher(None, pred_text, gt_text)
    matches = sum(block.size for block in matcher.get_matching_blocks())

    return matches / len(gt_text)


def position_accuracy(results: List[Dict], condition: str) -> float:
    """位置准确率:  目标词位置是否正确"""
    correct = 0
    total = 0

    pred_key = f"pred_{condition}"

    for r in results:
        target_word = r['target_word']
        pred = r[pred_key]  # ← 已经清理过，直接使用

        if contains_target_word(pred, target_word):
            correct += 1

        total += 1

    return correct / total * 100 if total > 0 else 0.0


def sentence_accuracy(results: List[Dict], condition: str) -> float:
    """句子级准确率"""
    correct = 0
    total = 0

    pred_key = f"pred_{condition}"

    for r in results:
        pred = r[pred_key]
        if exact_match(pred, r['text_gt']):
            correct += 1
        total += 1

    return correct / total * 100 if total > 0 else 0.0


def avg_char_accuracy(results: List[Dict], condition: str) -> float:
    """平均字符准确率"""
    total_acc = 0.0

    pred_key = f"pred_{condition}"

    for r in results:
        pred = r[pred_key]
        total_acc += char_accuracy(pred, r['text_gt'])

    return total_acc / len(results) * 100 if results else 0.0


# ============================================================
# VCS计算
# ============================================================


def compute_vcs(acc_with: float, acc_without: float) -> float:
    """计算VCS"""
    if acc_without == 0:
        return 0.0
    return (acc_with - acc_without) / acc_without * 100


# ============================================================
# 细粒度分析
# ============================================================


def analyze_by_candidates(results: List[Dict]) -> Dict:
    """按候选词数量分组分析"""

    # 统计每个group的词数
    group_words = defaultdict(set)
    for r in results:
        gid = r.get('group_id')
        if gid is not None:
            group_words[gid].add(r['target_word'])

    group_to_num_words = {
        gid: len(words)
        for gid, words in group_words.items()
    }

    # 按候选词数分组
    grouped = defaultdict(list)
    for r in results:
        gid = r.get('group_id')
        if gid is None:
            continue

        num_words = group_to_num_words.get(gid, 0)
        if num_words == 0:
            continue

        grouped[num_words].append(r)

    # 计算各组指标
    analysis = {}
    for num_words in sorted(grouped.keys()):
        samples = grouped[num_words]

        acc_with = position_accuracy(samples, 'with_img')
        acc_no = position_accuracy(samples, 'no_img')
        acc_wrong = position_accuracy(samples, 'wrong_img')

        analysis[f"{num_words}_candidates"] = {
            "count": len(samples),
            "acc_with_img": acc_with,
            "acc_no_img": acc_no,
            "acc_wrong_img": acc_wrong,
            "vcs": compute_vcs(acc_with, acc_no),
            "robustness": acc_wrong / acc_no * 100 if acc_no > 0 else 0
        }

    return analysis


def analyze_by_length(results: List[Dict]) -> Dict:
    """按文本长度分组分析"""

    grouped = {"short": [], "medium": [], "long": []}

    for r in results:
        length = len(r['text_gt'])
        if length <= 8:
            grouped["short"].append(r)
        elif length <= 15:
            grouped["medium"].append(r)
        else:
            grouped["long"].append(r)

    analysis = {}
    for category, samples in grouped.items():
        if not samples:
            continue

        acc_with = position_accuracy(samples, 'with_img')
        acc_no = position_accuracy(samples, 'no_img')

        analysis[category] = {
            "count": len(samples),
            "acc_with_img": acc_with,
            "acc_no_img": acc_no,
            "vcs": compute_vcs(acc_with, acc_no)
        }

    return analysis


# ============================================================
# 错误分析
# ============================================================


def analyze_errors(results: List[Dict]) -> Dict:
    """分析错误类型"""

    error_types = {
        "visual_misled": 0,  # 错误图像误导
        "no_visual_help": 0,  # 正确图像也无帮助
        "visual_corrected": 0,  # 图像纠正了错误
        "context_insufficient": 0  # 上下文不足
    }

    for r in results:
        target = r['target_word']

        pred_with = r['pred_with_img']
        pred_no = r['pred_no_img']
        pred_wrong = r['pred_wrong_img']

        with_correct = contains_target_word(pred_with, target)
        no_correct = contains_target_word(pred_no, target)
        wrong_correct = contains_target_word(pred_wrong, target)

        # 错误图像误导
        if with_correct and not wrong_correct:
            error_types["visual_misled"] += 1

        # 正确图像也无帮助
        if not with_correct and not no_correct:
            error_types["no_visual_help"] += 1

        # 图像纠正了错误
        if with_correct and not no_correct:
            error_types["visual_corrected"] += 1

        # 上下文不足（无图时错误）
        if not no_correct:
            error_types["context_insufficient"] += 1

    total = len(results)

    return {
        k: {"count": v, "percentage": v / total * 100 if total > 0 else 0}
        for k, v in error_types.items()
    }


# ============================================================
# 主评估函数
# ============================================================


def evaluate_task2(result_file: str) -> Dict[str, Any]:
    """完整评估流程"""

    print("\n" + "=" * 70)
    print("Task 2 评估")
    print("=" * 70)

    # 读取结果
    print(f"\n📖 读取推理结果: {result_file}")
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"   共 {len(results)} 个样本")

    # 主要指标
    print(f"\n📊 计算主要指标...")

    pos_acc_with = position_accuracy(results, 'with_img')
    pos_acc_no = position_accuracy(results, 'no_img')
    pos_acc_wrong = position_accuracy(results, 'wrong_img')

    sent_acc_with = sentence_accuracy(results, 'with_img')
    sent_acc_no = sentence_accuracy(results, 'no_img')

    char_acc_with = avg_char_accuracy(results, 'with_img')
    char_acc_no = avg_char_accuracy(results, 'no_img')

    vcs = compute_vcs(pos_acc_with, pos_acc_no)
    robustness = pos_acc_wrong / pos_acc_no * 100 if pos_acc_no > 0 else 0

    # 细粒度分析
    print(f"📊 细粒度分析...")

    by_candidates = analyze_by_candidates(results)
    by_length = analyze_by_length(results)
    error_analysis = analyze_errors(results)

    # 汇总结果
    evaluation = {
        "total_samples": len(results),

        "position_accuracy": {
            "with_img": pos_acc_with,
            "no_img": pos_acc_no,
            "wrong_img": pos_acc_wrong
        },

        "sentence_accuracy": {
            "with_img": sent_acc_with,
            "no_img": sent_acc_no
        },

        "char_accuracy": {
            "with_img": char_acc_with,
            "no_img": char_acc_no
        },

        "vcs": vcs,
        "robustness": robustness,

        "by_candidates": by_candidates,
        "by_length": by_length,
        "error_analysis": error_analysis
    }

    return evaluation


# ============================================================
# 打印报告
# ============================================================


def print_report(evaluation: Dict[str, Any]):
    """打印评估报告"""

    print("\n" + "=" * 70)
    print("Task 2:  同音词消歧评估结果")
    print("=" * 70)

    print(f"\n总样本数: {evaluation['total_samples']}")

    print(f"\n【位置准确率 (Position Accuracy)】")
    print(f"  匹配图像 (+I)          :{evaluation['position_accuracy']['with_img']:6.2f}%")
    print(f"  无图像 (-I)            : {evaluation['position_accuracy']['no_img']:6.2f}%")
    print(f"  错误图像 (-I_wrong)    : {evaluation['position_accuracy']['wrong_img']:6.2f}%")

    print(f"\n【句子准确率 (Sentence Accuracy)】")
    print(f"  匹配图像 (+I)          :{evaluation['sentence_accuracy']['with_img']:6.2f}%")
    print(f"  无图像 (-I)            :{evaluation['sentence_accuracy']['no_img']:6.2f}%")

    print(f"\n【字符准确率 (Character Accuracy)】")
    print(f"  匹配图像 (+I)          :{evaluation['char_accuracy']['with_img']:6.2f}%")
    print(f"  无图像 (-I)            :{evaluation['char_accuracy']['no_img']:6.2f}%")

    print(f"\n【视觉贡献分数 (VCS)】")
    print(f"  相对提升               :{evaluation['vcs']:+.2f}%")
    abs_gain = evaluation['position_accuracy']['with_img'] - evaluation['position_accuracy']['no_img']
    print(f"  绝对提升               :{abs_gain:+.2f} percentage points")

    print(f"\n【对抗鲁棒性】")
    print(f"  鲁棒性得分             :{evaluation['robustness']:.2f}%")

    if evaluation['by_candidates']:
        print(f"\n" + "━" * 70)
        print("【细粒度分析 - 按候选词数量】")
        print("━" * 70)

        for key, data in sorted(evaluation['by_candidates'].items()):
            print(f"\n  {key} ({data['count']} 样本):")
            print(
                f"    +I:{data['acc_with_img']:5.2f}%  |  -I: {data['acc_no_img']:5.2f}%  |  VCS: {data['vcs']:+.2f}%")
            print(f"    鲁棒性: {data['robustness']:.2f}%")

    if evaluation['by_length']:
        print(f"\n" + "━" * 70)
        print("【细粒度分析 - 按文本长度】")
        print("━" * 70)

        for key, data in evaluation['by_length'].items():
            print(f"\n  {key} ({data['count']} 样本):")
            print(
                f"    +I: {data['acc_with_img']: 5.2f}%  |  -I: {data['acc_no_img']:5.2f}%  |  VCS: {data['vcs']:+.2f}%")

    print(f"\n" + "━" * 70)
    print("【错误分析】")
    print("━" * 70)

    err = evaluation['error_analysis']
    print(
        f"\n  视觉误导 (错误图像导致错误)     : {err['visual_misled']['count']} ({err['visual_misled']['percentage']:.1f}%)")
    print(
        f"  视觉无效 (正确图像也无帮助)     : {err['no_visual_help']['count']} ({err['no_visual_help']['percentage']:.1f}%)")
    print(
        f"  视觉纠正 (图像纠正了文本错误)   : {err['visual_corrected']['count']} ({err['visual_corrected']['percentage']:.1f}%)")
    print(
        f"  上下文不足 (无图时错误)         : {err['context_insufficient']['count']} ({err['context_insufficient']['percentage']:.1f}%)")

    print("\n" + "=" * 70)


# ============================================================
# 主函数
# ============================================================


def main():
    # 评估
    evaluation = evaluate_task2(INPUT_FILE)

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