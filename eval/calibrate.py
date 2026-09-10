#!/usr/bin/env python3
"""置信度校准（保序回归）+ 贝叶斯闸门决策。

背景：rank 阶段吐出的分数不是概率，只是排序用的打分——0.8 分不代表 80% 会对。
要把「自动通过 / 转人工确认」做成可解释的决策，得先把分数映射成 P(正确)，
再按错送代价 c_fa 和误拦代价 c_fr 算出该在哪个概率上画线（评测体系设计 §2.6）。

    分数 --PAV--> 校准表 --predict--> P(正确) --decision_threshold(c_fa, c_fr)--> 通过/确认

全部只用标准库；PAV 是保序回归里最简单的一种，不需要 sklearn。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from stats import ece  # noqa: E402


def pav(xs: list[float], ys: list[float]) -> list[tuple[float, float]]:
    """保序回归（pool-adjacent-violators）。

    输入 (score, 0/1 是否正确)，输出单调不减的分段常数表 [(x_start, p), ...]（x_start 升序）。
    score >= 该段起点且 < 下一段起点时映射为 p；score 落在最后一段之后也用最后一段的 p。

    做法：先按 score 排序、同分的样本合并成一个点（同分同段），
    然后从左到右扫，一旦相邻两段的均值不是严格递增就把它们池化（加权平均），
    池化可能连锁触发更早的段——用一个栈不断往前收敛即可。
    """
    if not xs:
        return []
    if len(xs) != len(ys):
        raise ValueError(f"xs 与 ys 长度不一致：{len(xs)} vs {len(ys)}")

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    merged: list[list[float]] = []  # [x, sum_y, weight]
    for i in order:
        x, y = xs[i], float(ys[i])
        if merged and merged[-1][0] == x:
            merged[-1][1] += y
            merged[-1][2] += 1
        else:
            merged.append([x, y, 1])

    blocks: list[list[float]] = []  # [x_start, sum_y, weight]
    for x, sum_y, w in merged:
        blocks.append([x, sum_y, w])
        while len(blocks) >= 2 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]:
            _, sum2, w2 = blocks.pop()
            blocks[-1][1] += sum2
            blocks[-1][2] += w2

    return [(x, sum_y / w) for x, sum_y, w in blocks]


def predict(breakpoints: list[list[float]], score: float) -> float:
    """按分段常数表把一个分数映射成 P(正确)。低于第一段起点时取第一段的 p。"""
    if not breakpoints:
        raise ValueError("breakpoints 为空，先调用 fit()/pav() 拟合")
    p = breakpoints[0][1]
    for x, val in breakpoints:
        if score >= x:
            p = val
        else:
            break
    return p


def fit(scores: list[float], correct: list[bool], bins: int = 10) -> dict:
    """拟合校准表，并报告校准前后的 ECE（期望校准误差）对比。

    返回 {"breakpoints": [[x, p], ...], "n": n, "n_pos": k,
          "ece_before": float | None, "ece_after": float | None, "bins": bins}。
    ece_before 用原始 score（clip 到 [0,1]）当置信度；ece_after 用映射后的 P(正确)。
    空输入返回 n=0、breakpoints=[]、ece_before/ece_after=None。
    """
    if len(scores) != len(correct):
        raise ValueError(f"scores 与 correct 长度不一致：{len(scores)} vs {len(correct)}")
    n = len(scores)
    if n == 0:
        return {"breakpoints": [], "n": 0, "n_pos": 0, "ece_before": None, "ece_after": None, "bins": bins}

    ys = [1.0 if c else 0.0 for c in correct]
    blocks = pav(list(scores), ys)
    breakpoints = [[x, p] for x, p in blocks]

    clipped = [min(1.0, max(0.0, s)) for s in scores]
    mapped = [predict(breakpoints, s) for s in scores]
    ece_before = ece(clipped, list(correct), bins=bins)["ece"]
    ece_after = ece(mapped, list(correct), bins=bins)["ece"]

    return {
        "breakpoints": breakpoints,
        "n": n,
        "n_pos": sum(1 for c in correct if c),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "bins": bins,
    }


def decision_threshold(c_fa: float, c_fr: float) -> float:
    """贝叶斯决策：自动通过当且仅当 P(正确) >= p*。

    c_fa = 错送代价（放行了错的），c_fr = 确认代价（把对的也转人工）。
    p* = 1 - c_fr / c_fa —— c_fa 越大（错送代价越高）门槛越高，越谨慎。
    """
    if c_fa <= 0:
        raise ValueError("c_fa 必须为正")
    return 1 - c_fr / c_fa


def gate_at(breakpoints: list[list[float]], p_star: float, scores: list[float], correct: list[bool]) -> dict:
    """用 P(正确) >= p_star 做闸门，统计通过率与放行风险。

    TP = 放行且对，FA = 放行且错（误伤代价 c_fa 的来源），
    FR = 拦下但其实是对的（多余的人工确认），TR = 拦下且确实是错的。
    coverage = 放行占比；risk = 放行样本里的错误占比。
    """
    if len(scores) != len(correct):
        raise ValueError(f"scores 与 correct 长度不一致：{len(scores)} vs {len(correct)}")
    n = len(scores)
    tp = fa = fr = tr = 0
    for s, c in zip(scores, correct):
        passed = predict(breakpoints, s) >= p_star
        if passed:
            tp += 1 if c else 0
            fa += 0 if c else 1
        else:
            fr += 1 if c else 0
            tr += 0 if c else 1
    passed_n = tp + fa
    coverage = passed_n / n if n else 0.0
    risk = fa / passed_n if passed_n else 0.0
    return {"coverage": round(coverage, 4), "risk": round(risk, 4), "TP": tp, "FA": fa, "FR": fr, "TR": tr}


def class_counts(decisions: list[str]) -> dict:
    """统计闸门产出的决策分布，供 §2.6 门槛检查用（calib ≥ 300，每类 ≥ 50 才可信）。

    decisions 元素为 "confident" / "non_confident"，其余值一律计入 non_confident。
    """
    n_confident = sum(1 for d in decisions if d == "confident")
    return {"confident": n_confident, "non_confident": len(decisions) - n_confident}


if __name__ == "__main__":
    import random

    rng = random.Random(0)
    n = 300
    # 构造一批"系统性过度自信"的合成分数：真实正确率明显低于分数本身。
    true_p = [rng.uniform(0.3, 1.0) for _ in range(n)]
    scores = [min(1.0, p + 0.15) for p in true_p]
    correct = [rng.random() < p for p in true_p]

    result = fit(scores, correct)
    print(f"n={result['n']}  n_pos={result['n_pos']}  段数={len(result['breakpoints'])}")
    print(f"ece_before={result['ece_before']:.4f}  ece_after={result['ece_after']:.4f}")

    p_star = decision_threshold(c_fa=10, c_fr=1)
    gate = gate_at(result["breakpoints"], p_star, scores, correct)
    print(f"p*={p_star:.3f}  gate={gate}")

    decisions = ["confident" if predict(result["breakpoints"], s) >= p_star else "non_confident" for s in scores]
    print(f"class_counts={class_counts(decisions)}")
