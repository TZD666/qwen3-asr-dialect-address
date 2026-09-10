"""小样本统计：Wilson 区间、rule of three、bootstrap、McNemar。全部只用标准库。

原则（评测体系设计 §2.6 / §9）：
  * n < 20 的分片只报 k/n，不报百分比
  * 报百分比时旁边一律带 95% Wilson 区间
  * 配对比较用 bootstrap（按样本重采）给净收益区间，用 McNemar 给 p 值
"""

from __future__ import annotations

import math
import random

Z95 = 1.959964
MIN_N_FOR_PCT = 20


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_rate(k: int, n: int) -> str:
    """n ≥ 20：`87.5% [70.1, 95.9]`；n < 20：`7/8`；n = 0：`-`。"""
    if n == 0:
        return "-"
    if n < MIN_N_FOR_PCT:
        return f"{k}/{n}"
    lo, hi = wilson(k, n)
    return f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def rule_of_three(n: int) -> float | None:
    """观测到 0 错时错误率的 95% 上限 ≈ 3/n。"""
    return 3.0 / n if n else None


def bootstrap_diff(a: list[bool], b: list[bool], iters: int = 1000, seed: int = 0) -> dict:
    """配对布尔序列 b − a 的均值差及 95% bootstrap 区间（按样本重采）。"""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    diffs = [int(y) - int(x) for x, y in zip(a, b)]
    point = sum(diffs) / n
    samples = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    lo = samples[int(0.025 * iters)]
    hi = samples[min(iters - 1, int(0.975 * iters))]
    return {"diff": round(point, 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": n, "iters": iters}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """配对对错的 McNemar 精确检验（二项分布，双侧）。b01 = a 错 b 对，b10 = a 对 b 错。"""
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    m = b01 + b10
    if m == 0:
        return {"b01": b01, "b10": b10, "p": 1.0}
    k = min(b01, b10)
    # 双侧精确 p = 2 * P(X ≤ k), X ~ Bin(m, 0.5)
    p = 2 * sum(math.comb(m, i) for i in range(k + 1)) / (2 ** m)
    return {"b01": b01, "b10": b10, "p": round(min(1.0, p), 4)}


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> dict:
    """期望校准误差 + 各桶数据（reliability 图用）。"""
    n = len(confidences)
    buckets = []
    total = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        idx = [j for j, c in enumerate(confidences) if (lo <= c < hi) or (i == bins - 1 and c == 1.0)]
        if not idx:
            buckets.append({"lo": lo, "hi": hi, "n": 0})
            continue
        acc = sum(1 for j in idx if correct[j]) / len(idx)
        conf = sum(confidences[j] for j in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
        buckets.append({"lo": lo, "hi": hi, "n": len(idx), "acc": round(acc, 3), "conf": round(conf, 3)})
    return {"ece": round(total, 4), "buckets": buckets, "n": n,
            "nonempty_buckets": sum(1 for b in buckets if b["n"])}
