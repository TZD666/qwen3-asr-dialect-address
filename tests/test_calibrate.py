"""保序回归校准 + 贝叶斯闸门的回归用例。纯 assert，`python tests/test_calibrate.py` 直接跑，装了 pytest 也能收。

覆盖：PAV 手算样例与打结样例、单调性、predict 边界、decision_threshold 的贝叶斯公式、
gate_at 计数、class_counts 统计（见 eval/calibrate.py 顶部说明的 §2.6 用途）。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from calibrate import class_counts, decision_threshold, fit, gate_at, pav, predict  # noqa: E402


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_pav_toy_sequence():
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ys = [0, 1, 0, 1, 1, 1]
    blocks = pav(xs, ys)
    expect = [(0.1, 0.0), (0.2, 0.5), (0.4, 1.0)]
    assert len(blocks) == len(expect), blocks
    for (x, p), (ex, ep) in zip(blocks, expect):
        assert _approx(x, ex) and _approx(p, ep), blocks


def test_pav_ties_pooled_first():
    xs = [0.5, 0.5, 0.7]
    ys = [1, 0, 1]
    blocks = pav(xs, ys)
    expect = [(0.5, 0.5), (0.7, 1.0)]
    assert len(blocks) == len(expect), blocks
    for (x, p), (ex, ep) in zip(blocks, expect):
        assert _approx(x, ex) and _approx(p, ep), blocks


def test_pav_empty_input():
    assert pav([], []) == []


def test_fit_empty_input():
    result = fit([], [])
    assert result["n"] == 0
    assert result["breakpoints"] == []
    assert result["ece_before"] is None
    assert result["ece_after"] is None


def test_pav_monotone_on_random_points():
    rng = random.Random(42)
    xs = [rng.random() for _ in range(200)]
    ys = [1.0 if rng.random() < x else 0.0 for x in xs]  # 分数越高，越可能对：非单调噪声下仍应校准成单调
    blocks = pav(xs, ys)
    ps = [p for _, p in blocks]
    xs_sorted = [x for x, _ in blocks]
    assert xs_sorted == sorted(xs_sorted), "起点必须升序"
    assert all(ps[i] <= ps[i + 1] + 1e-12 for i in range(len(ps) - 1)), "p 必须单调不减"


def test_predict_reproduces_block_means_on_training_points():
    xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ys = [0, 1, 0, 1, 1, 1]
    result = fit(xs, [bool(y) for y in ys])
    bp = result["breakpoints"]
    expect_p = {0.1: 0.0, 0.2: 0.5, 0.3: 0.5, 0.4: 1.0, 0.5: 1.0, 0.6: 1.0}
    for x, ep in expect_p.items():
        assert _approx(predict(bp, x), ep), (x, predict(bp, x), ep)


def test_predict_below_first_breakpoint_returns_first_p():
    bp = [[0.3, 0.2], [0.5, 0.8]]
    assert predict(bp, 0.0) == 0.2
    assert predict(bp, 0.1) == 0.2


def test_ece_after_le_ece_before_when_overconfident():
    # 分数系统性偏高（比真实正确率高 0.2），校准后 ECE 不应变差。
    rng = random.Random(7)
    true_p = [rng.uniform(0.1, 0.8) for _ in range(200)]
    scores = [min(1.0, p + 0.2) for p in true_p]
    correct = [rng.random() < p for p in true_p]
    result = fit(scores, correct)
    assert result["ece_after"] <= result["ece_before"] + 1e-9, result


def test_decision_threshold_bayes_formula():
    assert _approx(decision_threshold(10, 1), 0.9)
    assert _approx(decision_threshold(2, 1), 0.5)


def test_gate_at_counts_on_tiny_example():
    # 校准表：score < 0.5 → p=0.2；score >= 0.5 → p=0.9
    bp = [[0.0, 0.2], [0.5, 0.9]]
    scores = [0.1, 0.2, 0.6, 0.7, 0.8]
    correct = [True, False, True, False, True]
    # p_star=0.5：放行 0.6/0.7/0.8（三条），其中 0.7 错 → FA=1，TP=2
    # 拦下 0.1/0.2，其中 0.1 对 → FR=1，0.2 错 → TR=1
    result = gate_at(bp, 0.5, scores, correct)
    assert result == {"coverage": 0.6, "risk": round(1 / 3, 4), "TP": 2, "FA": 1, "FR": 1, "TR": 1}, result


def test_class_counts():
    decisions = ["confident", "confident", "non_confident", "confident", "non_confident"]
    assert class_counts(decisions) == {"confident": 3, "non_confident": 2}
    assert class_counts([]) == {"confident": 0, "non_confident": 0}


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if fails else 0)
