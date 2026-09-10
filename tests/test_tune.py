"""调参搜索：玩具特征集上能找回植入的最优；约束器拒绝降 EM 的提案；可辨识表识别平的参数。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from rescore import FeatureSet  # noqa: E402
from tune import (  # noqa: E402
    REPLAYABLE, Objective, _set_simplex, check_constraints, coordinate_descent, identifiability, search, slice_deltas,
)

HAND = {"W_SIM": 0.5, "W_COV": 0.2, "W_PRIOR": 0.15, "W_DEPTH": 0.15, "W_CONFLICT": 0.25,
        "MARGIN_MIN": 0.06, "SIM_MIN": 0.62, "SINGLE_HIT_MIN_COV": 0.25}
BOUNDS = json.loads((ROOT / "data/params/rank_params.json").read_text(encoding="utf-8"))["bounds"]
CFG = {"restarts": 4, "seed": 1, "max_sweeps": 6, "weight_grid": 11, "threshold_step": 0.02, "l1_tiebreak": 0.001}
COSTS = {"c_fa": 10, "c_fr": 1, "c_tr": 1, "c_miss": 0}


def toy_rows(n: int = 60, seed: int = 0) -> list[dict]:
    """植入的规律：真值链 prior 高、sim 略低；错链 sim 高、prior 低。
    手设权重（sim 0.5）会选错链；把 prior 权重抬高才对。"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        # 手设权重：bad 0.77 > good 0.69 → 自动通过且错（FA）；W_PRIOR 抬到 0.6：good 0.83 > bad 0.41
        good = {"name": "gold", "f": [0.70, 0.5, 0.95, 0.67, 0.0], "ok": True, "admin": "A", "deep": True, "n_ev": 2, "shd": None}
        bad = {"name": "bad", "f": [0.99, 0.8, 0.10, 0.67, 0.0], "ok": False, "admin": "B", "deep": True, "n_ev": 2, "shd": None}
        chains = [bad, good] if rng.random() < 0.9 else [good, bad]
        rows.append({"id": f"toy:{i}", "source": "toy", "split": "train" if i % 2 == 0 else "eval", "group": str(i),
                     "meta": {"has_address": True, "split": "train" if i % 2 == 0 else "eval", "source": "toy",
                              "negative_type": None, "dialect_group": "官话", "label_status": "confirmed"},
                     "decision_live": "confident", "top_live": 0, "chains": chains})
    return rows


def test_set_simplex_keeps_sum_one():
    q = _set_simplex(HAND, "W_PRIOR", 0.6)
    assert sum(q[k] for k in ("W_SIM", "W_COV", "W_PRIOR", "W_DEPTH")) == pytest.approx(1.0)
    assert q["W_PRIOR"] == pytest.approx(0.6) and q["W_CONFLICT"] == HAND["W_CONFLICT"]


def test_search_finds_planted_optimum():
    fs = FeatureSet.from_rows(toy_rows())
    obj = Objective(fs, fs.where(split="train"), COSTS, HAND, CFG["l1_tiebreak"])
    j_hand = obj.cost(HAND)
    assert j_hand > 5.0                       # 手设权重在玩具集上几乎全错
    found = search(obj, HAND, BOUNDS, CFG)
    best = {**HAND, **found["best"]}
    assert obj.cost(best) == pytest.approx(0.0)
    # L1 拉回：只走到代价归零的最近一点，不会跑到 W_PRIOR 压倒一切——prior 抬、sim 降就够了
    assert best["W_PRIOR"] > HAND["W_PRIOR"] and best["W_SIM"] < HAND["W_SIM"]


def test_coordinate_descent_prefers_minimal_change_on_plateau():
    """目标平的时候（全对），提案不该乱动：J 含 L1 拉回手设值。"""
    rows = toy_rows()
    for r in rows:
        for c in r["chains"]:
            c["f"] = [0.99, 0.9, 0.95, 1.0, 0.0] if c["ok"] else [0.5, 0.3, 0.1, 0.33, 0.0]   # 真值链各项都最高：任何权重都选它
    fs = FeatureSet.from_rows(rows)
    obj = Objective(fs, fs.where(split="train"), COSTS, HAND, 0.001)
    p, j, _ = coordinate_descent(obj, HAND, BOUNDS, CFG)
    assert all(p[k] == pytest.approx(HAND[k]) for k in REPLAYABLE)


def test_identifiability_reports_flat_parameters():
    rows = toy_rows()
    for r in rows:
        for c in r["chains"]:
            c["f"][4] = 0.0                                     # 没有冲突 → W_CONFLICT 完全平
    fs = FeatureSet.from_rows(rows)
    tr, ev = fs.where(split="train"), fs.where(split="eval")
    obj_t, obj_e = Objective(fs, tr, COSTS, HAND, 0.001), Objective(fs, ev, COSTS, HAND, 0.0)
    ident = identifiability(obj_t, obj_e, HAND, BOUNDS, CFG)
    assert ident["W_CONFLICT"]["train"]["flat_share"] == pytest.approx(1.0)
    assert ident["W_CONFLICT"]["pinned"] is False


def test_constraints_reject_a_proposal_that_lowers_em():
    fs = FeatureSet.from_rows(toy_rows())
    ev = fs.where(split="eval")
    good = {**HAND, "W_SIM": 0.2, "W_COV": 0.1, "W_PRIOR": 0.6, "W_DEPTH": 0.1}
    gate = {"met": False, "detail": "toy", "calib_ok": False}
    deltas = slice_deltas(fs, ev, good, HAND)
    cons = check_constraints(fs, ev, good, HAND, COSTS, gate, False, deltas)   # 从好参数“提案”回手设：EM 下降
    names = {c["name"]: c["ok"] for c in cons}
    assert names["eval_em_not_lower"] is False and names["sample_gate"] is False
    cons2 = check_constraints(fs, ev, HAND, good, COSTS, gate, True, slice_deltas(fs, ev, HAND, good))
    names2 = {c["name"]: c["ok"] for c in cons2}
    assert names2["eval_em_not_lower"] and names2["eval_risk_not_higher"] and names2["sample_gate"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
