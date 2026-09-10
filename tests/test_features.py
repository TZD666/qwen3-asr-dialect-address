"""特征重放必须和现场决策完全一致：同一组参数下，重放出来的决策和 Top-1 与 rank() 现场给的一样。

不一致就说明 rescore.decide_all 和 rank.decide 的闸门顺序或条件漂了——调参搜索建立在这个等式上。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr import rank as rank_mod  # noqa: E402
from features import build  # noqa: E402
from rescore import DEC_CODES, DEC_NAMES, FeatureSet, decide_all, metrics, quadrants  # noqa: E402
from stages import replay_decision  # noqa: E402

_CACHE: dict = {}


def rows_for(sources: tuple[str, ...], limit: int = 0) -> list[dict]:
    key = (sources, limit)
    if key not in _CACHE:
        _CACHE[key] = build(list(sources), limit=limit, verbose=False)[0]
    return _CACHE[key]


def test_replay_reproduces_live_decisions_on_text_and_oov():
    rows = rows_for(("text", "oov"))
    fs = FeatureSet.from_rows(rows)
    d = decide_all(fs, rank_mod.current_params())
    for i, r in enumerate(rows):
        assert DEC_NAMES[int(d["decision"][i])] == r["decision_live"], r["id"]
        assert int(d["top"][i]) == (r["top_live"] if r["top_live"] is not None else -1), r["id"]


def test_replay_reproduces_live_decisions_on_synthetic_head():
    rows = rows_for(("synthetic",), limit=60)
    fs = FeatureSet.from_rows(rows)
    d = decide_all(fs, rank_mod.current_params())
    mism = [r["id"] for i, r in enumerate(rows) if DEC_NAMES[int(d["decision"][i])] != r["decision_live"]]
    assert not mism, mism


def test_replay_matches_stages_replay_decision():
    """rescore 与 stages.replay_decision 是同一套闸门的两种写法，必须给出同样的结果。"""
    rows = rows_for(("text", "oov"))
    fs = FeatureSet.from_rows(rows)
    p = rank_mod.current_params()
    d = decide_all(fs, p)
    for i, r in enumerate(rows):
        if not r["chains"]:
            continue
        top = r["chains"][0]
        second = r["chains"][1] if len(r["chains"]) > 1 else None
        feats = {
            "empty": False, "top_total": top["total_live"], "top_sim": top["f"][0],
            "second_total": second["total_live"] if second else None,
            "second_admin_differs": bool(second and second["admin"] != top["admin"]),
            "deep_hit": top["deep"], "margin": (top["total_live"] - second["total_live"]) if second else 1.0,
            "coverage": top["f"][1], "single_hit_dist": top["shd"],
        }
        assert replay_decision(feats) == DEC_NAMES[int(d["decision"][i])], r["id"]


def test_gate_precedence_sim_then_single_then_partial_then_margin():
    rows = rows_for(("text",))
    fs = FeatureSet.from_rows(rows)
    p = rank_mod.current_params()
    d = decide_all(fs, {**p, "SIM_MIN": 1.01})
    assert all(d["decision"] == DEC_CODES["reject"])
    d = decide_all(fs, {**p, "MARGIN_MIN": 1.0})
    assert (d["decision"] == DEC_CODES["ambiguous"]).sum() >= 10
    # 分差闸门只拦异区：同区分歧不拦
    d0 = decide_all(fs, p)
    assert all(d0["decision"] == DEC_CODES["confident"])


def test_quadrants_and_metrics_are_consistent():
    rows = rows_for(("text", "oov"))
    fs = FeatureSet.from_rows(rows)
    p = rank_mod.current_params()
    q = quadrants(fs, p)
    m = metrics(fs, p)
    assert m["n"] == fs.n == len(q)
    assert m["TP"] + m["FA"] + m["FR"] + m["TR"] == fs.n
    assert m["oov_fa"] == int(((q == "FA") & fs.where(negative_type="oov_db")).sum())
    # 无地址样本：永远算错，放行即 FA
    fs2 = FeatureSet.from_rows(rows)
    fs2.has_addr[:] = False
    q2 = quadrants(fs2, p)
    assert set(q2) <= {"FA", "TR"}


def test_replay_is_fast():
    rows = rows_for(("synthetic",), limit=60) + rows_for(("text", "oov"))
    fs = FeatureSet.from_rows(rows)
    p = rank_mod.current_params()
    decide_all(fs, p)
    t = time.time()
    for _ in range(50):
        decide_all(fs, p)
    assert (time.time() - t) / 50 < 0.05


def test_where_filters_by_split_and_meta():
    rows = rows_for(("text", "oov"))
    fs = FeatureSet.from_rows(rows)
    assert fs.where(split="eval").sum() == fs.n
    assert fs.where(source="oov").sum() == 10
    assert fs.where(negative_type="oov_db").sum() == 10
    assert np.array_equal(fs.subset(np.where(fs.where(source="text"))[0]).source, np.array(["text"] * 24))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
