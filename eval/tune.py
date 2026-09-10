#!/usr/bin/env python3
"""调参闭环：在特征重放上搜权重与阈值，留出片验证，写提案；人执行 --apply 才生效。

    python eval/features.py                    # 数据变了先跑一次（约 5 分钟）
    python eval/tune.py                        # 出提案：eval/tuning/<时间>_proposal.json + .md
    python eval/tune.py --confirm              # 用候选参数真跑 run_eval + 回归，报重放偏差
    python eval/tune.py --apply <proposal>     # verdict=APPLICABLE 才写 data/params/rank_params.json v+1
    python eval/tune.py --apply <proposal> --allow-below-threshold   # 样本未达门槛也写，标 tuned-below-threshold

目标：J(θ) = 训练片平均代价 + 0.001·‖θ − θ_hand‖₁，代价 = c_fa·FA + c_fr·FR + (c_tr + c_miss)·TR。
平台上（目标不变的区域）偏向离手设值最近的点——数据钉不住的参数就不要动。

提案里最要紧的一张表是**可辨识区间**：每个参数在最优点附近 J 不变的范围。今天的数据上
大部分参数是平的，这张表如实说"数据钉不住它"，比一组看起来学出来的新值诚实得多。

诚实的边界（也写进提案头）：
  * 重放里"正确" = Top-1 链是真值链，拼装错不在环内；--confirm 用真跑量化偏差
  * 音频只用第 1 遍转写；STRONG_HIT / LONGER_NAME_TOL / UNIQUE_SHORT_MAX / MAX_DIST 影响检索，不可重放，保持手设
  * 合成集由同一代价矩阵生成，只能约束下游；仅靠它的提案不得 apply（门槛机制强制）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr import rank as rank_mod  # noqa: E402

from rescore import DEC_CODES, FeatureSet, decide_all, metrics, quadrants  # noqa: E402
from splits import effective_counts, gate_status, load_splits  # noqa: E402

FEATURES = ROOT / "eval" / "cache" / "features.jsonl"
CONFIG = ROOT / "data" / "params" / "tune_config.json"
PARAMS_FILE = ROOT / "data" / "params" / "rank_params.json"
TUNING_DIR = ROOT / "eval" / "tuning"
SIMPLEX = ("W_SIM", "W_COV", "W_PRIOR", "W_DEPTH")
OTHERS = ("W_CONFLICT", "MARGIN_MIN", "SIM_MIN", "SINGLE_HIT_MIN_COV")
REPLAYABLE = SIMPLEX + OTHERS


# --------------------------------------------------------------------------
# 目标
# --------------------------------------------------------------------------


class Objective:
    def __init__(self, fs: FeatureSet, sel: np.ndarray, costs: dict, hand: dict, l1: float):
        self.fs, self.sel, self.costs, self.hand, self.l1 = fs, sel, costs, hand, l1
        self.n = int(sel.sum())

    def cost(self, p: dict, sel: np.ndarray | None = None) -> float:
        sel = self.sel if sel is None else sel
        q = quadrants(self.fs, p)[sel]
        n = len(q)
        if n == 0:
            return 0.0
        fa, fr, tr = (int((q == k).sum()) for k in ("FA", "FR", "TR"))
        c = self.costs
        return (c["c_fa"] * fa + c["c_fr"] * fr + (c["c_tr"] + c.get("c_miss", 0)) * tr) / n

    def J(self, p: dict) -> float:
        pen = sum(abs(p[k] - self.hand[k]) for k in REPLAYABLE)
        return self.cost(p) + self.l1 * pen


def _grid(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


def _set_simplex(p: dict, k: str, v: float) -> dict:
    """把权重 k 设成 v，其余三个等比缩放保持和为 1。"""
    q = dict(p)
    others = [o for o in SIMPLEX if o != k]
    rest = sum(p[o] for o in others)
    q[k] = v
    if rest <= 1e-12:
        for o in others:
            q[o] = (1 - v) / len(others)
    else:
        for o in others:
            q[o] = p[o] * (1 - v) / rest
    return q


def coordinate_descent(obj: Objective, start: dict, bounds: dict, cfg: dict) -> tuple[dict, float, int]:
    p = dict(start)
    best = obj.J(p)
    wgrid = [i / (cfg["weight_grid"] - 1) for i in range(cfg["weight_grid"])]
    sweeps = 0
    for _ in range(cfg["max_sweeps"]):
        sweeps += 1
        changed = False
        for k in SIMPLEX:
            cands = [(_set_simplex(p, k, v), v) for v in wgrid if bounds[k][0] - 1e-9 <= v <= bounds[k][1] + 1e-9]
            scored = [(obj.J(q), abs(v - p[k]), q) for q, v in cands]
            j, _, q = min(scored, key=lambda t: (round(t[0], 12), t[1]))
            if j < best - 1e-12:
                p, best, changed = q, j, True
        for k in OTHERS:
            cands = [{**p, k: v} for v in _grid(bounds[k][0], bounds[k][1], cfg["threshold_step"])]
            scored = [(obj.J(q), abs(q[k] - p[k]), q) for q in cands]
            j, _, q = min(scored, key=lambda t: (round(t[0], 12), t[1]))
            if j < best - 1e-12:
                p, best, changed = q, j, True
        if not changed:
            break
    return p, best, sweeps


def search(obj: Objective, hand: dict, bounds: dict, cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["seed"])
    starts = [dict(hand)]
    for _ in range(cfg["restarts"] - 1):
        w = rng.dirichlet(np.ones(len(SIMPLEX)))
        s = dict(hand)
        for k, v in zip(SIMPLEX, w):
            s[k] = float(v)
        for k in OTHERS:
            lo, hi = bounds[k]
            s[k] = float(rng.uniform(lo, hi))
        starts.append(s)
    results = []
    t0 = time.time()
    for i, s in enumerate(starts):
        p, j, sweeps = coordinate_descent(obj, s, bounds, cfg)
        results.append((j, i, p, sweeps))
    results.sort(key=lambda t: (round(t[0], 12), t[1]))
    best_j, _, best_p, _ = results[0]
    # 去重后的前 5 个不同最优点
    top = []
    for j, i, p, sw in results:
        key = tuple(round(p[k], 4) for k in REPLAYABLE)
        if key not in {tuple(round(t["params"][k], 4) for k in REPLAYABLE) for t in top}:
            top.append({"J": round(j, 6), "start": i, "sweeps": sw, "params": {k: round(p[k], 4) for k in REPLAYABLE}})
        if len(top) >= 5:
            break
    return {"best": {k: round(best_p[k], 4) for k in REPLAYABLE}, "best_J": round(best_j, 6),
            "top_optima": top, "n_starts": len(starts), "elapsed": round(time.time() - t0, 1)}


def identifiability(obj_train: Objective, obj_eval: Objective, p: dict, bounds: dict, cfg: dict) -> dict:
    """每个参数：在最优点附近，走网格时目标不变的最大连续区间（train 与 eval 各一份）。"""
    out = {}
    for k in REPLAYABLE:
        if k in SIMPLEX:
            grid = [i / (cfg["weight_grid"] - 1) for i in range(cfg["weight_grid"])]
            mk = lambda v, k=k: _set_simplex(p, k, v)
        else:
            grid = _grid(bounds[k][0], bounds[k][1], cfg["threshold_step"])
            mk = lambda v, k=k: {**p, k: v}
        res = {}
        for name, obj in (("train", obj_train), ("eval", obj_eval)):
            base = obj.cost(p)
            # 找离当前值最近的网格点，向两边扩到目标变化为止
            cur = p[k]
            idx = min(range(len(grid)), key=lambda i: abs(grid[i] - cur))
            lo = hi = grid[idx]
            i = idx - 1
            while i >= 0 and abs(obj.cost(mk(grid[i])) - base) < 1e-12:
                lo = grid[i]
                i -= 1
            i = idx + 1
            while i < len(grid) and abs(obj.cost(mk(grid[i])) - base) < 1e-12:
                hi = grid[i]
                i += 1
            span = round(hi - lo, 4)
            total = round(grid[-1] - grid[0], 4)
            res[name] = {"flat": [round(lo, 4), round(hi, 4)], "flat_share": round(span / total, 3) if total else 1.0}
        res["value"] = round(p[k], 4)
        res["pinned"] = res["train"]["flat_share"] < 0.2
        out[k] = res
    return out


def frontier(fs: FeatureSet, sel: np.ndarray, p: dict) -> list[dict]:
    from run_eval import GRID_MARGIN, GRID_SIM

    pts = []
    for mm in GRID_MARGIN:
        for sm in GRID_SIM:
            m = metrics(fs, {**p, "MARGIN_MIN": mm, "SIM_MIN": sm}, sel)
            pts.append({"margin_min": mm, "sim_min": sm, "coverage": m["coverage"], "risk": m["risk"], "FA": m["FA"], "TP": m["TP"]})
    return pts


def coverage_at_risk(pts: list[dict], targets: list[float]) -> dict:
    out = {}
    for t in targets:
        ok = [q for q in pts if q["risk"] is not None and q["risk"] <= t and q["coverage"] is not None]
        if ok:
            b = max(ok, key=lambda q: (q["coverage"], -q["risk"]))
            out[str(t)] = {"coverage": b["coverage"], "risk": b["risk"], "margin_min": b["margin_min"], "sim_min": b["sim_min"]}
        else:
            out[str(t)] = None
    return out


def slice_deltas(fs: FeatureSet, sel_eval: np.ndarray, hand: dict, prop: dict) -> list[dict]:
    """留出片逐分片：TP 数在提案下不得比手设少（n ≥ 20 的分片）。"""
    qh, qp = quadrants(fs, hand), quadrants(fs, prop)
    rows = []
    axes = ("source", "dialect_group", "address_depth", "noise", "negative_type", "rule")
    for ax in axes:
        vals = sorted({(m.get(ax) if ax != "source" else fs.source[i]) for i, m in enumerate(fs.meta) if sel_eval[i]}, key=str)
        for v in vals:
            if v is None:
                continue
            sel = sel_eval & (fs.source == v if ax == "source" else np.array([m.get(ax) == v for m in fs.meta]))
            n = int(sel.sum())
            if n == 0:
                continue
            h = {k: int((qh[sel] == k).sum()) for k in ("TP", "FA", "FR", "TR")}
            pp = {k: int((qp[sel] == k).sum()) for k in ("TP", "FA", "FR", "TR")}
            rows.append({"slice": f"{ax}={v}", "n": n, "hand": h, "proposed": pp, "delta_tp": pp["TP"] - h["TP"],
                         "delta_fa": pp["FA"] - h["FA"]})
    return rows


def check_constraints(fs: FeatureSet, sel_eval: np.ndarray, hand: dict, prop: dict, costs: dict,
                      gate: dict, allow_below: bool, deltas: list[dict]) -> list[dict]:
    mh, mp = metrics(fs, hand, sel_eval, costs), metrics(fs, prop, sel_eval, costs)
    cs = [
        {"name": "eval_em_not_lower", "ok": (mp["em_proxy"] or 0) >= (mh["em_proxy"] or 0), "hand": mh["em_proxy"], "proposed": mp["em_proxy"]},
        {"name": "eval_risk_not_higher", "ok": (mp["risk"] or 0) <= (mh["risk"] or 0), "hand": mh["risk"], "proposed": mp["risk"]},
        {"name": "oov_fa_not_higher", "ok": mp["oov_fa"] <= mh["oov_fa"], "hand": mh["oov_fa"], "proposed": mp["oov_fa"]},
        {"name": "no_address_fa_not_higher", "ok": mp["neg_fa"] <= mh["neg_fa"], "hand": mh["neg_fa"], "proposed": mp["neg_fa"]},
        {"name": "no_slice_n20_loses_tp", "ok": all(d["delta_tp"] >= 0 for d in deltas if d["n"] >= 20),
         "detail": [d["slice"] for d in deltas if d["n"] >= 20 and d["delta_tp"] < 0]},
        {"name": "sample_gate", "ok": bool(gate["met"]) or allow_below, "detail": gate["detail"]},
    ]
    return cs


# --------------------------------------------------------------------------
# --confirm：真跑
# --------------------------------------------------------------------------


def confirm(prop: dict, fs: FeatureSet) -> dict:
    """用候选参数真跑 run_eval（text / oov / synthetic / audio 走缓存）和回归，量化重放偏差。"""
    import run_eval
    import regression

    saved = rank_mod.current_params()
    rank_mod.apply_params({k: prop[k] for k in REPLAYABLE})
    out: dict = {"ran": True, "replay_divergence": {}, "run_eval": {}, "regression": None}
    try:
        cases = {
            "text": regression.make_args(),
            "oov": regression.make_args(negatives="oov"),
            "synthetic": regression.make_args(eval=str(regression.SYNTH), tag="synthetic"),
            "audio": regression.make_args(mode="audio", asr_cache_readonly=True),
        }
        qp = quadrants(fs, prop)
        idx = {i: k for k, i in enumerate(fs.ids)}
        fails: list[str] = []
        for name, args in cases.items():
            try:
                r = run_eval.run(args)
            except Exception as e:  # 音频缓存缺失等
                out["run_eval"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
                continue
            s = r["summary"]["slices"].get("全部=全部", {})
            out["run_eval"][name] = {"n": s.get("n"), "exact": s.get("exact"), "quadrant": s.get("E.quadrant"),
                                     "attribution": s.get("attribution")}
            # 逐条比：真跑四格 vs 重放四格
            src = {"text": "text", "oov": "oov", "synthetic": "synthetic", "audio": "audio"}[name]
            div = 0
            n = 0
            for row in r["rows"]:
                if not row.get("scored") or row.get("skipped"):
                    continue
                key = f"{src}:{row['id']}"
                j = idx.get(key)
                if j is None:
                    continue
                n += 1
                real = row.get("stage_e", {}).get("quadrant")
                if real != qp[j]:
                    div += 1
            out["replay_divergence"][name] = {"n": n, "diverged": div}
            gpath = regression.GOLDEN / f"{'text' if name == 'text' else 'text_' + name}.json"
            if gpath.exists() and name != "audio":
                snap = regression.snapshot(r)
                fails.extend(regression.compare(name, snap, json.loads(gpath.read_text(encoding="utf-8"))))
        out["regression"] = "pass" if not fails else "fail"
        out["regression_fails"] = fails
    finally:
        rank_mod.apply_params(saved)
    return out


# --------------------------------------------------------------------------
# 提案 / 应用
# --------------------------------------------------------------------------


def _git_head() -> str:
    try:
        head = (ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = ROOT / ".git" / head[5:]
            return ref.read_text(encoding="utf-8").strip()[:7] if ref.exists() else "unknown"
        return head[:7]
    except Exception:
        return "unknown"


def build_records(fs: FeatureSet, strict: bool) -> list[dict]:
    recs = []
    for i, m in enumerate(fs.meta):
        if strict and m.get("label_status") == "draft":
            continue
        recs.append({"split": str(fs.split[i]), "source": str(fs.source[i]), "group": m.get("group"),
                     "label_status": m.get("label_status"), "difficulty": m.get("difficulty") or [], "id": fs.ids[i]})
    return recs


def propose(a: Namespace) -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    pfile = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    hand = {k: float(v) for k, v in pfile["params"].items()}
    bounds = pfile["bounds"]
    fs = FeatureSet.load(a.features)
    fmeta_path = Path(a.features).with_name(Path(a.features).stem + "_meta.json")
    fmeta = json.loads(fmeta_path.read_text(encoding="utf-8")) if fmeta_path.exists() else {}
    strict_mask = np.array([not (a.strict and m.get("label_status") == "draft") for m in fs.meta])
    sel_train = fs.where(split="train") & strict_mask
    sel_calib = fs.where(split="calib") & strict_mask
    sel_eval = fs.where(split="eval") & strict_mask
    costs = cfg["costs"]
    counts = effective_counts(build_records(fs, a.strict))
    gate = gate_status(counts, {**load_splits().get("thresholds", {}), **cfg.get("thresholds", {})})

    obj_train = Objective(fs, sel_train, costs, hand, cfg["search"]["l1_tiebreak"])
    obj_eval = Objective(fs, sel_eval, costs, hand, 0.0)
    print(f"特征 {fs.n} 条：train {int(sel_train.sum())} / calib {int(sel_calib.sum())} / eval {int(sel_eval.sum())}"
          f"（有效 train_eff={gate['train_eff']} calib_eff={gate['calib_eff']}）")
    t0 = time.time()
    if sel_train.sum() == 0:
        print("训练片为空：只报手设参数在各片上的指标，不搜索")
        found = {"best": {k: hand[k] for k in REPLAYABLE}, "best_J": None, "top_optima": [], "n_starts": 0, "elapsed": 0}
    else:
        found = search(obj_train, hand, bounds, cfg["search"])
    prop = {**hand, **found["best"]}
    print(f"搜索完成 {found['elapsed']}s  J_hand={obj_train.J(hand):.5f}  J*={found['best_J']}")
    ident = identifiability(obj_train, obj_eval, prop, bounds, cfg["search"])
    front = frontier(fs, sel_eval, prop)
    front_hand = frontier(fs, sel_eval, hand)
    deltas = slice_deltas(fs, sel_eval, hand, prop)
    cons = check_constraints(fs, sel_eval, hand, prop, costs, gate, a.allow_below_threshold, deltas)

    # 校准（calib 片；calibrate.py 缺失时跳过）
    cal: dict = {"n_calib": int(sel_calib.sum()), "stored": False}
    try:
        import calibrate

        d = decide_all(fs, prop)
        idx = np.where(sel_calib & d["has_any"])[0]
        scores = [float(min(1.0, max(0.0, d["total_top"][i]))) for i in idx]
        correct = [bool(d["ok_top"][i]) for i in idx]
        if scores:
            fit = calibrate.fit(scores, correct)
            decs = [("confident" if d["decision"][i] == DEC_CODES["confident"] else "non_confident") for i in idx]
            cc = calibrate.class_counts(decs) if hasattr(calibrate, "class_counts") else {}
            p_star = calibrate.decision_threshold(costs["c_fa"], costs["c_fr"])
            de = decide_all(fs, prop)
            idx_e = np.where(sel_eval & de["has_any"])[0]
            gate_eval = calibrate.gate_at(fit["breakpoints"], p_star,
                                         [float(min(1.0, max(0.0, de["total_top"][i]))) for i in idx_e],
                                         [bool(de["ok_top"][i]) for i in idx_e]) if idx_e.size else None
            need = cfg["thresholds"]
            cal.update({**{k: fit[k] for k in ("breakpoints", "n", "n_pos", "ece_before", "ece_after")},
                        "class_counts": cc, "p_star": p_star, "gate_eval_at_p_star": gate_eval,
                        "stored": bool(gate["calib_ok"] and all(v >= need.get("calib_each_class", 50) for v in cc.values()) if cc else gate["calib_ok"])})
    except ImportError:
        cal["note"] = "eval/calibrate.py 不存在，跳过校准"

    hard_ok = all(c["ok"] for c in cons if c["name"] != "sample_gate")
    verdict = "REJECTED" if not hard_ok else ("APPLICABLE" if gate["met"] or a.allow_below_threshold else "REPORT_ONLY")
    changed = {k: (hand[k], prop[k]) for k in REPLAYABLE if abs(hand[k] - prop[k]) > 1e-9}
    proposal = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": _git_head(),
        "features": str(a.features), "features_sha1": fmeta.get("features_sha1"), "features_meta": fmeta.get("sources"),
        "params_from": {"version": pfile["version"], "source": pfile["source"]},
        "config": cfg, "strict": a.strict,
        "gate": gate, "effective_counts": counts,
        "params_hand": hand, "params_proposed": prop, "changed": {k: {"hand": v[0], "proposed": v[1]} for k, v in changed.items()},
        "metrics": {name: {"hand": metrics(fs, hand, sel, costs), "proposed": metrics(fs, prop, sel, costs)}
                    for name, sel in (("train", sel_train), ("calib", sel_calib), ("eval", sel_eval))},
        "J": {"hand": round(obj_train.J(hand), 6), "proposed": found["best_J"]},
        "per_slice_eval": deltas,
        "identifiability": ident,
        "frontier_eval": front, "coverage_at_risk": {"hand": coverage_at_risk(front_hand, cfg.get("risk_targets", [0.01, 0.02, 0.05])),
                                                     "proposed": coverage_at_risk(front, cfg.get("risk_targets", [0.01, 0.02, 0.05]))},
        "calibration": cal, "top_optima": found["top_optima"],
        "constraints": cons, "confirm": {"ran": False},
        "verdict": verdict, "reasons": [c["name"] for c in cons if not c["ok"]],
        "elapsed": round(time.time() - t0, 1),
        "approximations": ["正确 = Top-1 链是真值链（拼装错不在环内）", "音频只用第 1 遍转写",
                           "STRONG_HIT / LONGER_NAME_TOL / UNIQUE_SHORT_MAX / MAX_DIST 不可重放，保持手设",
                           "合成集由同一代价矩阵生成，只约束下游"],
    }
    if a.confirm:
        proposal["confirm"] = confirm(prop, fs)
        if proposal["confirm"].get("regression") == "fail":
            proposal["verdict"] = "REJECTED"
            proposal["reasons"].append("regression")
        elif proposal["verdict"] == "APPLICABLE" and proposal["confirm"].get("regression") != "pass":
            proposal["verdict"] = "REPORT_ONLY"
            proposal["reasons"].append("confirm 未通过回归")
    elif proposal["verdict"] == "APPLICABLE":
        proposal["verdict"] = "REPORT_ONLY"
        proposal["reasons"].append("未 --confirm（apply 前必须真跑）")
    return proposal


def render_md(p: dict) -> str:
    L = ["---", "type: report", "title: 调参提案", f"timestamp: {p['created'][:10]}", "---", "",
         f"# 调参提案 {p['created']}  · verdict **{p['verdict']}**", "",
         f"- 代码 {p['git_commit']}，参数来自 v{p['params_from']['version']}（{p['params_from']['source']}），特征 {p['features_sha1']}",
         f"- 门槛：{p['gate']['detail']} → {'满足' if p['gate']['met'] else '未达门槛'}" + ("（--strict 排除草稿）" if p["strict"] else ""),
         f"- 目标：c_fa={p['config']['costs']['c_fa']} c_fr={p['config']['costs']['c_fr']}；J_hand={p['J']['hand']} J*={p['J']['proposed']}",
         "- 近似：" + "；".join(p["approximations"]), ""]
    L += ["## 参数", "", "| 参数 | 手设 | 提案 | train 平台 | eval 平台 | 数据钉住了？ |", "|---|---|---|---|---|---|"]
    for k, v in p["identifiability"].items():
        L.append(f"| {k} | {p['params_hand'][k]} | {p['params_proposed'][k]} | {v['train']['flat']} ({v['train']['flat_share']:.0%}) | "
                 f"{v['eval']['flat']} ({v['eval']['flat_share']:.0%}) | {'是' if v['pinned'] else '否（平）'} |")
    L += ["", "## 指标（手设 → 提案）", "", "| 片 | n | EM 代理 | coverage | risk | over_reject | cost | oov FA | 无地址 FA |", "|---|---|---|---|---|---|---|---|---|"]
    for name, mm in p["metrics"].items():
        h, q = mm["hand"], mm["proposed"]
        f = lambda k: f"{h[k]} → {q[k]}"
        L.append(f"| {name} | {h['n']} | {f('em_proxy')} | {f('coverage')} | {f('risk')} | {f('over_reject')} | {f('cost')} | {f('oov_fa')} | {f('neg_fa')} |")
    L += ["", "## 留出片上 risk 目标下的最大 coverage", "", "| 目标 risk | 手设 | 提案 |", "|---|---|---|"]
    for t, hv in p["coverage_at_risk"]["hand"].items():
        pv = p["coverage_at_risk"]["proposed"].get(t)
        L.append(f"| ≤{float(t):.0%} | {hv} | {pv} |")
    L += ["", "## 约束", "", "| 约束 | 通过 | 手设 | 提案 |", "|---|---|---|---|"]
    for c in p["constraints"]:
        L.append(f"| {c['name']} | {'✓' if c['ok'] else '✗'} | {c.get('hand', c.get('detail', ''))} | {c.get('proposed', '')} |")
    bad = [d for d in p["per_slice_eval"] if d["delta_tp"] < 0 or d["delta_fa"] > 0]
    L += ["", f"留出片逐分片：{len(p['per_slice_eval'])} 个分片，{len(bad)} 个退化" +
          ("：" + "; ".join(f"{d['slice']} ΔTP={d['delta_tp']} ΔFA={d['delta_fa']}" for d in bad[:10]) if bad else "")]
    cal = p["calibration"]
    L += ["", "## 校准", ""]
    if cal.get("breakpoints") is not None:
        L.append(f"calib n={cal['n_calib']}（{cal.get('class_counts')}）ECE {cal['ece_before']} → {cal['ece_after']}；"
                 f"p*={cal['p_star']}；若用 P≥p* 做闸门，eval 上 {cal.get('gate_eval_at_p_star')}；"
                 f"{'已写入参数文件' if cal['stored'] else '未达门槛，只报不存'}")
    else:
        L.append(cal.get("note", "无校准数据"))
    c = p["confirm"]
    L += ["", "## 真跑确认", ""]
    if c.get("ran"):
        L.append(f"回归：{c.get('regression')}" + (f"（{c.get('regression_fails')}）" if c.get("regression_fails") else ""))
        L.append("重放偏差（真跑四格 ≠ 重放四格）：" + ", ".join(f"{k} {v['diverged']}/{v['n']}" for k, v in c["replay_divergence"].items()))
        for k, v in c["run_eval"].items():
            L.append(f"- {k}: {v}")
    else:
        L.append("未运行（--confirm）")
    L += ["", f"## 结论：{p['verdict']}", "", "原因：" + ("、".join(p["reasons"]) if p["reasons"] else "全部约束通过")]
    if p["changed"]:
        L.append("变化：" + "，".join(f"{k} {v['hand']}→{v['proposed']}" for k, v in p["changed"].items()))
    else:
        L.append("提案参数与手设一致（目标在这些数据上是平的）")
    return "\n".join(L)


def apply(a: Namespace) -> None:
    prop = json.loads(Path(a.apply).read_text(encoding="utf-8"))
    if prop["verdict"] != "APPLICABLE":
        if not (a.allow_below_threshold and prop["verdict"] == "REPORT_ONLY" and
                all(c["ok"] for c in prop["constraints"] if c["name"] != "sample_gate") and prop["confirm"].get("regression") == "pass"):
            sys.exit(f"提案 verdict={prop['verdict']}（{prop['reasons']}），不能 apply")
    pfile = json.loads(PARAMS_FILE.read_text(encoding="utf-8"))
    new = dict(pfile)
    new["version"] = pfile["version"] + 1
    new["source"] = "tuned" if prop["gate"]["met"] else "tuned-below-threshold"
    new["created"] = time.strftime("%Y-%m-%d")
    new["note"] = f"由 {Path(a.apply).name} 应用；变化 {prop['changed']}"
    new["params"] = {k: (prop["params_proposed"][k] if k in prop["params_proposed"] else v) for k, v in pfile["params"].items()}
    new["below_threshold"] = not prop["gate"]["met"]
    new["proposal"] = str(Path(a.apply).relative_to(ROOT)) if str(a.apply).startswith(str(ROOT)) else str(a.apply)
    new["data_snapshot"] = {**pfile.get("data_snapshot", {}), "features": prop.get("features_sha1")}
    cal = prop.get("calibration", {})
    new["calibration"] = {"breakpoints": cal["breakpoints"], "ece_before": cal["ece_before"], "ece_after": cal["ece_after"],
                          "n": cal["n"], "fitted": prop["created"]} if cal.get("stored") else pfile.get("calibration")
    new.setdefault("history", []).append({"version": new["version"], "source": new["source"], "created": new["created"],
                                          "proposal": new["proposal"], "changed": prop["changed"]})
    PARAMS_FILE.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {PARAMS_FILE.relative_to(ROOT)} v{new['version']}（{new['source']}）；接下来跑 eval/regression.py --update-golden 并提交")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(FEATURES))
    ap.add_argument("--confirm", action="store_true", help="用候选参数真跑 run_eval + 回归，量化重放偏差")
    ap.add_argument("--apply", default=None, metavar="PROPOSAL_JSON")
    ap.add_argument("--allow-below-threshold", action="store_true")
    ap.add_argument("--strict", action="store_true", help="排除 label_status=draft 的录音")
    ap.add_argument("--out", default=str(TUNING_DIR))
    a = ap.parse_args()
    if a.apply:
        apply(a)
        return
    if not Path(a.features).exists():
        sys.exit(f"没有特征文件 {a.features}：先跑 python eval/features.py")
    p = propose(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    js = out / f"{stamp}_proposal.json"
    js.write_text(json.dumps(p, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / f"{stamp}_proposal.md").write_text(render_md(p), encoding="utf-8")
    print(render_md(p))
    print(f"\n提案: {js}")


if __name__ == "__main__":
    main()
