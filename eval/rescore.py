"""特征重放：对 features.py 存下来的候选链，用任意 (权重, 阈值) 重算总分、重走三道闸门。

与 rank.decide 同一顺序：相似度闸门 → 孤证闸门 → 只到省市（partial）→ 分差闸门（异区才拦）。
一次 decide_all 在 800 条样本上不到 1 毫秒，调参搜索才跑得起来。

近似：正确与否按 "Top-1 链是真值链" 算（拼装不在环内），无地址样本永远算错（放行即错送）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHT_KEYS = ("W_SIM", "W_COV", "W_PRIOR", "W_DEPTH", "W_CONFLICT")
DEC_CODES = {"confident": 0, "ambiguous": 1, "partial": 2, "reject": 3, "empty": 4}
DEC_NAMES = {v: k for k, v in DEC_CODES.items()}


@dataclass
class FeatureSet:
    ids: list[str]
    meta: list[dict]
    F: np.ndarray            # (N, C, 5)
    mask: np.ndarray         # (N, C) bool
    ok: np.ndarray           # (N, C) bool
    admin: np.ndarray        # (N, C) int
    deep: np.ndarray         # (N, C) bool
    n_ev: np.ndarray         # (N, C) int
    shd: np.ndarray          # (N, C) float, nan 表示不是孤证
    has_addr: np.ndarray     # (N,) bool
    split: np.ndarray        # (N,) str
    source: np.ndarray       # (N,) str
    decision_live: np.ndarray  # (N,) int code
    top_live: np.ndarray     # (N,) int, -1 表示无候选

    @property
    def n(self) -> int:
        return len(self.ids)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "FeatureSet":
        N = len(rows)
        C = max((len(r["chains"]) for r in rows), default=1) or 1
        F = np.zeros((N, C, 5), dtype=np.float64)
        mask = np.zeros((N, C), dtype=bool)
        ok = np.zeros((N, C), dtype=bool)
        admin = np.zeros((N, C), dtype=np.int64)
        deep = np.zeros((N, C), dtype=bool)
        n_ev = np.zeros((N, C), dtype=np.int64)
        shd = np.full((N, C), np.nan)
        admin_ids: dict[str, int] = {}
        for i, r in enumerate(rows):
            for j, c in enumerate(r["chains"]):
                F[i, j] = c["f"]
                mask[i, j] = True
                ok[i, j] = bool(c["ok"])
                admin[i, j] = admin_ids.setdefault(c["admin"], len(admin_ids) + 1)
                deep[i, j] = bool(c["deep"])
                n_ev[i, j] = int(c["n_ev"])
                if c.get("shd") is not None:
                    shd[i, j] = float(c["shd"])
        return cls(
            ids=[r["id"] for r in rows], meta=[r.get("meta", {}) for r in rows],
            F=F, mask=mask, ok=ok, admin=admin, deep=deep, n_ev=n_ev, shd=shd,
            has_addr=np.array([bool(r.get("meta", {}).get("has_address", True)) for r in rows]),
            split=np.array([r.get("split", "eval") for r in rows]),
            source=np.array([r.get("source", "") for r in rows]),
            decision_live=np.array([DEC_CODES.get(r.get("decision_live"), 3) for r in rows]),
            top_live=np.array([r["top_live"] if r.get("top_live") is not None else -1 for r in rows]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls.from_rows(rows)

    def subset(self, idx: np.ndarray) -> "FeatureSet":
        idx = np.asarray(idx)
        return FeatureSet(
            ids=[self.ids[i] for i in idx], meta=[self.meta[i] for i in idx],
            F=self.F[idx], mask=self.mask[idx], ok=self.ok[idx], admin=self.admin[idx], deep=self.deep[idx],
            n_ev=self.n_ev[idx], shd=self.shd[idx], has_addr=self.has_addr[idx], split=self.split[idx],
            source=self.source[idx], decision_live=self.decision_live[idx], top_live=self.top_live[idx],
        )

    def where(self, **kw) -> np.ndarray:
        """按 meta 字段筛选：where(split="train")、where(negative_type="oov_db")。"""
        sel = np.ones(self.n, dtype=bool)
        for k, v in kw.items():
            if k == "split":
                sel &= self.split == v
            elif k == "source":
                sel &= self.source == v
            else:
                sel &= np.array([m.get(k) == v for m in self.meta])
        return sel


# --------------------------------------------------------------------------


def decide_all(fs: FeatureSet, p: dict) -> dict:
    """任意参数下的决策。返回 dict of arrays：decision（code）、top、ok_top、total_top、margin、sim_top。"""
    w = np.array([p["W_SIM"], p["W_COV"], p["W_PRIOR"], p["W_DEPTH"], -p["W_CONFLICT"]])
    total = fs.F @ w
    total = np.where(fs.mask, total, -np.inf)
    N, C = total.shape
    if C == 1:
        order = np.zeros((N, 1), dtype=np.int64)
    else:
        order = np.argsort(-total, axis=1, kind="stable")     # 稳定排序：同分保持 rank() 的原顺序
    rows = np.arange(N)
    top = order[:, 0]
    second = order[:, 1] if C > 1 else top
    has_any = fs.mask[rows, top]
    has_second = fs.mask[rows, second] & (C > 1)
    t_top = total[rows, top]
    t_second = np.where(has_second, total[rows, second], 0.0)     # 无第二名时不参与相减，避免 -inf - -inf
    margin = np.where(has_second, np.where(has_any, t_top, 0.0) - t_second, 1.0)
    sim_top = fs.F[rows, top, 0]
    cov_top = fs.F[rows, top, 1]
    deep_top = fs.deep[rows, top]
    nev_top = fs.n_ev[rows, top]
    shd_top = fs.shd[rows, top]
    admin_differs = fs.admin[rows, top] != fs.admin[rows, second]

    # 孤证闸门：整句只有一处近似命中（dist>0）且覆盖率低于阈值
    single = (nev_top == 1) & (np.nan_to_num(shd_top, nan=0.0) > 1e-9) & (cov_top < p["SINGLE_HIT_MIN_COV"])
    reject = (sim_top < p["SIM_MIN"]) | single
    partial = ~reject & ~deep_top
    ambiguous = ~reject & ~partial & has_second & (margin < p["MARGIN_MIN"]) & admin_differs
    decision = np.full(N, DEC_CODES["confident"], dtype=np.int64)
    decision[ambiguous] = DEC_CODES["ambiguous"]
    decision[partial] = DEC_CODES["partial"]
    decision[reject] = DEC_CODES["reject"]
    decision[~has_any] = fs.decision_live[~has_any]        # 没有候选：沿用现场决策（empty/reject）
    ok_top = np.where(has_any, fs.ok[rows, top], False) & fs.has_addr
    return {"decision": decision, "top": np.where(has_any, top, -1), "ok_top": ok_top,
            "total_top": np.where(has_any, t_top, np.nan), "margin": margin, "sim_top": np.where(has_any, sim_top, np.nan),
            "has_any": has_any}


def quadrants(fs: FeatureSet, p: dict, d: dict | None = None) -> np.ndarray:
    """TP/FA/FR/TR，与 stages.quadrant 一致；无地址样本：放行 = FA，拦下 = TR。"""
    d = d or decide_all(fs, p)
    auto = d["decision"] == DEC_CODES["confident"]
    ok = d["ok_top"]
    q = np.where(auto, np.where(ok, "TP", "FA"), np.where(ok, "FR", "TR"))
    return q


def metrics(fs: FeatureSet, p: dict, sel: np.ndarray | None = None, costs: dict | None = None,
            d: dict | None = None) -> dict:
    costs = costs or {"c_fa": 10, "c_fr": 1, "c_tr": 1, "c_miss": 0}
    d = d or decide_all(fs, p)
    q = quadrants(fs, p, d)
    if sel is None:
        sel = np.ones(fs.n, dtype=bool)
    q = q[sel]
    n = int(sel.sum())
    tp, fa, fr, tr = (int((q == k).sum()) for k in ("TP", "FA", "FR", "TR"))
    cost = (costs["c_fa"] * fa + costs["c_fr"] * fr + (costs["c_tr"] + costs.get("c_miss", 0)) * tr) / n if n else 0.0
    oov = sel & fs.where(negative_type="oov_db")
    neg = sel & ~fs.has_addr
    qa = quadrants(fs, p, d)
    return {
        "n": n, "TP": tp, "FA": fa, "FR": fr, "TR": tr,
        "em_proxy": round((tp + fr) / n, 4) if n else None,
        "coverage": round((tp + fa) / n, 4) if n else None,
        "risk": round(fa / (tp + fa), 4) if (tp + fa) else None,
        "over_reject": round(fr / (tp + fr), 4) if (tp + fr) else None,
        "cost": round(cost, 5),
        "oov_fa": int(((qa == "FA") & oov).sum()),
        "neg_fa": int(((qa == "FA") & neg).sum()),
    }


def params_vector(p: dict, keys: tuple[str, ...]) -> np.ndarray:
    return np.array([p[k] for k in keys], dtype=np.float64)
