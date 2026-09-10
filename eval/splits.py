"""划分：eval / calib / train 物理分开，data/eval/splits.json 是唯一来源（评测体系设计 §2.5）。

    split_of("read_item", "003")                → "eval"
    split_of("synthetic", "003")                → 按 source_id 分组查表
    split_of("recording", "recordings/x.wav")   → 自发录音全部 eval（超过 150 条后按说话人分）

有效样本按"句"计：合成集同一句的 30 个变体只算 1 条，否则 720 会把门槛（train ≥ 500）虚假地凑满。
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_PATH = ROOT / "data" / "eval" / "splits.json"
SPLITS = ("train", "calib", "eval")


@lru_cache(maxsize=1)
def load_splits(path: str | None = None) -> dict:
    return json.loads(Path(path or SPLITS_PATH).read_text(encoding="utf-8"))


def split_of(kind: str, key: str, speaker: str | None = None, n_spontaneous: int = 0) -> str:
    sp = load_splits()
    if kind == "read_item":
        return sp.get("read_items", {}).get("split", "eval")
    if kind == "synthetic":
        groups = sp.get("synthetic_by_source", {})
        for name in SPLITS:
            if key in groups.get(name, []):
                return name
        return "eval"
    if kind == "recording":
        rec = sp.get("recordings", {})
        ov = rec.get("overrides", {})
        if key in ov:
            return ov[key]
        # 门槛前全部 eval；门槛后按说话人（overrides 里写死，避免随录音数变动而漂移）
        return "eval"
    raise ValueError(f"未知样本类型 {kind}")


def check_synthetic_partition(source_ids: list[str]) -> list[str]:
    """合成集分组必须恰好把全部 source_id 分一次：不重不漏。"""
    groups = load_splits().get("synthetic_by_source", {})
    errs = []
    seen: Counter = Counter()
    for name in SPLITS:
        for sid in groups.get(name, []):
            seen[sid] += 1
    for sid, c in seen.items():
        if c > 1:
            errs.append(f"synthetic source_id {sid} 出现在 {c} 个划分里")
    for sid in source_ids:
        if sid not in seen:
            errs.append(f"synthetic source_id {sid} 没有划分")
    return errs


def effective_counts(records: list[dict]) -> dict[str, dict]:
    """按划分统计有效样本：text 1 条 = 1，录音 1 条 = 1，合成同一 source_id 算 1。

    records 每条至少有 split / source（text|synthetic|oov|audio）/ group（去重键）/ label_status。
    返回 {split: {"effective": n, "raw": n, "drafts": n, "by_difficulty": {...}}}。
    """
    out: dict[str, dict] = {s: {"effective": 0, "raw": 0, "drafts": 0, "groups": set(), "by_difficulty": Counter()}
                            for s in SPLITS}
    for r in records:
        s = r.get("split", "eval")
        if s not in out:
            continue
        o = out[s]
        o["raw"] += 1
        if r.get("label_status") == "draft":
            o["drafts"] += 1
        g = f"{r.get('source')}:{r.get('group') or r.get('id')}"
        if g not in o["groups"]:
            o["groups"].add(g)
            o["effective"] += 1
            for d in r.get("difficulty") or []:
                o["by_difficulty"][d] += 1
    for o in out.values():
        o["groups"] = len(o["groups"])
        o["by_difficulty"] = dict(o["by_difficulty"])
    return out


def gate_status(counts: dict[str, dict], thresholds: dict | None = None) -> dict:
    """§2.6 门槛：train_eff ≥ 500 且各 difficulty ≥ 30；calib_eff ≥ 300。"""
    thr = thresholds or load_splits().get("thresholds", {})
    train, calib = counts.get("train", {}), counts.get("calib", {})
    need_train, need_calib, per_diff = thr.get("train", 500), thr.get("calib", 300), thr.get("per_difficulty", 30)
    weak_tags = {d: n for d, n in train.get("by_difficulty", {}).items() if n < per_diff}
    train_ok = train.get("effective", 0) >= need_train and not weak_tags
    calib_ok = calib.get("effective", 0) >= need_calib
    return {
        "train_eff": train.get("effective", 0), "train_need": need_train, "train_ok": train_ok,
        "calib_eff": calib.get("effective", 0), "calib_need": need_calib, "calib_ok": calib_ok,
        "weak_difficulty_tags": weak_tags,
        "met": train_ok,
        "detail": (f"train_eff={train.get('effective', 0)}/{need_train}, calib_eff={calib.get('effective', 0)}/{need_calib}"
                   + (f", difficulty 不足 30 的标签 {len(weak_tags)} 个" if weak_tags else "")),
    }
