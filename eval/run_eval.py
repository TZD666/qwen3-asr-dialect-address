#!/usr/bin/env python3
"""评测：分阶段、分片、带负样本与 oracle 上限（评测体系设计 v2）。

不微调，ASR 的错误分布就是固定输入，能改的只有后处理。评测因此只回答两件事：
  1. 后处理在这个固定错误分布上的**净收益**（纠对 − 改坏）
  2. 每个环节的**天花板**各有多高（oracle），从而决定先改哪一个

三种模式
--------
    --mode text      朗读稿当"完美 ASR 输出"跑后处理（不需要模型）
    --mode audio     真实音频跑完整流水线；同一遍里顺带算出 baseline（第 1 遍裸转写 + 数字归一化）做配对
    --mode baseline  真实音频只做裸 ASR + 数字归一化

常用
----
    python eval/run_eval.py --mode text
    python eval/run_eval.py --mode text --negatives oov            # 库缺负样本：临时删条目
    python eval/run_eval.py --mode text --oracle all               # oracle 表（db / rank / gate / 叠加）
    python eval/run_eval.py --mode text --eval data/eval/synthetic/perturbed.jsonl --tag synthetic
    python eval/run_eval.py --mode audio                           # 按 manifest 跑全部可用录音（走 ASR 缓存）
    python eval/run_eval.py --mode audio --negatives injection     # 注入负样本回归

报告固定七节（§8），json 保留每条样本的全部中间量。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr.address_db import AddressDB, AddressEntry  # noqa: E402
from dialect_addr.normalize import normalize as num_normalize  # noqa: E402
from dialect_addr.pipeline import Pipeline  # noqa: E402
from dialect_addr import rank as rank_mod  # noqa: E402
from dialect_addr.rank import Chain, RankResult, decide  # noqa: E402

from attribution import LABELS, attribute  # noqa: E402
from stages import (  # noqa: E402
    GEO, GoldChain, chain_matches, gold_chain, replay_decision, replay_features,
    score_e2e, stage_a, stage_b, stage_c, stage_d, stage_e, stage_f,
)
from stats import MIN_N_FOR_PCT, bootstrap_diff, ece, fmt_rate, mcnemar, rule_of_three  # noqa: E402

AUDIO_ROOT = ROOT / "data" / "eval" / "audio"
MANIFEST = ROOT / "data" / "eval" / "manifest.jsonl"
NEG_DIR = ROOT / "data" / "eval" / "negatives"
GOLDEN_DIR = ROOT / "eval" / "golden"
AXES = ("dialect_group", "address_depth", "noise", "difficulty", "negative_type", "short_name", "rule")
OPTIONAL_AXES = ("rule",)      # 只有合成集才有；值为空的样本不进这个轴
GRID_MARGIN = [round(0.02 + 0.02 * i, 2) for i in range(10)]        # 0.02 … 0.20
GRID_SIM = [round(0.50 + 0.02 * i, 2) for i in range(16)]           # 0.50 … 0.80
STAGE_METRICS = {
    "A": [("cer_char", "mean"), ("cer_phon", "mean"), ("cer_name_char", "mean"), ("cer_name_phon", "mean"),
          ("recoverable_share", "mean")],
    "B": [("lexicon_hit", "rate"), ("number_ok", "rate"), ("tail_exact", "rate"),
          ("illegal_number_caught", "rate"), ("filler_leak", "mean")],
    "C": [("recall_hit", "rate"), ("recall_chain@1", "rate"), ("recall_chain@3", "rate"), ("recall_chain@all", "rate")],
    "D": [("top1_correct", "rate"), ("mrr", "mean")],
    "E": [("coverage", "special"), ("risk", "special"), ("over_reject", "special"), ("nbest_recall@3", "special")],
    "F": [("backfill_correct", "rate"), ("wrong_hit_wrong_city", "rate"), ("backfill_confidence_gap", "rate")],
}


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------


def load_items(path: Path) -> tuple[dict, list[dict]]:
    """评测集 json（{_meta, items}）或 jsonl（每行一个 item，合成集）。"""
    if path.suffix == ".jsonl":
        items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cfg = path.with_name("gen_config.json")
        meta = json.loads(cfg.read_text(encoding="utf-8")) if cfg.exists() else {}
        meta.setdefault("name", path.stem)
        return meta, items
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("_meta", {"name": path.stem}), data["items"]


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def manifest_to_items(rows: list[dict]) -> list[dict]:
    """清单行 → 评测 item（字段名对齐 xinan_guanhua.json）。未标注的行也进，但 scored=False。"""
    items = []
    for r in rows:
        if r.get("quality") != "ok":
            continue
        has_addr = r.get("has_address", True)
        labeled = r.get("label_status") in ("confirmed", "draft") and (r.get("address_gold") or not has_addr)
        items.append({
            "id": Path(r["file"]).stem, "file": r["file"], "city": "",
            "spoken": r.get("transcript_gold") or r.get("transcript_asr", ""),
            "transcript_gold": r.get("transcript_gold", ""),
            "ground_truth": r.get("address_gold", ""), "fields": r.get("fields_gold") or {},
            "difficulty": [], "notes": r.get("notes", ""),
            "dialect_group": r.get("dialect_group", ""), "sub_dialect": r.get("sub_dialect", ""),
            "address_depth": r.get("address_depth", ""), "noise": r.get("noise", ""),
            "orthography_expected": r.get("orthography_expected", ""), "gold_in_db": r.get("gold_in_db"),
            "negative_type": r.get("negative_type"), "split": r.get("split", "eval"),
            "set": r.get("set", "spontaneous"), "speaker_id": r.get("speaker_id", ""),
            "has_address": has_addr, "scored": bool(labeled), "label_status": r.get("label_status"),
            "quality": r.get("quality", "ok"),
        })
    return items


# --------------------------------------------------------------------------
# 地址库改造：负样本删条目 / oracle 插条目
# --------------------------------------------------------------------------


def db_without(db: AddressDB, drops: list[dict]) -> tuple[AddressDB, list[str]]:
    """按 oov_drop.json 删条目。parent 可以是父条目名或 adcode，缺省则删全部同名。"""
    removed: list[str] = []
    keep = []
    for e in db.entries:
        hit = False
        for d in drops:
            if e.name != d["name"] and d["name"] not in e.aliases:
                continue
            par = d.get("parent")
            if par:
                pe = db.by_adcode.get(e.parent or "")
                if not (e.parent == par or (pe and pe.name == par)):
                    continue
            hit = True
            break
        if hit:
            removed.append(f"{e.name}({e.adcode})")
        else:
            keep.append(e)
    return AddressDB(keep, provider=db.provider), removed


def db_with_gold(db: AddressDB, items: list[dict]) -> tuple[AddressDB, list[str]]:
    """oracle db：把真值里库缺的路级条目临时插进去（挂在真值区/市下）。"""
    added: list[AddressEntry] = []
    names: list[str] = []
    seen = set()
    for it in items:
        if not it.get("fields") or not it.get("has_address", True):
            continue
        g = gold_chain(db, it["fields"])
        parent = g.admin.get("district") or g.admin.get("city") or g.admin.get("province")
        if parent is None:
            continue
        for nm in g.missing:
            key = (nm, parent.adcode)
            if key in seen:
                continue
            seen.add(key)
            added.append(AddressEntry(name=nm, level="road", adcode=f"oracle-{len(added):03d}",
                                      parent=parent.adcode, prior=0.6))
            names.append(f"{nm}←{parent.name}")
    return AddressDB(list(db.entries) + added, provider=db.provider), names


def oracle_rank_hook(gold_by_text: dict[str, GoldChain]):
    """oracle rank：真值链在候选里时强制换到 Top-1，再按原闸门重新决策。"""
    def hook(rr: RankResult) -> RankResult:
        g = gold_by_text.get(rr.geo_text)
        if g is None or not rr.nbest:
            return rr
        pos = next((i for i, c in enumerate(rr.nbest) if chain_matches(c, g)), None)
        if pos is None or pos == 0:
            return rr
        nb = [rr.nbest[pos]] + [c for i, c in enumerate(rr.nbest) if i != pos]
        decision, reason = decide(nb)
        return RankResult(decision, nb[0], nb, rr.geo_text, rr.dialect, rr.space_name,
                          f"[oracle rank] {reason}", rr.all_hits)
    return hook


# --------------------------------------------------------------------------
# 单条评测
# --------------------------------------------------------------------------


def _chain_json(ch: Chain) -> dict:
    return {"name": ch.full_name(), "total": round(ch.total, 4), "sim": round(ch.sim, 3),
            "cov": round(ch.coverage, 3), "prior": round(ch.prior, 3), "depth": round(ch.depth, 3),
            "conflict": round(ch.conflict, 3),
            "hits": [{"level": lv, "name": h.entry.name, "matched": h.matched_name, "dist": round(h.dist, 3)}
                     for lv, h in ch.hits.items()] +
                    [{"level": h.entry.level + "+", "name": h.entry.name, "matched": h.matched_name, "dist": round(h.dist, 3)}
                     for h in ch.extra]}


def _gold_name_variants(db: AddressDB, gold: GoldChain, fields: dict) -> list[list[str]]:
    out = []
    for k in GEO:
        v = fields.get(k, "")
        if not v:
            continue
        variants = [v]
        for e in db.entries:
            if e.adcode in gold.road_codes and (e.name == v or v in e.aliases):
                variants = list(e.all_names())
                break
        out.append(variants)
    return out


def evaluate(rec: dict, res, truth: dict, db: AddressDB, dialect: str | None, mode: str,
             oracle_gate: bool, raw_for_a: str) -> None:
    """把流水线结果打成六段分数 + 端到端 + 归因，写进 rec。"""
    fields = truth.get("fields") or {}
    if not truth.get("has_address", True):
        # 无地址负样本：期望不自动通过（ambiguous/partial 会交给人确认，不算错送）
        dec = rec["decision"]
        ok = dec != "confident"
        rec["e2e"] = {"exact": ok, "deliverable": ok, "geo_ok": ok, "admin_all": ok, "geo_recall": 1.0,
                      "tail_all": True, "cer": 0.0, "severity": "正确" if ok else "严重(无地址却输出)"}
        feats = replay_features(res.final.ranking) if res is not None else {"empty": True, "decision": dec}
        rec["replay"] = feats
        rec["stage_e"] = stage_e(dec, ok, ok, feats, True)
        # 无地址：拦下来就是"有效拦截"（TR），不是"多余确认"；放行就是错送（FA）
        rec["stage_e"]["quadrant"] = rec["stage_e"]["quadrant_geo"] = "TR" if ok else "FA"
        rec["attribution"], rec["attribution_detail"] = (None, "") if ok else ("gate_over_accept", f"无地址却 {dec}")
        return

    gold = gold_chain(db, fields)
    rec["gold_in_db"] = gold.in_db
    rec["gold_deepest"] = gold.deepest.name if gold.deepest else ""
    rec["gold_deepest_name"] = gold.deepest_name
    rec["gold_missing"] = gold.missing
    rec["short_name"] = gold.short_name
    e2e = score_e2e(rec["fields"], rec["pred"], truth)
    rec["e2e"] = e2e

    if mode != "text" or rec.get("stage_a_force"):
        rec["stage_a"] = stage_a(raw_for_a, truth.get("transcript_gold", ""),
                                 _gold_name_variants(db, gold, fields), dialect)
    if res is None:                     # baseline：没有检索/排序/闸门
        rec["stage_b"] = stage_b(raw_for_a, [], rec["pred"], rec["fields"], truth)
        feats = {"empty": True, "decision": "baseline"}
        rec["replay"] = feats
        rec["stage_e"] = stage_e("baseline", e2e["deliverable"], e2e["geo_ok"], feats, e2e["tail_all"])
        rec["stage_e"]["quadrant"] = "TP" if e2e["deliverable"] else "FA"
        rec["stage_e"]["quadrant_geo"] = "TP" if e2e["geo_ok"] else "FA"
        rec["attribution"], rec["attribution_detail"] = attribute(rec)
        return

    final = res.final
    rec["stage_b"] = stage_b(final.raw_text, final.lex_subs, rec["pred"], rec["fields"], truth)
    c = stage_c(final.ranking, gold, db, dialect)
    d = stage_d(final.ranking, gold, c)
    c.pop("gold_chain_obj", None)
    rec["stage_c"], rec["stage_d"] = c, d
    feats = replay_features(final.ranking)
    rec["replay"] = feats
    decision = rec["decision"]
    # 闸门只看得见候选链，看不见拼装：归因用的"地名链对不对"按 Top-1 链算（可评时），
    # 否则退到行政区全对 + 地名全部出现。闲话泄漏进地址串不是闸门的错。
    if d.get("evaluable"):
        ok_chain = bool(d.get("top1_correct"))
    else:
        ok_chain = bool(e2e["admin_all"] and e2e["geo_recall"] == 1.0)
    rec["ok_chain"] = ok_chain
    if oracle_gate:
        decision = "confident" if ok_chain else "ambiguous"
        rec["decision_oracle_gate"] = decision
    rec["stage_e"] = stage_e(decision, e2e["deliverable"], ok_chain, feats, e2e["tail_all"])
    rec["stage_f"] = stage_f(rec["fields"], truth, truth.get("transcript_gold") or truth.get("spoken", ""),
                             res.segments, rec.get("address_depth", ""), e2e)
    # 库缺项：说出来的名字（去掉 街道/镇/路 这类后缀的词干）是否原样留在了输出里；
    # 没留下且自动通过 = 过度纠正（被改成了库里某个邻居）
    if gold.missing:
        out = rec["pred"] + rec["fields"].get("unverified", "")
        stems = [_stem(m) for m in gold.missing]
        rec["oov_preserved"] = all(s in out for s in stems)
        rec["over_correction"] = (rec["stage_e"]["quadrant"] == "FA") and not rec["oov_preserved"]
    rec["attribution"], rec["attribution_detail"] = attribute(rec)


_SUFFIXES = ("街道", "步行街", "小区", "大道", "路", "街", "巷", "道", "镇", "村", "区", "县", "市")


def _stem(name: str) -> str:
    for s in _SUFFIXES:
        if name.endswith(s) and len(name) > len(s) + 1:
            return name[: -len(s)]
    return name


# --------------------------------------------------------------------------
# 跑一遍
# --------------------------------------------------------------------------


def run_once(mode: str, items: list[dict], db: AddressDB, asr, dialect: str | None,
             two_pass: bool, oracle: set[str], verbose: bool = True) -> list[dict]:
    hook = None
    if "rank" in oracle:
        # 以 geo_text 为键在 rank 挂钩里找真值链；geo_text 是归一化后剥掉门牌的地名段
        gold_by_text: dict[str, GoldChain] = {}
        probe = Pipeline(db=db, asr=None, two_pass=False)  # type: ignore[arg-type]
        for it in items:
            if it.get("fields") and it.get("has_address", True):
                r0 = probe.process_text(it["spoken"], dialect)
                gold_by_text[r0.final.ranking.geo_text] = gold_chain(db, it["fields"])
        hook = oracle_rank_hook(gold_by_text)
    pipe = Pipeline(db=db, asr=asr, two_pass=(two_pass and mode == "audio"), rank_hook=hook)  # type: ignore[arg-type]

    rows = []
    for it in items:
        rec: dict = {k: it.get(k) for k in ("id", "city", "difficulty", "dialect_group", "sub_dialect", "address_depth",
                                             "noise", "orthography_expected", "negative_type", "split", "set",
                                             "notes", "rule", "source_id", "label_status")}
        rec["truth"] = it.get("ground_truth", "")
        rec["scored"] = it.get("scored", True)
        rec["quality"] = it.get("quality", "ok")
        rec["has_address"] = it.get("has_address", True)
        res = None
        raw_for_a = it["spoken"]
        if mode == "text":
            res = pipe.process_text(it["spoken"], dialect)
            rec.update(raw=it["spoken"], pred=res.address, fields=res.fields, decision=res.final.decision,
                       reason=res.final.ranking.reason, chosen="text", lang=None)
            used_dialect = dialect
        else:
            ap = AUDIO_ROOT / it["file"] if it.get("file") else None
            if ap is None or not ap.exists():
                rec.update(raw="", pred="", fields={}, decision="no_audio", chosen="-", skipped=True)
                rows.append(rec)
                continue
            if mode == "baseline":
                a = asr.transcribe(str(ap), language=None)
                norm, tail = num_normalize(a.text)
                rec.update(raw=a.text, pred=norm, fields=tail.as_dict(), decision="baseline", chosen="baseline",
                           lang=a.language, reason="")
                raw_for_a = a.text
                from dialect_addr.asr import normalize_dialect_label
                used_dialect = normalize_dialect_label(a.language) or dialect
            else:
                res = pipe.process(str(ap), dialect_hint=dialect)
                rec.update(raw=res.pass1.raw_text, pred=res.address, fields=res.fields, decision=res.final.decision,
                           reason=res.final.ranking.reason, chosen=res.chosen, lang=res.dialect,
                           pass2_raw=res.pass2.raw_text if res.pass2 else "",
                           ctx_used=bool(res.pass2 and res.pass2.asr and res.pass2.asr.context_used),
                           pass1_decision=res.pass1.decision, pass2_decision=res.pass2.decision if res.pass2 else None)
                raw_for_a = res.pass1.raw_text
                used_dialect = res.dialect
                # 配对用 baseline：第 1 遍裸转写 + 数字归一化，同一段音频同一次解码
                bnorm, btail = num_normalize(res.pass1.raw_text)
                rec["baseline_pred"], rec["baseline_fields"] = bnorm, btail.as_dict()
        if res is not None:
            rec["nbest"] = [_chain_json(c) for c in res.final.ranking.nbest[:5]]
            rec["n_hits"] = len(res.final.ranking.all_hits)
            rec["segments"] = res.segments
        if rec["scored"]:
            evaluate(rec, res, it, db, used_dialect, mode, "gate" in oracle, raw_for_a)
            if mode == "audio" and rec.get("has_address", True) and it.get("fields"):
                rec["baseline_e2e"] = score_e2e(rec["baseline_fields"], rec["baseline_pred"], it)
        rows.append(rec)
        if verbose:
            if not rec["scored"]:
                mark, tag = "?", "未标注"
            else:
                s = rec["e2e"]
                mark = "✓" if s["exact"] else ("~" if s.get("admin_all") else "✗")
                tag = rec.get("attribution") or ""
            print(f"{mark} {rec['id']} [{rec['decision']:9}] {rec['pred']}  {tag}")
            if rec["scored"] and not rec["e2e"]["exact"]:
                print(f"      真值 {rec['truth']}   {rec.get('attribution_detail', '')[:120]}")
    return rows


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def _slices(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {"全部": {"全部": rows}}
    for ax in AXES:
        d: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if ax == "difficulty":
                for t in r.get("difficulty") or []:
                    d[t].append(r)
            elif ax == "short_name":
                if r.get("short_name"):
                    d["短名(≤2音节)"].append(r)
            else:
                v = r.get(ax)
                if v is None and ax in OPTIONAL_AXES:
                    continue
                d[str(v) if v is not None else "null"].append(r)
        if d:
            out[ax] = dict(d)
    return out


def _rate(rows: list[dict], key: str, sub: str | None = None) -> tuple[int, int]:
    k = n = 0
    for r in rows:
        v = r.get(sub, {}) if sub else r
        if key in v and v[key] is not None and (not sub or v.get("evaluable", True)):
            n += 1
            k += 1 if v[key] else 0
    return k, n


def _mean(rows: list[dict], key: str, sub: str) -> tuple[float | None, int]:
    vals = [r[sub][key] for r in rows if r.get(sub) and r[sub].get(key) is not None]
    return (round(sum(vals) / len(vals), 3), len(vals)) if vals else (None, 0)


def e_metrics(rows: list[dict]) -> dict:
    q = Counter(r["stage_e"]["quadrant"] for r in rows if r.get("stage_e"))
    tp, fa, fr, tr = q["TP"], q["FA"], q["FR"], q["TR"]
    n = tp + fa + fr + tr
    m = {"TP": tp, "FA": fa, "FR": fr, "TR": tr, "n": n}
    m["coverage"] = (tp + fa, n)
    m["risk"] = (fa, tp + fa)
    m["over_reject"] = (fr, tp + fr)
    held = [r for r in rows if r.get("stage_e") and r["stage_e"]["quadrant"] in ("FR", "TR") and r.get("stage_c", {}).get("evaluable")]
    m["nbest_recall@3"] = (sum(1 for r in held if r["stage_c"].get("recall_chain@3")), len(held))
    return m


def summarize(rows: list[dict]) -> dict:
    """分片汇总，报告和 golden 都用它。只含计数，不含逐条。"""
    scored = [r for r in rows if r.get("scored") and not r.get("skipped")]
    sl = _slices(scored)
    out: dict = {"n": len(scored), "n_unscored": sum(1 for r in rows if not r.get("scored")), "slices": {}}
    for ax, groups in sl.items():
        for name, rs in groups.items():
            s: dict = {"n": len(rs)}
            s["exact"] = _rate(rs, "exact", "e2e")
            s["deliverable"] = _rate(rs, "deliverable", "e2e")
            s["admin_all"] = _rate(rs, "admin_all", "e2e")
            s["tail_all"] = _rate(rs, "tail_all", "e2e")
            s["geo_recall_mean"] = _mean(rs, "geo_recall", "e2e")
            for st, metrics in STAGE_METRICS.items():
                for key, kind in metrics:
                    if kind == "rate":
                        s[f"{st}.{key}"] = _rate(rs, key, f"stage_{st.lower()}")
                    elif kind == "mean":
                        s[f"{st}.{key}"] = _mean(rs, key, f"stage_{st.lower()}")
            em = e_metrics(rs)
            for key in ("coverage", "risk", "over_reject", "nbest_recall@3"):
                s[f"E.{key}"] = em[key]
            s["E.quadrant"] = {k: em[k] for k in ("TP", "FA", "FR", "TR")}
            s["attribution"] = dict(Counter(r["attribution"] for r in rs if r.get("attribution")))
            s["decisions"] = dict(Counter(r["decision"] for r in rs))
            s["over_correction"] = sum(1 for r in rs if r.get("negative_type") == "oov_db" and r.get("over_correction"))
            s["preserved"] = sum(1 for r in rs if r.get("negative_type") == "oov_db" and r.get("oov_preserved"))
            if any(r.get("baseline_e2e") for r in rs):
                pair = Counter()
                for r in rs:
                    if not r.get("baseline_e2e"):
                        continue
                    b, a = r["baseline_e2e"]["exact"], r["e2e"]["exact"]
                    pair["保持" if b and a else "纠对" if (not b and a) else "改坏" if (b and not a) else "漏纠"] += 1
                s["pair"] = dict(pair)
            out["slices"][f"{ax}={name}"] = s
    return out


def _grid_point(rs: list[dict], mm: float, sm: float) -> dict:
    tp = fa = fr = tr = 0
    for r in rs:
        dec = replay_decision(r["replay"], mm, sm)
        ok = r["e2e"]["deliverable"]
        if dec == "confident":
            tp += ok
            fa += (not ok)
        else:
            fr += ok
            tr += (not ok)
    n = tp + fa + fr + tr
    return {"margin_min": mm, "sim_min": sm, "TP": tp, "FA": fa, "FR": fr, "TR": tr,
            "coverage": round((tp + fa) / n, 4) if n else None,
            "risk": round(fa / (tp + fa), 4) if (tp + fa) else None}


def grid_scan(rows: list[dict]) -> dict:
    """对 MARGIN_MIN × SIM_MIN 网格重放闸门，每点算 (coverage, risk)。"""
    rs = [r for r in rows if r.get("scored") and r.get("replay") and r.get("e2e") and r.get("has_address", True)]
    pts = [_grid_point(rs, mm, sm) for mm in GRID_MARGIN for sm in GRID_SIM]
    cur = next((p for p in pts if abs(p["margin_min"] - rank_mod.MARGIN_MIN) < 1e-9
                and abs(p["sim_min"] - rank_mod.SIM_MIN) < 1e-9), None)
    if cur is None:      # 当前阈值不在网格上（临时改过阈值），单独算一个点
        cur = _grid_point(rs, rank_mod.MARGIN_MIN, rank_mod.SIM_MIN)
    # 单调性：coverage 随 margin_min 增大不增、随 sim_min 增大不增
    viol = []
    by = {(p["margin_min"], p["sim_min"]): p for p in pts}
    for (mm, sm), p in by.items():
        nxt = by.get((round(mm + 0.02, 2), sm))
        if nxt and p["coverage"] is not None and nxt["coverage"] is not None and nxt["coverage"] > p["coverage"] + 1e-9:
            viol.append(f"margin {mm}→{round(mm+0.02,2)} @sim {sm}: {p['coverage']}→{nxt['coverage']}")
        nxt = by.get((mm, round(sm + 0.02, 2)))
        if nxt and p["coverage"] is not None and nxt["coverage"] is not None and nxt["coverage"] > p["coverage"] + 1e-9:
            viol.append(f"sim {sm}→{round(sm+0.02,2)} @margin {mm}: {p['coverage']}→{nxt['coverage']}")
    return {"n": len(rs), "points": pts, "current": cur, "monotonic_violations": viol}


def calibration(rows: list[dict]) -> dict:
    rs = [r for r in rows if r.get("scored") and r.get("replay") and not r["replay"].get("empty") and r.get("e2e")]
    if not rs:
        return {"n": 0}
    return ece([min(1.0, max(0.0, r["replay"]["top_total"])) for r in rs], [r["e2e"]["deliverable"] for r in rs])


def paired_stats(rows: list[dict]) -> dict | None:
    rs = [r for r in rows if r.get("baseline_e2e") and r.get("e2e")]
    if not rs:
        return None
    a = [r["baseline_e2e"]["exact"] for r in rs]
    b = [r["e2e"]["exact"] for r in rs]
    grid = Counter()
    for x, y in zip(a, b):
        grid["保持" if x and y else "纠对" if (not x and y) else "改坏" if (x and not y) else "漏纠"] += 1
    return {"n": len(rs), "grid": dict(grid), "net": grid["纠对"] - grid["改坏"],
            "bootstrap": bootstrap_diff(a, b), "mcnemar": mcnemar(a, b),
            "baseline_exact": sum(a), "audio_exact": sum(b)}


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def _fmt(v, kind: str) -> str:
    if kind == "rate":
        return fmt_rate(*v) if v else "-"
    if kind == "mean":
        return f"{v[0]:.3f} (n={v[1]})" if v and v[0] is not None else "-"
    return str(v)


def _gap_table(rows: list[dict]) -> list[str]:
    cells = Counter((r.get("dialect_group") or "?", r.get("address_depth") or "?", r.get("noise") or "?") for r in rows)
    lines = ["| dialect_group | address_depth | noise | n | 距 20 还差 |", "|---|---|---|---|---|"]
    for (g, d, nz), n in sorted(cells.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {g} | {d} | {nz} | {n} | {max(0, MIN_N_FOR_PCT - n)} |")
    return lines


def _slice_table(summary: dict, cols: list[tuple[str, str, str]], axes: tuple[str, ...] = ("全部",) + AXES) -> list[str]:
    head = "| 分片 | n | " + " | ".join(c[2] for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 2)]
    for key, s in summary["slices"].items():
        ax, name = key.split("=", 1)
        if ax not in axes:
            continue
        label = name if ax == "全部" else f"{ax}={name}"
        lines.append(f"| {label} | {s['n']} | " + " | ".join(_fmt(s.get(c[0]), c[1]) for c in cols) + " |")
    return lines


def build_report(run: dict) -> str:
    rows, summary = run["rows"], run["summary"]
    mode, meta = run["mode"], run["meta"]
    scored = [r for r in rows if r.get("scored") and not r.get("skipped")]
    n = len(scored)
    L: list[str] = [
        "---", "type: report", f"title: 方言地址识别评测 v2 · {mode}{(' · ' + run['tag']) if run.get('tag') else ''}",
        f"description: {meta.get('name', '')}，{n} 条计分样本，模式 {mode}，负样本 {run['negatives']}，oracle {run['oracle'] or '无'}",
        "tags: [ASR, 方言, 评测, 分阶段]", f"timestamp: {time.strftime('%Y-%m-%d')}", "---", "",
        f"# 方言地址识别评测 v2（{mode}{' · ' + run['tag'] if run.get('tag') else ''}）", "",
        f"- 评测集: {meta.get('name', '')}  计分 {n} 条，未标注/跳过 {len(rows) - n} 条  耗时 {run['elapsed']:.0f}s",
        f"- 负样本: {run['negatives']}  oracle: {run['oracle'] or '无'}  阈值: MARGIN_MIN={rank_mod.MARGIN_MIN} SIM_MIN={rank_mod.SIM_MIN}",
    ]
    if run.get("db_note"):
        L.append(f"- 地址库改动: {run['db_note']}")
    if run.get("label_note"):
        L.append(f"- ⚠ {run['label_note']}")
    if n == 0:
        L.append("\n没有可计分的样本。")
        return "\n".join(L)

    # 1. 总览
    L += ["", "## 1. 总览", "", "| 轴 | 取值分布 |", "|---|---|"]
    for ax in AXES:
        c = Counter()
        for r in scored:
            if ax == "difficulty":
                c.update(r.get("difficulty") or [])
            elif ax == "short_name":
                c["短名"] += 1 if r.get("short_name") else 0
            else:
                c[str(r.get(ax))] += 1
        L.append(f"| {ax} | " + ", ".join(f"{k}={v}" for k, v in c.most_common()) + " |")
    L += ["", f"样本缺口（dialect_group × address_depth × noise，每格目标 ≥ {MIN_N_FOR_PCT}）：", ""] + _gap_table(scored)
    if any(k in (r.get("stage_e") or {}) for r in scored for k in ("quadrant",)):
        fa_n = sum(1 for r in scored if r.get("stage_e", {}).get("quadrant") in ("TP", "FA"))
        rot = rule_of_three(fa_n)
        L += ["", f"自动通过 {fa_n} 条；若其中 0 错送，错送率 95% 上限（rule of three）≈ {rot:.1%}" if rot else ""]

    # 2. 阶段指标
    L += ["", "## 2. 阶段指标", "", f"百分比带 95% Wilson 区间；n < {MIN_N_FOR_PCT} 只显示 k/n。"]
    titles = {"A": "Stage A 裸 ASR（仅音频模式；参照 transcript_gold）", "B": "Stage B 归一化与抽取",
              "C": "Stage C 召回（真值链在库的子集）", "D": "Stage D 排序（recall_chain@all 的子集）",
              "E": "Stage E 闸门与决策", "F": "Stage F 补全（address_depth ∈ {district, street_only}）"}
    for st, metrics in STAGE_METRICS.items():
        if st == "A" and mode == "text":
            continue
        cols = [(f"{st}.{k}", "rate" if kind in ("rate", "special") else "mean", k) for k, kind in metrics]
        L += ["", f"### {titles[st]}", ""] + _slice_table(summary, cols)
        if st == "C":
            reasons = Counter(r["stage_c"].get("miss_reason") for r in scored if r.get("stage_c", {}).get("miss_reason"))
            if reasons:
                L += ["", "miss_reason 分布: " + ", ".join(f"{k}={v}" for k, v in reasons.items())]
        if st == "D":
            doms = Counter(r["stage_d"].get("dominant_term") for r in scored if r.get("stage_d", {}).get("dominant_term"))
            if doms:
                L += ["", "dominant_term 分布: " + ", ".join(f"{k}={v}" for k, v in doms.items())]
        if st == "E":
            gates = Counter(r["stage_e"].get("gate_attrib") for r in scored if r.get("stage_e", {}).get("gate_attrib"))
            if gates:
                L += ["", "gate_attrib 分布: " + ", ".join(f"{k}={v}" for k, v in gates.items())]
    oov = [r for r in scored if r.get("negative_type") == "oov_db"]
    if oov:
        s = summary["slices"].get("negative_type=oov_db", {})
        L += ["", f"**oov_db 负样本** n={len(oov)}：过度纠正（confident 且地名被改）= {s.get('over_correction')}，"
              f"原文保住 = {s.get('preserved')}，决策分布 {s.get('decisions')}"]

    # 3. 决策四格与 risk-coverage
    em = e_metrics(scored)
    L += ["", "## 3. 决策四格与 risk-coverage", "",
          "| | Top-1 正确（deliverable） | Top-1 错误 |", "|---|---|---|",
          f"| 自动通过 confident | TP={em['TP']} | **FA={em['FA']}** |",
          f"| 拦截 ambiguous/partial/reject | FR={em['FR']} | TR={em['TR']} |", "",
          f"- coverage = {fmt_rate(*em['coverage'])}   risk = {fmt_rate(*em['risk'])}   "
          f"over_reject = {fmt_rate(*em['over_reject'])}   nbest_recall@3(被拦截) = {fmt_rate(*em['nbest_recall@3'])}"]
    g = run.get("grid")
    if g and g.get("points"):
        L += ["", f"risk-coverage 网格（重放闸门，n={g['n']}；格内 coverage/risk，行 MARGIN_MIN，列 SIM_MIN）：", ""]
        sims = [s for s in GRID_SIM if round(s * 100) % 5 == 0]
        L.append("| MARGIN \\ SIM | " + " | ".join(f"{s:.2f}" for s in sims) + " |")
        L.append("|---|" + "---|" * len(sims))
        by = {(p["margin_min"], p["sim_min"]): p for p in g["points"]}
        for mm in GRID_MARGIN:
            cells = []
            for sm in sims:
                p = by[(mm, sm)]
                cov = f"{p['coverage']:.2f}" if p["coverage"] is not None else "-"
                rk = f"{p['risk']:.2f}" if p["risk"] is not None else "-"
                cells.append(f"{cov}/{rk}")
            L.append(f"| {mm:.2f} | " + " | ".join(cells) + " |")
        cur = g.get("current")
        if cur:
            L.append(f"\n当前阈值 ({rank_mod.MARGIN_MIN}, {rank_mod.SIM_MIN}) 所在点: coverage={cur['coverage']} risk={cur['risk']} "
                     f"(TP={cur['TP']} FA={cur['FA']} FR={cur['FR']} TR={cur['TR']})")
        L.append("单调性: " + ("通过" if not g["monotonic_violations"] else "违反 " + "; ".join(g["monotonic_violations"][:5])))
    cal = run.get("calibration") or {}
    if cal.get("n"):
        L += ["", f"校准（Chain.total 十桶，ECE={cal['ece']}，{cal['nonempty_buckets']}/10 桶有数据；calib 集为空，未做保序回归）：", "",
              "| 桶 | n | 平均分 | 实际正确率 |", "|---|---|---|---|"]
        for b in cal["buckets"]:
            if b["n"]:
                L.append(f"| [{b['lo']:.1f}, {b['hi']:.1f}) | {b['n']} | {b['conf']} | {b['acc']} |")

    # 4. 纠正四格
    L += ["", "## 4. 纠正四格（baseline 对 audio 配对）", ""]
    ps = run.get("paired")
    if ps:
        gd = ps["grid"]
        L += ["| | baseline 对 | baseline 错 |", "|---|---|---|",
              f"| audio 对 | 保持 {gd.get('保持', 0)} | **纠对 {gd.get('纠对', 0)}** |",
              f"| audio 错 | **改坏 {gd.get('改坏', 0)}** | 漏纠 {gd.get('漏纠', 0)} |", "",
              f"- 净收益 = {ps['net']} / {ps['n']}；bootstrap 95% 区间 [{ps['bootstrap']['lo']:+.3f}, {ps['bootstrap']['hi']:+.3f}]"
              f"（点估计 {ps['bootstrap']['diff']:+.3f}）；McNemar p={ps['mcnemar']['p']}"
              + ("（n < 20，p 值仅供参考）" if ps["n"] < MIN_N_FOR_PCT else ""),
              f"- baseline EM {ps['baseline_exact']}/{ps['n']} → audio EM {ps['audio_exact']}/{ps['n']}"]
        asr_ok = [r for r in scored if r.get("baseline_e2e") and r["baseline_e2e"]["geo_recall"] == 1.0]
        broke = [r for r in asr_ok if r["e2e"]["geo_recall"] < 1.0]
        L.append(f"- asr_correct 子集（裸 ASR 地名已对）n={len(asr_ok)}，其中被后处理改坏 {len(broke)}"
                 + (": " + ", ".join(r["id"] for r in broke) if broke else ""))
    else:
        L.append("仅 audio 模式有。")

    # 5. 归因
    L += ["", "## 5. 归因分布", ""]
    att = Counter(r["attribution"] for r in scored if r.get("attribution"))
    L += ["| 标签 | 数量 |", "|---|---|"] + [f"| {k} | {att.get(k, 0)} |" for k in LABELS if att.get(k)]
    if not att:
        L.append("| （无错例） | 0 |")
    L += ["", "按分片："] + _slice_table(summary, [("attribution", "raw", "归因")],
                                         axes=("dialect_group", "address_depth", "noise", "negative_type", "short_name", "rule"))
    errs = [r for r in scored if r.get("attribution")]
    if errs:
        L += ["", "| # | ASR/输入 | 真值 | 输出 | 决策 | 归因 | 关键中间量 |", "|---|---|---|---|---|---|---|"]
        for r in errs[:80]:
            a = r.get("stage_a") or {}
            mid = r.get("attribution_detail", "")
            if a.get("name_phon_dist_max") is not None:
                mid = f"音距离 {a['name_phon_dist_max']}; " + mid
            L.append(f"| {r['id']} | {r['raw'][:40]} | {r['truth'][:40]} | {r['pred'][:40]} | {r['decision']} | {r['attribution']} | {mid[:110]} |")
        if len(errs) > 80:
            L.append(f"\n（只列前 80 条，全部见 json）")

    # 6. Oracle
    L += ["", "## 6. Oracle 表", ""]
    ot = run.get("oracle_table")
    if ot:
        L += ["| 配置 | EM | deliverable | coverage | risk | 归因分布 |", "|---|---|---|---|---|---|"]
        for row in ot["rows"]:
            L.append(f"| {row['config']} | {fmt_rate(*row['exact'])} | {fmt_rate(*row['deliverable'])} | "
                     f"{fmt_rate(*row['coverage'])} | {fmt_rate(*row['risk'])} | {row['attribution']} |")
        L.append("\n单调性检查（任一 oracle 任一分片不得低于 current）: " +
                 ("通过" if not ot["violations"] else "违反 " + "; ".join(ot["violations"][:8])))
        if ot.get("residual"):
            L.append(f"\n三个 oracle 叠加后剩余错例（后处理够不着的部分）: " + "; ".join(ot["residual"][:10]))
    else:
        L.append("未启用（--oracle all）。")

    # 7. 与上次对比
    L += ["", "## 7. 与上次对比", ""]
    cmp_ = run.get("compare")
    if cmp_:
        L.append(f"对比基线: {cmp_['path']}")
        if cmp_["changes"]:
            L += ["", "| 分片 | 指标 | 上次 | 本次 |", "|---|---|---|---|"]
            L += [f"| {c['slice']} | {c['metric']} | {c['prev']} | {c['now']} |" for c in cmp_["changes"][:60]]
        else:
            L.append("逐分片无 ≥ 1 个样本的变化。")
    else:
        L.append("没有可对比的基线（eval/golden/ 为空且未指定 --compare）。")

    # 附：逐条
    L += ["", "## 附. 逐条", "", "| # | 结果 | 决策 | 输出 | 真值 | 归因 |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("skipped"):
            continue
        if not r.get("scored"):
            L.append(f"| {r['id']} | ? | {r['decision']} | {r['pred']} | （未标注） | |")
            continue
        s = r["e2e"]
        mark = "✓" if s["exact"] else ("~" if s.get("admin_all") else "✗")
        L.append(f"| {r['id']} | {mark} | {r['decision']} | {r['pred']} | {r['truth']} | {r.get('attribution') or ''} |")
    return "\n".join(L)


def compare_with(summary: dict, prev_path: Path) -> dict:
    prev = json.loads(prev_path.read_text(encoding="utf-8"))
    ps = prev.get("summary", prev).get("slices", {})
    changes = []
    for key, s in summary["slices"].items():
        p = ps.get(key)
        if not p:
            continue
        for metric in ("exact", "deliverable", "E.coverage", "E.risk", "C.recall_chain@all", "D.top1_correct"):
            a, b = p.get(metric), s.get(metric)
            if not a or not b:
                continue
            if abs(a[0] - b[0]) >= 1 or a[1] != b[1]:
                changes.append({"slice": key, "metric": metric, "prev": f"{a[0]}/{a[1]}", "now": f"{b[0]}/{b[1]}"})
        for metric in ("over_correction",):
            if p.get(metric) is not None and s.get(metric) != p.get(metric):
                changes.append({"slice": key, "metric": metric, "prev": p[metric], "now": s[metric]})
    return {"path": str(prev_path), "changes": changes}


# --------------------------------------------------------------------------
# 注入负样本
# --------------------------------------------------------------------------


def run_injection(asr, cases_path: Path) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    out = []
    new_flips, known_flips, fixed = 0, 0, 0
    for c in cases:
        ap = AUDIO_ROOT / c["audio"]
        o = asr.transcribe(str(ap), language=None, context=c.get("inject"))
        ok = c["expect_key"] in o.text
        flipped = not ok
        if flipped and c.get("known_flip"):
            known_flips += 1
        elif flipped:
            new_flips += 1
        elif c.get("known_flip"):
            fixed += 1
        out.append({**c, "output": o.text, "ok": ok, "flipped": flipped})
        print(f"{'✓' if ok else '✗'} {c.get('name', '')} inject={c.get('inject')} → {o.text}")
    return {"cases": out, "new_flips": new_flips, "known_flips": known_flips, "fixed_known_flips": fixed,
            "n": len(cases)}


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def run(a: argparse.Namespace) -> dict:
    t0 = time.time()
    mode = a.mode
    oracle = set() if not a.oracle or a.oracle == "none" else (
        {"db", "rank", "gate"} if a.oracle == "all" else set(a.oracle.split(","))
    )
    base_db = AddressDB.default()
    db_note = ""
    label_note = ""

    # ---- 样本 ----
    if mode == "text":
        meta, items = load_items(Path(a.eval))
        for it in items:
            it.setdefault("transcript_gold", it["spoken"])
            it.setdefault("scored", True)
    else:
        meta = {"name": f"manifest {Path(a.manifest).name}"}
        items = manifest_to_items(load_manifest(Path(a.manifest)))
        drafts = sum(1 for it in items if it.get("label_status") == "draft")
        if drafts:
            label_note = (f"{drafts} 条录音的真值为**草稿**（由 ASR 输出 + 文档记载预填，未经说话人确认），"
                          "Stage A / 纠正四格的数字不可对外。确认后把 manifest 的 label_status 改成 confirmed。")
    if a.limit:
        items = items[: a.limit]

    # ---- 负样本：库缺 ----
    negatives = a.negatives or "none"
    db = base_db
    if negatives in ("oov", "all") and mode == "text":
        spec = json.loads((NEG_DIR / "oov_drop.json").read_text(encoding="utf-8"))
        db, removed = db_without(base_db, spec["drop"])
        affected = {i for d in spec["drop"] for i in d.get("affects", [])}
        for it in items:
            if it["id"] in affected or it.get("source_id") in affected:
                it["negative_type"] = "oov_db"
        db_note = f"oov_drop 删除 {len(removed)} 条: {', '.join(removed)}"

    # ---- ASR ----
    asr = None
    if mode in ("audio", "baseline") or negatives in ("injection", "all"):
        from asr_cache import CachedASR
        if a.asr_cache_readonly:
            asr = CachedASR(None, readonly=True)
        else:
            from dialect_addr.asr import Qwen3ASR
            asr = CachedASR(Qwen3ASR())

    # ---- oracle db ----
    if "db" in oracle:
        db, added = db_with_gold(db, items)
        db_note += (" | " if db_note else "") + f"oracle db 插入 {len(added)} 条: {', '.join(added)}"

    rows = run_once(mode, items, db, asr, a.dialect, not a.no_two_pass, oracle)
    summary = summarize(rows)
    run_obj: dict = {
        "mode": mode, "tag": a.tag, "meta": meta, "negatives": negatives, "oracle": ",".join(sorted(oracle)),
        "db_note": db_note, "label_note": label_note, "rows": rows, "summary": summary,
        "thresholds": {"MARGIN_MIN": rank_mod.MARGIN_MIN, "SIM_MIN": rank_mod.SIM_MIN},
        "grid": grid_scan(rows) if not a.no_grid else None,
        "calibration": calibration(rows),
        "paired": paired_stats(rows) if mode == "audio" else None,
    }

    # ---- oracle 表 ----
    if a.oracle == "all" and mode == "text":
        run_obj["oracle_table"] = oracle_table(items, base_db if negatives == "none" else db, a, rows)

    # ---- 注入负样本 ----
    if negatives in ("injection", "all") and asr is not None:
        run_obj["injection"] = run_injection(asr, NEG_DIR / "injection.json")

    # ---- 与上次对比 ----
    cmp_path = None
    if a.compare:
        cmp_path = Path(a.compare)
    else:
        cand = GOLDEN_DIR / f"{golden_name(a)}.json"
        if cand.exists():
            cmp_path = cand
    if cmp_path and cmp_path.exists():
        run_obj["compare"] = compare_with(summary, cmp_path)

    run_obj["elapsed"] = time.time() - t0
    return run_obj


def golden_name(a: argparse.Namespace) -> str:
    parts = [a.mode]
    if a.tag:
        parts.append(a.tag)
    if a.negatives and a.negatives != "none":
        parts.append(a.negatives)
    return "_".join(parts)


def _table_row(config: str, rows: list[dict]) -> dict:
    sc = [r for r in rows if r.get("scored") and not r.get("skipped")]
    em = e_metrics(sc)
    return {"config": config, "exact": _rate(sc, "exact", "e2e"), "deliverable": _rate(sc, "deliverable", "e2e"),
            "coverage": em["coverage"], "risk": em["risk"],
            "attribution": dict(Counter(r["attribution"] for r in sc if r.get("attribution")))}


def oracle_table(items: list[dict], db: AddressDB, a: argparse.Namespace, current_rows: list[dict]) -> dict:
    """§6：baseline / current / db / rank / gate / db+rank+gate 六行。"""
    print("\n== oracle 表 ==")
    configs = [("baseline（输入 + 归一化，无检索）", "baseline"), ("current", set()),
               ("oracle db", {"db"}), ("oracle rank", {"rank"}), ("oracle gate", {"gate"}),
               ("oracle db+rank+gate", {"db", "rank", "gate"})]
    table_rows, per_config_summary, residual = [], {}, []
    for name, cfg in configs:
        if cfg == "baseline":
            rows = []
            for it in items:
                norm, tail = num_normalize(it["spoken"])
                rec = {**{k: it.get(k) for k in ("id", "difficulty", "dialect_group", "address_depth", "noise", "negative_type")},
                       "truth": it["ground_truth"], "scored": True, "raw": it["spoken"], "pred": norm,
                       "fields": tail.as_dict(), "decision": "baseline", "has_address": it.get("has_address", True)}
                evaluate(rec, None, it, db, a.dialect, "text", False, it["spoken"])
                rows.append(rec)
        elif cfg == set():
            rows = current_rows
        else:
            d = db
            if "db" in cfg:
                d, _ = db_with_gold(db, items)
            rows = run_once("text", copy.deepcopy(items), d, None, a.dialect, False, cfg, verbose=False)
        table_rows.append(_table_row(name, rows))
        per_config_summary[name] = summarize(rows)
        if cfg == {"db", "rank", "gate"}:
            residual = [f"{r['id']}:{r['attribution']}({r.get('attribution_detail', '')[:60]})"
                        for r in rows if r.get("attribution")]
        print(f"  {name:<28} EM {table_rows[-1]['exact']}  cov {table_rows[-1]['coverage']}  risk {table_rows[-1]['risk']}")
    # 单调性：每个 oracle 在每个分片上的 exact / deliverable 不低于 current
    cur = per_config_summary["current"]["slices"]
    viol = []
    for name, s in per_config_summary.items():
        if not name.startswith("oracle"):
            continue
        for key, sv in s["slices"].items():
            cv = cur.get(key)
            if not cv:
                continue
            for m in ("exact", "deliverable"):
                if sv[m][0] < cv[m][0]:
                    viol.append(f"{name} {key} {m} {sv[m][0]}<{cv[m][0]}")
    return {"rows": table_rows, "violations": viol, "residual": residual}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "audio", "baseline"], default="text")
    ap.add_argument("--eval", default=str(ROOT / "data" / "eval" / "xinan_guanhua.json"))
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--negatives", choices=["none", "oov", "injection", "all"], default="none")
    ap.add_argument("--oracle", default="none", help="none | db | rank | gate | db,rank | all（all 时输出 §6 六行表）")
    ap.add_argument("--dialect", default=None, help="方言提示，如 Sichuan；不给则由 ASR 自动识别")
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    ap.add_argument("--tag", default="", help="报告/golden 名后缀，如 synthetic")
    ap.add_argument("--compare", default=None, help="上一份报告 json；缺省读 eval/golden/<mode>_<tag>.json")
    ap.add_argument("--no-two-pass", action="store_true")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--asr-cache-readonly", action="store_true", help="只用 ASR 缓存，不加载模型（CI）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="临时覆盖 rank 模块常量做故意破坏测试，如 --set SIM_MIN=0.95 --set W_CONFLICT=1.0")
    a = ap.parse_args()
    if a.model_dir:
        os.environ["DIALECT_ADDR_MODEL_DIR"] = a.model_dir
    for kv in a.set:
        k, v = kv.split("=", 1)
        if not hasattr(rank_mod, k):
            sys.exit(f"--set: rank 模块没有常量 {k}")
        setattr(rank_mod, k, float(v))
        print(f"[override] rank.{k} = {v}")

    run_obj = run(a)
    report = build_report(run_obj)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = golden_name(a)
    md = out_dir / f"{stamp}_{name}.md"
    md.write_text(report, encoding="utf-8")
    js = out_dir / f"{stamp}_{name}.json"
    dump = {k: v for k, v in run_obj.items()}
    js.write_text(json.dumps(dump, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if not a.quiet:
        print("\n" + report)
    print(f"\n报告: {md}\nJSON: {js}")


if __name__ == "__main__":
    main()
