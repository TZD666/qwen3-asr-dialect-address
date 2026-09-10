#!/usr/bin/env python3
"""评测自检：故意把系统弄坏，看评测叫不叫、叫的名字对不对（评测体系设计 §10 步骤 2 / 步骤 6）。

"24/24、720/720、归因表为空"这句话，只有在"坏了它会叫"的前提下才有信息量。
这里对每个归因标签构造一种破坏：改 rank 参数、删地址库条目、改真值，
然后断言归因树贴出预期标签。叫得出来的写进"能抓住"；叫不出来、叫错名字的写进"抓不住"，
并写明缺什么数据才能补上。

报告五节：
    a. 故意破坏场景表（每个归因标签至少一个场景，外加"回归变红"）
    b. 样本缺口与标注状态（有效样本按句计，§2.6 门槛）
    c. 合成扰动集的规则覆盖（哪些音变规则不足 20 条、哪些一条都没有）
    d. 真实数据上出现过哪些标签（最近一份 audio / 合成 / text 报告）
    e. 参数可辨识性（eval/tuning/ 最近一份调参提案；没有就写"尚无提案"）
最后一张"能抓住什么、抓不住什么"。

只跑 text 模式、进程内、不加载模型；每个场景结束（含异常）都把 rank 参数恢复原样。

    python eval/selfcheck.py          # 全量：合成集两个权重场景各跑 720 条，约 5 分钟
    python eval/selfcheck.py --fast   # 合成集只跑已知敏感的 source_id 组，约 1 分钟
    python eval/selfcheck.py --ci     # 只打印失败项和一行汇总（可与 --fast 同用）

声明 reachable=True 的场景没有贴出预期标签 → 退出码 1。
输出 eval/selfcheck/latest.md 与 latest.json，每次覆盖。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr import rank as rank_mod  # noqa: E402
from dialect_addr.address_db import AddressDB  # noqa: E402

import regression  # noqa: E402
from attribution import LABELS, attribute  # noqa: E402
from run_eval import (  # noqa: E402
    MANIFEST, _gap_table, db_without, load_items, load_manifest, manifest_to_items, run_once, summarize,
)
from splits import effective_counts, gate_status  # noqa: E402
from stats import MIN_N_FOR_PCT  # noqa: E402

READ_ITEMS = ROOT / "data" / "eval" / "xinan_guanhua.json"
GEN_CONFIG = regression.SYNTH.with_name("gen_config.json")
REPORTS = ROOT / "eval" / "reports"
TUNING = ROOT / "eval" / "tuning"
OUT_DIR = ROOT / "eval" / "selfcheck"
SECTIONS = ("scenarios", "samples", "rule_coverage", "labels_observed", "identifiability")

# 破坏用的值都是实测挑出来的（2026-09-10），不是拍的。换值之前先跑一遍，看标签还在不在。
ASR_JUNK = "泼皮鸭"          # 顶替 003 的「民权路」：最近片段音距离 0.648 > MAX_DIST
RECALL_MAX_DIST = 0.01       # 0.05 时 60 条里 0 条 recall_miss：合成扰动的加权音距离摊到 4–6 个音节后很小
RECALL_PROBE_N = 60
RANK_BREAK = {"W_SIM": 0.0, "W_PRIOR": 1.0}
# 三道阈值放到约等于 0。恰好取 0 时 stages.stage_e 算 gate_attrib 会除零（见 sc_gate_over_accept）
GATE_OPEN = {"SINGLE_HIT_MIN_COV": 0.0, "SIM_MIN": 1e-6, "MARGIN_MIN": 1e-6}
GATE_ZERO = {"SINGLE_HIT_MIN_COV": 0.0, "SIM_MIN": 0.0, "MARGIN_MIN": 0.0}
COMPLETION_ITEM = "018"      # 只报到区；删掉它的路级条目后，闸门拦下、省市没回溯
CONFLICT_BREAK = {"W_CONFLICT": 1.0}
# --fast 时合成集场景只跑这些 source_id 组。全量 720 条实测：W_SIM=0/W_PRIOR=1 的 rank_error
# 全部落在 010/011；W_CONFLICT=1.0 唯一翻转的样本是 014-017。前 240 行一条 rank_error 都没有，
# 拿它当快速子集只会让场景假失败。代码一变敏感样本会挪窝，出现"--fast 红、全量绿"时回来改这两行。
FAST_RANK_SOURCES = ("010", "011")
FAST_CONFLICT_SOURCES = ("014",)


# --------------------------------------------------------------------------
# 数据与运行
# --------------------------------------------------------------------------


def _prep(items: list[dict]) -> list[dict]:
    for it in items:
        it.setdefault("transcript_gold", it["spoken"])
        it.setdefault("scored", True)
    return items


class Ctx:
    """场景共用的数据，懒加载一次；每次取出的都是深拷贝，场景之间互不污染。

    只改参数的场景共用同一个 AddressDB 对象（拼音碰撞表缓存在库对象上，与参数无关）；
    删条目的场景由 db_without 新建库对象，缓存自然分开。
    """

    def __init__(self, fast: bool):
        self.fast = fast
        self._cache: dict = {}

    def _once(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def db(self) -> AddressDB:
        return self._once("db", AddressDB.default)

    def read(self) -> list[dict]:
        return copy.deepcopy(self._once("read", lambda: _prep(load_items(READ_ITEMS)[1])))

    def synth(self) -> list[dict]:
        return copy.deepcopy(self._once("synth", lambda: _prep(load_items(regression.SYNTH)[1])))

    def recordings(self) -> list[dict]:
        return copy.deepcopy(self._once("recordings", lambda: manifest_to_items(load_manifest(MANIFEST))))


def run_text(items: list[dict], db: AddressDB, params: dict | None = None) -> list[dict]:
    """text 模式跑一遍。params 只在这次调用里生效，结束（含异常）一律恢复。"""
    saved = rank_mod.current_params()
    try:
        if params:
            rank_mod.apply_params(params)
        return run_once("text", items, db, None, None, False, set(), verbose=False)
    finally:
        rank_mod.apply_params(saved)


def label_counts(rows: list[dict]) -> dict[str, int]:
    """计分样本的归因分布；端到端正确且自动通过（无标签）记为 none。"""
    return dict(Counter(r.get("attribution") or "none" for r in rows if r.get("scored")))


def _item(items: list[dict], item_id: str) -> dict:
    return next(it for it in items if it["id"] == item_id)


# --------------------------------------------------------------------------
# a. 故意破坏场景
# --------------------------------------------------------------------------


def sc_db_missing(ctx: Ctx) -> dict:
    db, removed = db_without(ctx.db(), [{"name": "解放碑街道"}, {"name": "犀浦街道"}])
    rows = run_text(ctx.read(), db)
    labelled = {r["id"]: r["attribution"] for r in rows if r["attribution"]}
    rest = [r for r in rows if r["id"] not in labelled]
    rest_exact = sum(1 for r in rest if r["e2e"]["exact"])
    return {"rows": rows,
            "passed": labelled == {"003": "db_missing", "005": "db_missing"} and rest_exact == len(rest),
            "evidence": f"删 {'、'.join(removed)} → {labelled}；其余 {rest_exact}/{len(rest)} 条 exact"}


def sc_asr_unrecoverable(ctx: Ctx) -> dict:
    it = _item(ctx.read(), "003")
    it["transcript_gold"] = it["spoken"]
    it["spoken"] = it["spoken"].replace("民权路", ASR_JUNK)
    it["stage_a_force"] = True          # text 模式默认不算 Stage A；强制算，才有音距离可比
    r = run_text([it], ctx.db())[0]
    a = r.get("stage_a") or {}
    return {"rows": [r], "passed": r["attribution"] == "asr_unrecoverable",
            "evidence": f"输入「{it['spoken']}」name_phon_dist_max={a.get('name_phon_dist_max')} → {r['attribution_detail']}"}


def sc_normalize_error(ctx: Ctx) -> dict:
    it = _item(ctx.read(), "003")
    it["fields"]["house_no"] = "19号"
    it["ground_truth"] = it["ground_truth"].replace("18号", "19号")
    r = run_text([it], ctx.db())[0]
    return {"rows": [r], "passed": r["attribution"] == "normalize_error",
            "evidence": f"输出「{r['pred']}」vs 真值「{it['ground_truth']}」→ {r['attribution_detail']}"}


def sc_recall_miss(ctx: Ctx) -> dict:
    sub = [it for it in ctx.synth() if it.get("rule") != "同音异形"][:RECALL_PROBE_N]
    rows = run_text(sub, ctx.db(), {"MAX_DIST": RECALL_MAX_DIST})
    lc = label_counts(rows)
    reasons = dict(Counter(r["stage_c"]["miss_reason"] for r in rows if (r.get("stage_c") or {}).get("miss_reason")))
    n_norm = lc.get("normalize_error", 0)
    finding = None
    if n_norm:
        ex = next(r for r in rows if r.get("attribution") == "normalize_error")
        finding = (f"召回只看真值链最深层条目：MAX_DIST={RECALL_MAX_DIST} 时同批 {n_norm} 条是市/区/街道层被扰动、没召回，"
                   f"扰动字原样留在输出里，被 Stage B 记成闲话泄漏 → normalize_error，而不是 recall_miss"
                   f"（例 {ex['id']}「{ex['raw']}」：{ex['attribution_detail']}）")
    return {"rows": rows, "passed": lc.get("recall_miss", 0) >= 1,
            "evidence": f"recall_miss×{lc.get('recall_miss', 0)}，miss_reason {reasons}；同批 normalize_error×{n_norm}",
            "finding": finding, "extra": {"miss_reason": reasons, "normalize_error": n_norm}}


def sc_rank_error(ctx: Ctx) -> dict:
    syn = ctx.synth()
    sub = [it for it in syn if it.get("source_id") in FAST_RANK_SOURCES] if ctx.fast else syn
    rows = run_text(sub, ctx.db(), RANK_BREAK)
    errs = [r for r in rows if r.get("attribution") == "rank_error"]
    doms = dict(Counter(r["stage_d"].get("dominant_term") for r in errs))
    return {"rows": rows, "passed": len(errs) >= 1,
            "evidence": f"{len(sub)} 条里 rank_error×{len(errs)}，dominant_term {doms}",
            "extra": {"dominant_term": doms, "n": len(errs)}}


def sc_gate_over_accept(ctx: Ctx) -> dict:
    items = [it for it in ctx.recordings() if it.get("negative_type") == "no_address"]
    rows = run_text(items, ctx.db(), GATE_OPEN)
    decisions = {r["id"]: r["decision"] for r in rows}
    fa = [f"{r['id']}「{r['raw']}」→「{r['pred']}」" for r in rows if r.get("attribution") == "gate_over_accept"]
    finding = None
    try:
        run_text(copy.deepcopy(items), ctx.db(), GATE_ZERO)
    except ZeroDivisionError as e:
        finding = (f"MARGIN_MIN 或 SIM_MIN 恰为 0 时，eval/stages.py stage_e 算 gate_attrib 的 slack 除以阈值，"
                   f"错送样本直接抛 ZeroDivisionError（{e}）；本场景改用 1e-6 绕开")
    return {"rows": rows, "passed": bool(fa),
            "evidence": f"{len(rows)} 条无地址录音，决策 {decisions}；错送 {fa or '无'}",
            "finding": finding, "extra": {"decisions": decisions}}


def sc_gate_over_reject_sim(ctx: Ctx) -> dict:
    rows = run_text(ctx.read(), ctx.db(), {"SIM_MIN": 1.01})
    n = label_counts(rows).get("gate_over_reject", 0)
    attrib = dict(Counter((r.get("stage_e") or {}).get("gate_attrib") for r in rows))
    return {"rows": rows, "passed": n == len(rows) and attrib == {"sim": len(rows)},
            "evidence": f"{n}/{len(rows)} 条 gate_over_reject，gate_attrib {attrib}"}


def sc_gate_over_reject_margin(ctx: Ctx) -> dict:
    rows = run_text(ctx.read(), ctx.db(), {"MARGIN_MIN": 1.0})
    n = sum(1 for r in rows if r.get("attribution") == "gate_over_reject"
            and (r.get("stage_e") or {}).get("gate_attrib") == "margin")
    return {"rows": rows, "passed": n >= 10,
            "evidence": (f"gate_over_reject(margin)×{n}/{len(rows)}；其余 {len(rows) - n} 条仍 confident："
                         "只有一条候选或 Top-2 同行政区，分差闸门按 rank.decide 不拦"),
            "extra": {"n": n}}


def sc_assembly_error(ctx: Ctx) -> dict:
    it = _item(ctx.read(), "003")
    it["ground_truth"] += "甲"
    r = run_text([it], ctx.db())[0]
    e = r["e2e"]
    blocked = [f for f in regression.compare("selfcheck", {"summary": summarize([r])}, {"summary": summarize([r])})
               if "assembly_error" in f]
    return {"rows": [r], "passed": r["attribution"] == "assembly_error",
            "evidence": (f"真值末尾加「甲」、字段不动：admin_all={e['admin_all']} geo_recall={e['geo_recall']} "
                         f"tail_all={e['tail_all']} gate_attrib={r['stage_e'].get('gate_attrib')} → {r['attribution']}；"
                         f"回归阻断规则{'同时报红' if blocked else '没有报红'}")}


def sc_completion_error(ctx: Ctx) -> dict:
    """声明不可达的标签也要有活的证据：把最像"补全错"的情形跑一遍，确认评测把它记到真因（库缺）上。

    2026-09-10 自检曾发现：路级条目全缺 + 地名原样保住 + 闸门正确拦下 + 省市没回溯 → 被记成 completion_error，
    真因是库缺。归因树第 2 步随后补了这条路径。真正的补全错（Top-1 链对、回溯出的省市错）按构造不可达：
    stages.chain_matches 要求行政区一致。这里的 passed 表示"没有再误归到 completion_error"。
    """
    it = _item(ctx.read(), COMPLETION_ITEM)
    names = [it["fields"][k] for k in ("street", "road", "community") if it["fields"].get(k)]
    db, removed = db_without(ctx.db(), [{"name": n} for n in names])
    r = run_text([it], db)[0]
    ok = r["attribution"] == "db_missing"
    finding = None if ok else f"库缺 + 拦下的样本被记成 {r['attribution']}，应为 db_missing（归因树第 2 步）"
    return {"rows": [r], "passed": ok, "finding": finding,
            "evidence": (f"删 {'、'.join(removed)} → 决策 {r['decision']}，输出「{r['pred']}」，"
                         f"四格 {r['stage_e'].get('quadrant_geo')} → {r['attribution']}：{r['attribution_detail']}")}


def sc_regression_red(ctx: Ctx) -> dict:
    syn = ctx.synth()
    gpath = regression.GOLDEN / "text_synthetic.json"
    if ctx.fast or not gpath.exists():
        sub = [it for it in syn if it.get("source_id") in FAST_CONFLICT_SOURCES] if ctx.fast else syn
        gold = {"summary": summarize(run_text(copy.deepcopy(sub), ctx.db()))}
        baseline = "同子集未破坏的对照运行"
    else:
        sub = syn
        gold = json.loads(gpath.read_text(encoding="utf-8"))
        baseline = str(gpath.relative_to(ROOT))
    rows = run_text(sub, ctx.db(), CONFLICT_BREAK)
    fails = regression.compare("text_synthetic", {"summary": summarize(rows)}, gold)
    flipped = [f"{r['id']}:{r['attribution']}" for r in rows if r.get("attribution")]
    # 同一破坏在 24 句朗读稿上：text 套件红不红
    trows = run_text(ctx.read(), ctx.db(), CONFLICT_BREAK)
    tgold = json.loads((regression.GOLDEN / "text.json").read_text(encoding="utf-8"))
    tfails = regression.compare("text", {"summary": summarize(trows)}, tgold)
    finding = None
    if fails and not tfails:
        finding = (f"W_CONFLICT=1.0 在 24 句朗读稿上回归全绿（text 套件 0 项失败），只有合成集能让它变红，"
                   f"而且只靠 {len(flipped)} 条样本（{'、'.join(flipped)}）")
    lc = label_counts(rows)
    return {"observed": {"regression_red": len(fails), **lc}, "passed": bool(fails),
            "evidence": (f"合成集 {len(sub)} 条对 {baseline}：{len(fails)} 项失败（翻转 {flipped or '无'}）；"
                         f"同一破坏在 text 套件 {len(tfails)} 项失败"),
            "finding": finding,
            "extra": {"fails": fails[:10], "n_fails": len(fails), "flipped": flipped, "text_fails": tfails,
                      "baseline": baseline}}


def sc_skip(ctx: Ctx) -> dict:
    label, detail = attribute({"quality": "silent", "e2e": {"exact": False}})
    non_ok = [r for r in load_manifest(MANIFEST) if r.get("quality") != "ok"]
    return {"observed": {label or "none": 1}, "passed": label == "skip",
            "evidence": f"构造 quality=silent 的记录 → {label}（{detail}）",
            "finding": (f"manifest_to_items 在评测前就滤掉 quality≠ok 的行（清单里 {len(non_ok)} 行），"
                        "真实报告里 skip 恒为 0；quality 判定本身归 check_manifest 管，这里不验证"),
            "extra": {"non_ok_rows": len(non_ok)}}


# label：期望标签；reachable：是否声明可达（可达却没贴出标签 → 退出码 1）；
# text_only：只用 24 句朗读稿或单条构造样本，秒级，tests/test_selfcheck.py 逐个跑
SCENARIOS: list[dict] = [
    {"name": "db_missing", "label": "db_missing", "reachable": True, "text_only": True, "run": sc_db_missing,
     "setup": "地址库删「解放碑街道」「犀浦街道」", "subset": "24 句朗读稿",
     "expect": "003、005 记 db_missing，其余 22 条 exact",
     "note": "删的是口语说法与正名不同的条目（说的是解放碑/犀浦镇），流水线没有库条目就没法规范成正名；同 tests/test_attribution.py"},
    {"name": "asr_unrecoverable", "label": "asr_unrecoverable", "reachable": True, "text_only": True,
     "run": sc_asr_unrecoverable,
     "setup": f"003 的输入把「民权路」换成音上毫不相干的「{ASR_JUNK}」，transcript_gold 保持原文，强制算 Stage A",
     "subset": "朗读稿 003", "expect": "asr_unrecoverable",
     "note": "Stage A 只在 audio 模式或 stage_a_force 时计算；最近窗口音距离 0.648 远超 MAX_DIST"},
    {"name": "normalize_error", "label": "normalize_error", "reachable": True, "text_only": True,
     "run": sc_normalize_error,
     "setup": "003 的真值门牌 18号 → 19号（fields 与 ground_truth 同改）", "subset": "朗读稿 003",
     "expect": "normalize_error（C/D/E 仍对，只有门牌字段不符）",
     "note": "改真值而不是改代码：流水线输出不变，评测应把不一致记在 Stage B"},
    {"name": "recall_miss", "label": "recall_miss", "reachable": True, "text_only": False, "run": sc_recall_miss,
     "setup": f"MAX_DIST={RECALL_MAX_DIST}", "subset": f"合成集前 {RECALL_PROBE_N} 条非同音异形",
     "expect": "≥ 1 条 recall_miss，报告 miss_reason 分布",
     "note": "规格原定 0.05，实测 0 条 recall_miss，改用 0.01"},
    {"name": "rank_error", "label": "rank_error", "reachable": True, "text_only": False, "run": sc_rank_error,
     "setup": "W_SIM=0, W_PRIOR=1", "subset": f"合成集全量 720 条（--fast 只跑 source_id ∈ {FAST_RANK_SOURCES}）",
     "expect": "≥ 1 条 rank_error，报告 dominant_term 分布",
     "note": "前 240 行一条 rank_error 都没有，快速子集按 source_id 取（见 FAST_RANK_SOURCES 注释）"},
    {"name": "gate_over_accept", "label": "gate_over_accept", "reachable": True, "text_only": False,
     "run": sc_gate_over_accept,
     "setup": "SINGLE_HIT_MIN_COV=0, SIM_MIN=1e-6, MARGIN_MIN=1e-6", "subset": "清单里 4 条 has_address=false 的录音文本",
     "expect": "≥ 1 条 gate_over_accept（无地址却 confident）",
     "note": "有地址样本上 Top-1 错时决策树先贴 rank_error / recall_miss，错送标签主要只能从无地址负样本上来"},
    {"name": "gate_over_reject_sim", "label": "gate_over_reject", "reachable": True, "text_only": True,
     "run": sc_gate_over_reject_sim,
     "setup": "SIM_MIN=1.01", "subset": "24 句朗读稿", "expect": "24 条全部 gate_over_reject，gate_attrib 全是 sim",
     "note": "对应 §10 步骤 2 的 SIM_MIN=0.95 破坏测试；朗读稿 Top-1 音相似度全是 1.0，0.95 拦不下任何一条，必须越过 1"},
    {"name": "gate_over_reject_margin", "label": "gate_over_reject", "reachable": True, "text_only": True,
     "run": sc_gate_over_reject_margin,
     "setup": "MARGIN_MIN=1.0", "subset": "24 句朗读稿", "expect": "≥ 10 条 gate_over_reject 且 gate_attrib=margin",
     "note": "同区豁免：Top-2 与 Top-1 同行政区时分差闸门不拦，拦不满 24 条是规则如此"},
    {"name": "assembly_error", "label": "assembly_error", "reachable": True, "text_only": True,
     "run": sc_assembly_error,
     "setup": "003 的 ground_truth 末尾加「甲」，fields 不动", "subset": "朗读稿 003",
     "expect": "assembly_error（各阶段都过，只有串不同），且回归阻断规则报红",
     "note": "兜底标签：只说明各阶段判定都过，不指明拼装哪一步错"},
    {"name": "completion_error", "label": "completion_error", "reachable": False, "text_only": True,
     "run": sc_completion_error,
     "setup": f"地址库删 {COMPLETION_ITEM} 的全部路级条目（金阳南路、世纪城）", "subset": f"朗读稿 {COMPLETION_ITEM}",
     "expect": "不可达：该情形应记 db_missing（库缺 + 拦下 + 省市无从回溯），不得再误归到 completion_error",
     "note": "真正的补全错（Top-1 链对、回溯出的省市错）按构造不可达：chain_matches 要求行政区一致；"
             "这里跑的是曾经误归因的路径，passed = 现在记到了库缺头上"},
    {"name": "regression_red", "label": "regression_red", "reachable": True, "text_only": False,
     "run": sc_regression_red,
     "setup": "W_CONFLICT=1.0",
     "subset": f"合成集全量 720 条对 golden（--fast 只跑 source_id ∈ {FAST_CONFLICT_SOURCES}，对同子集的未破坏运行）；另跑 24 句朗读稿",
     "expect": "regression.compare 在合成集上 ≥ 1 项失败",
     "note": "对应 §10 步骤 6 的故意改坏测试；--fast 的子集比不了 720 条的 golden，只能对同子集的对照运行"},
    {"name": "skip", "label": "skip", "reachable": True, "text_only": True, "run": sc_skip,
     "setup": "构造 quality=silent 的记录直接调 attribution.attribute", "subset": "单条构造记录",
     "expect": "skip", "note": "决策树第 1 步，只看 quality"},
]


def run_scenario(sc: dict, ctx: Ctx) -> dict:
    """跑一个场景，统一记录：期望标签、观测分布、是否通过、耗时、一行证据。场景崩了记失败，不拖垮其余场景。"""
    res = {k: sc[k] for k in ("name", "label", "reachable", "setup", "subset", "expect", "note")}
    if not sc["reachable"] and not sc.get("run"):
        res.update(observed={}, passed=None, reachable_observed=False, elapsed=0.0, evidence="不可达，未运行",
                   finding=None, extra={})
        return res
    t = time.perf_counter()
    try:
        out = sc["run"](ctx)
    except Exception as e:  # noqa: BLE001
        out = {"passed": False, "evidence": f"场景出错 {type(e).__name__}: {e}", "error": traceback.format_exc(limit=4)}
    rows = out.get("rows")
    observed = out.get("observed") or (label_counts(rows) if rows is not None else {})
    reachable_observed = observed.get(sc["label"], 0) > 0
    passed = bool(out.get("passed"))
    if not sc["reachable"]:
        # 声明不可达的标签：跑的是"最像它"的情形，通过 = 标签确实没出现、且记到了真因上
        passed = passed and not reachable_observed
    res.update(observed=observed, passed=passed, reachable_observed=reachable_observed,
               elapsed=round(time.perf_counter() - t, 1), evidence=out.get("evidence", ""),
               finding=out.get("finding"), extra=out.get("extra", {}))
    if out.get("error"):
        res["error"] = out["error"]
    return res


# --------------------------------------------------------------------------
# b–e
# --------------------------------------------------------------------------


def section_samples(ctx: Ctx) -> dict:
    manifest = load_manifest(MANIFEST)
    recs = ctx.recordings()
    scored = [it for it in recs if it.get("scored")]
    read, synth = ctx.read(), ctx.synth()
    records = (
        [{"split": it["split"], "source": "text", "group": it["id"], "label_status": it.get("label_status"),
          "difficulty": it.get("difficulty")} for it in read]
        + [{"split": it["split"], "source": "synthetic", "group": it.get("source_id", it["id"]),
            "label_status": it.get("label_status"), "difficulty": it.get("difficulty")} for it in synth]
        + [{"split": it["split"], "source": "audio", "group": it["file"], "label_status": it.get("label_status"),
            "difficulty": it.get("difficulty")} for it in scored]
    )
    counts = effective_counts(records)
    return {
        "manifest_rows": len(manifest), "quality": dict(Counter(r.get("quality") for r in manifest)),
        "label_status_ok_rows": dict(Counter(it.get("label_status") for it in recs)),
        "recordings_ok": len(recs), "recordings_scored": len(scored), "read_items": len(read),
        "gap_table": _gap_table(scored + read), "effective": counts, "gate": gate_status(counts),
    }


def section_rules() -> dict:
    cfg = json.loads(GEN_CONFIG.read_text(encoding="utf-8")) if GEN_CONFIG.exists() else {}
    counts: dict[str, int] = cfg.get("counts_by_rule", {})
    try:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import gen_perturbed
        table = sorted({r.name for r in gen_perturbed.build_rules()})
        source = "scripts/gen_perturbed.py build_rules()"
    except ImportError as e:
        table = sorted(set(counts) | set(cfg.get("rules_without_samples", [])))
        source = f"gen_perturbed 导入失败（{e}），规则名退回 gen_config.json"
    names = table + [n for n in counts if n not in table]
    rows = sorted(({"rule": n, "n": counts.get(n, 0), "enough": counts.get(n, 0) >= MIN_N_FOR_PCT,
                    "in_confusion_table": n in table} for n in names), key=lambda x: (-x["n"], x["rule"]))
    zero = [x["rule"] for x in rows if x["n"] == 0]
    return {"source": source, "rows": rows, "zero": zero,
            "under_min": [x["rule"] for x in rows if 0 < x["n"] < MIN_N_FOR_PCT],
            "zero_vs_config_mismatch": sorted(set(zero) ^ set(cfg.get("rules_without_samples", []))),
            "config_note": cfg.get("rules_without_samples_note", "")}


def section_labels() -> dict:
    kinds = {"audio": "*_audio.json", "synthetic": "*_text_synthetic.json", "text": "*_text.json"}
    reports: dict[str, dict | None] = {}
    for kind, pattern in kinds.items():
        files = sorted(REPORTS.glob(pattern)) if REPORTS.is_dir() else []
        if not files:
            reports[kind] = None
            continue
        p = files[-1]           # 文件名以时间戳开头，字典序即时间序
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d.get("summary", {}).get("slices", {}).get("全部=全部", {})
        sims = [r["replay"]["top_sim"] for r in d.get("rows", [])
                if (r.get("replay") or {}).get("top_sim") is not None]
        reports[kind] = {"path": str(p.relative_to(ROOT)), "n": s.get("n"), "attribution": s.get("attribution", {}),
                         "top_sim_min": min(sims) if sims else None,
                         "top_sim_lt1": sum(1 for x in sims if x < 0.999), "n_replay": len(sims)}
    labels = {lb: {k: (v["attribution"].get(lb, 0) if v else None) for k, v in reports.items()} for lb in LABELS}
    return {"reports": reports, "labels": labels}


def section_identifiability() -> dict:
    files = sorted(TUNING.glob("*_proposal.json")) if TUNING.is_dir() else []
    if not files:
        return {"proposal": None, "identifiability": None, "note": "尚无提案"}
    p = files[-1]
    rel = str(p.relative_to(ROOT))
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"proposal": rel, "identifiability": None, "note": f"提案读不了：{e}"}
    ident = d.get("identifiability") if isinstance(d, dict) else None
    return {"proposal": rel, "identifiability": ident,
            "note": "" if ident is not None else "提案里没有 identifiability 字段"}


# --------------------------------------------------------------------------
# 能抓住什么、抓不住什么
# --------------------------------------------------------------------------


def closing_table(results: list[dict], samples: dict, labels: dict) -> list[dict]:
    res = {r["name"]: r for r in results}

    def ev(name: str) -> str:
        r = res.get(name)
        if r is None:
            return "（本次未跑）"
        if not r["reachable"]:
            return ("— 不可达（已核：" + r["evidence"] + "）") if r.get("evidence") and r["evidence"] != "不可达，未运行" else "— 不可达"
        return ("✓ " if r["passed"] else "✗ ") + r["evidence"]

    def extra(name: str, key: str, default=None):
        return (res.get(name) or {}).get("extra", {}).get(key, default)

    reps = labels["reports"]

    def on_audio(label: str) -> str:
        v = labels["labels"][label]["audio"]
        return "无 audio 报告" if v is None else f"最近 audio 报告里 ×{v}"

    def sims(kind: str) -> str:
        v = reps.get(kind)
        if not v or v["top_sim_min"] is None:
            return f"{kind} 无报告"
        return f"{kind} 最低 {v['top_sim_min']}（<1 的 {v['top_sim_lt1']}/{v['n_replay']} 条）"

    train_eff = samples["gate"]["train_eff"]
    flipped = extra("regression_red", "flipped", [])
    rows = [
        ("地址库 · db_missing", ev("db_missing"),
         "库缺但地名被原样保住时不记库缺（归因树第 2 步的细化）；这类样本若又被闸门拦下且只报到区，会落到 completion_error",
         f"oov_db 负样本与已确认录音（{on_audio('db_missing')}）"),
        ("声学层 · asr_unrecoverable", ev("asr_unrecoverable"),
         "text / 合成模式默认不算 Stage A，文本数据上这个标签恒为 0；合成集用本项目代价矩阵生成，验证不了音距离本身",
         f"带已确认 transcript_gold 的真实录音（{on_audio('asr_unrecoverable')}）"),
        ("归一化 · normalize_error", ev("normalize_error"),
         f"市/区/街道层没召回时扰动字原样留在输出里，记成闲话泄漏，混进 normalize_error（recall_miss 场景同批 "
         f"{extra('recall_miss', 'normalize_error', '?')} 条）",
         "按 Stage B 子项（方言词 / 数字 / 门牌 / 闲话）分开标注的真实口语样本"),
        ("召回 · recall_miss", ev("recall_miss"),
         f"只看真值链最深层条目；MAX_DIST 压到 {RECALL_MAX_DIST} 才出（0.05 时 0 条）；miss_reason 只观测到 "
         f"{extra('recall_miss', 'miss_reason', {})}，weak_filtered / window_miss 没被验证过",
         "真实录音里音距离落在 0.2–0.4 的地名错字"),
        ("排序 · rank_error", ev("rank_error"),
         f"要把 W_SIM 清零才造得出来；dominant_term 只观测到 {extra('rank_error', 'dominant_term', {})}；24 句朗读稿 Top-1 全精确命中，排序错只能在合成集上造",
         f"含近音竞争候选的真实样本；权重学习门槛 train ≥ 500 有效句（现 {train_eff}）"),
        ("权重（W_SIM/W_COV/W_PRIOR/W_DEPTH）", "只有合成集约束权重：见排序行与冲突惩罚行",
         f"合成集同一句 30 个变体只算 1 条有效样本，train 有效句 {train_eff}，远不到 §2.6 门槛；"
         f"W_CONFLICT=1.0 在 720 条里只翻转 {len(flipped)} 条，权重的可辨识性建立在个位数样本上",
         "train ≥ 500 有效句、各 difficulty ≥ 30"),
        ("冲突惩罚 · 回归变红", ev("regression_red"),
         f"24 句朗读稿对 W_CONFLICT=1.0 无反应（text 套件 {len(extra('regression_red', 'text_fails', []))} 项失败）；"
         f"合成集变红只靠 {'、'.join(flipped) or '无'}",
         "同一句里强命中两个行政区的真实口语（\"官渡区呈贡新区\"这类）"),
        ("闸门·放行 · gate_over_accept", ev("gate_over_accept"),
         "有地址样本上 Top-1 错时决策树先贴 rank_error / recall_miss；错送主要只能从无地址负样本上看，清单里仅 4 条，且要把三道阈值放到 ≈0 才触发一次",
         "no_address / injection 负样本 ≥ 150 条才能说错送率 ≤ 2%"),
        ("闸门·SIM · gate_over_reject", ev("gate_over_reject_sim"),
         f"Top-1 音相似度：{sims('text')}，{sims('synthetic')}；SIM 轴几乎平坦，0.62 附近的阈值在文本数据上看不见",
         f"音频集（{sims('audio')}）"),
        ("闸门·MARGIN · gate_over_reject", ev("gate_over_reject_margin"),
         "Top-2 与 Top-1 同行政区时分差闸门不拦（同区豁免），这部分的多余确认和错送都不经过它",
         "同区不同路的真实歧义样本"),
        ("补全 · completion_error", ev("completion_error"),
         "真正的补全错（Top-1 链对、回溯出的省市错）按构造不可达：chain_matches 要求行政区一致；实测唯一可达路径是库缺 + 闸门正确拦下的误归因",
         "只报街路（street_only）且库里有跨城重名路的真实录音"),
        ("拼装 · assembly_error", ev("assembly_error"),
         "兜底标签，不指明拼装哪一步错，需人工翻 json", "—"),
        ("质量 · skip", ev("skip"),
         f"quality≠ok 的行在 manifest_to_items 就被滤掉（清单里 {extra('skip', 'non_ok_rows', '?')} 行），真实报告里 skip 恒为 0；quality 判定本身不在这里验证",
         "—"),
    ]
    return [{"stage": a, "catch": b, "miss": c, "data": d} for a, b, c, d in rows]


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def _cell(v) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s.replace("|", "\\|").replace("\n", " ")


def _counts(d: dict) -> str:
    return ", ".join(f"{k}×{v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) or "-"


def _status(r: dict) -> str:
    if not r["reachable"]:
        if r["passed"] is None:
            return "— 不可达"
        return "— 不可达（已核）" if r["passed"] else "✗ 不可达但被误归因"
    return "✓ 通过" if r["passed"] else "✗ 失败"


def _table(rows: list[dict]) -> list[str]:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_cell(r.get(c, "")) for c in cols) + " |" for r in rows]
    return lines


def build_md(rep: dict) -> str:
    sec = rep["sections"]
    sc = sec["scenarios"]
    mode = "fast" if rep["fast"] else "full"
    n_pass = sum(1 for r in sc if r["passed"])
    n_unreach = sum(1 for r in sc if not r["reachable"])
    failed = rep["failed"]
    p = rep["params"]
    L = [
        "---", "type: report", f"title: 评测自检 · {mode}",
        f"description: 故意破坏 {len(sc)} 个场景，{n_pass} 通过 / {len(failed)} 失败 / {n_unreach} 不可达；"
        "附样本缺口、规则覆盖、真实数据上的标签与参数可辨识性",
        "tags: [评测, 自检, 故意破坏, 归因]", f"timestamp: {rep['created'][:10]}", "---", "",
        f"# 评测自检（{mode}）", "",
        f"- 时间 {rep['created']}，耗时 {rep['elapsed']:.0f}s，参数 v{p.get('version')}（{p.get('source')}）",
        "- 结论：" + ("声明可达的场景全部贴出了预期标签" if not failed else f"{len(failed)} 个场景没贴出预期标签：{'、'.join(failed)}"),
        "",
        "## a. 故意破坏：坏了会不会叫", "",
        "| # | 场景 | 期望标签 | 破坏方式 | 子集 | 观测 | 结果 | 耗时 | 证据 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(sc, 1):
        L.append(f"| {i} | {r['name']} | {r['label']} | {_cell(r['setup'])} | {_cell(r['subset'])} | "
                 f"{_cell(_counts(r['observed']))} | {_status(r)} | {r['elapsed']}s | {_cell(r['evidence'])} |")
    L += ["", "场景说明：", ""] + [f"- **{r['name']}**：期望 {r['expect']}。{r['note']}" for r in sc]
    errors = [r for r in sc if r.get("error")]
    if errors:
        L += ["", "场景异常：", ""] + [f"- {r['name']}：`{_cell(r['error'])}`" for r in errors]
    if rep["findings"]:
        L += ["", "### 意外发现", ""] + [f"- {f}" for f in rep["findings"]]

    s = sec["samples"]
    eff = s["effective"]
    g = s["gate"]
    L += [
        "", "## b. 样本缺口与标注状态", "",
        f"- 清单 {s['manifest_rows']} 行，quality 分布 {_counts(s['quality'])}",
        f"- quality=ok 的 {s['recordings_ok']} 行 label_status 分布 {_counts(s['label_status_ok_rows'])}；可计分 {s['recordings_scored']} 条",
        f"- 缺口表口径：可计分录音 {s['recordings_scored']} 条 + 朗读稿 {s['read_items']} 句（合成变体不是新样本，不进缺口表）",
        "", *s["gap_table"], "",
        "有效样本（按句计：合成集同一 source_id 只算 1 条）：", "",
        "| 划分 | 有效 | 原始 | draft |", "|---|---|---|---|",
        *[f"| {k} | {v['effective']} | {v['raw']} | {v['drafts']} |" for k, v in eff.items()],
        "", f"§2.6 门槛：{g['detail']}；权重学习 {'达标' if g['met'] else '未达标'}，校准 {'达标' if g['calib_ok'] else '未达标'}",
    ]

    rc = sec["rule_coverage"]
    L += ["", "## c. 合成扰动集规则覆盖", "", f"规则名来源：{rc['source']}；计数来自 gen_config.json counts_by_rule。", "",
          f"| 规则 | n | ≥ {MIN_N_FOR_PCT} | 混淆表规则 |", "|---|---|---|---|"]
    L += [f"| {_cell(x['rule'])} | {x['n']} | {'✓' if x['enough'] else '✗'} | {'是' if x['in_confusion_table'] else '否（生成器自有）'} |"
          for x in rc["rows"]]
    L += ["", f"- 0 条的规则（{len(rc['zero'])}）：{'、'.join(rc['zero']) or '无'}",
          f"- 不足 {MIN_N_FOR_PCT} 条的规则（{len(rc['under_min'])}）：{'、'.join(rc['under_min']) or '无'}"]
    if rc["zero_vs_config_mismatch"]:
        L.append(f"- ⚠ 与 gen_config.rules_without_samples 不一致：{'、'.join(rc['zero_vs_config_mismatch'])}")
    if rc["config_note"]:
        L.append(f"- 生成器说明：{rc['config_note']}")

    lb = sec["labels_observed"]
    L += ["", "## d. 真实数据上出现过的标签", ""]
    for kind, v in lb["reports"].items():
        L.append(f"- {kind}：" + ("无报告" if not v else
                 f"{v['path']}，n={v['n']}，Top-1 音相似度最低 {v['top_sim_min']}（<1 的 {v['top_sim_lt1']}/{v['n_replay']} 条）"))
    L += ["", "| 标签 | audio | 合成 | text |", "|---|---|---|---|"]
    for label, v in lb["labels"].items():
        L.append(f"| {label} | " + " | ".join("无报告" if v[k] is None else str(v[k]) for k in ("audio", "synthetic", "text")) + " |")

    idf = sec["identifiability"]
    L += ["", "## e. 参数可辨识性", ""]
    if idf["identifiability"] is None:
        L.append(f"{idf['note']}" + (f"（{idf['proposal']}）" if idf["proposal"] else ""))
    else:
        L += [f"来源：{idf['proposal']}", ""]
        ident = idf["identifiability"]
        if isinstance(ident, dict):
            L += _table([{"参数": k, **(v if isinstance(v, dict) else {"值": v})} for k, v in ident.items()])
        elif isinstance(ident, list):
            L += _table([x if isinstance(x, dict) else {"值": x} for x in ident])
        else:
            L.append(f"`{_cell(ident)}`")

    L += ["", "## 能抓住什么、抓不住什么", "", "| 环节 | 能抓住（证据） | 抓不住（原因） | 需要什么数据 |", "|---|---|---|---|"]
    L += [f"| {_cell(r['stage'])} | {_cell(r['catch'])} | {_cell(r['miss'])} | {_cell(r['data'])} |" for r in rep["closing"]]
    return "\n".join(L) + "\n"


def run_selfcheck(fast: bool = False, only: list[str] | None = None, out_dir: Path = OUT_DIR,
                  quiet: bool = False) -> dict:
    """跑场景 + 汇总五节，写 latest.md / latest.json，返回报告 dict。only 只跑指定名字的场景（测试用）。"""
    t0 = time.perf_counter()
    ctx = Ctx(fast)
    results = []
    for sc in SCENARIOS:
        if only and sc["name"] not in only:
            continue
        r = run_scenario(sc, ctx)
        results.append(r)
        if not quiet:
            print(_line(r), flush=True)
    samples = section_samples(ctx)
    labels = section_labels()
    rep = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "fast": fast, "params": rank_mod.params_info(),
        "sections": {"scenarios": results, "samples": samples, "rule_coverage": section_rules(),
                     "labels_observed": labels, "identifiability": section_identifiability()},
        "closing": closing_table(results, samples, labels),
        "findings": [r["finding"] for r in results if r.get("finding")],
        # 可达却没贴出标签，或声明不可达却又出现了（误归因回潮），都算失败
        "failed": [r["name"] for r in results if r["passed"] is False],
    }
    rep["elapsed"] = round(time.perf_counter() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out_dir / "latest.md").write_text(build_md(rep), encoding="utf-8")
    return rep


def _line(r: dict) -> str:
    mark = {True: "✓", False: "✗", None: "—"}[r["passed"]]
    ev = r["evidence"] if len(r["evidence"]) <= 110 else r["evidence"][:110] + "…"
    return f"{mark} {r['name']:<24} {r['elapsed']:>6.1f}s  {_counts(r['observed'])[:48]:<48}  {ev}"


def main() -> None:
    ap = argparse.ArgumentParser(description="评测自检：故意破坏，断言归因标签")
    ap.add_argument("--fast", action="store_true", help="合成集场景只跑已知敏感的 source_id 组")
    ap.add_argument("--ci", action="store_true", help="只打印失败项和一行汇总")
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()
    out = Path(a.out)
    rep = run_selfcheck(fast=a.fast, out_dir=out, quiet=a.ci)
    sc = rep["sections"]["scenarios"]
    if a.ci:
        for r in sc:
            if r["name"] in rep["failed"]:
                print(_line(r))
    n_unreach = sum(1 for r in sc if not r["reachable"])
    print(f"\n评测自检（{'fast' if a.fast else 'full'}）：{len(sc)} 个场景，{sum(1 for r in sc if r['passed'])} 通过 / "
          f"{len(rep['failed'])} 失败 / {n_unreach} 不可达，耗时 {rep['elapsed']:.0f}s")
    if not a.ci and rep["findings"]:
        print("意外发现：\n" + "\n".join(f"  - {f}" for f in rep["findings"]))
    print(f"报告: {out / 'latest.md'}")
    sys.exit(1 if rep["failed"] else 0)


if __name__ == "__main__":
    main()
