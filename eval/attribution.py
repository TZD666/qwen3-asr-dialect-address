"""错误归因决策树（评测体系设计 §5）。每条端到端不对的样本贴一个且只贴一个标签。

    1. quality != ok                              → skip
    2. gold_in_db == false 且缺失地名没被原样保住    → db_missing
    3. Stage A: 地名片段音距离 > 0.40              → asr_unrecoverable
    4. Stage B 任一指标失败，且 C/D/E 均正确        → normalize_error
    5. Stage C: recall_chain@all == false          → recall_miss（附 miss_reason）
    6. Stage D: top1_correct == false              → rank_error（附 dominant_term）
    7. Stage E: FA → gate_over_accept / FR → gate_over_reject（附 gate_attrib）
    8. Stage F: backfill_correct == false          → completion_error
    9. 以上都过但端到端仍错                         → assembly_error

顺序即优先级：越靠上的原因越上游，上游错了下游的错是它的传导，不重复计。
按固定顺序、先命中先停，标签天然互斥。

与 §5 原文的一处细化：第 2 步不是"gold_in_db=false 就一律 db_missing"，而是要求
**缺失的那个地名没有被原样交出去**。流水线对库里没有的段是原样保留的，
库缺但保住了、错在门牌的样本，记 db_missing 会把归一化的问题算到库头上。
"""

from __future__ import annotations

from dialect_addr import rank as rank_mod

LABELS = (
    "skip", "db_missing", "asr_unrecoverable", "normalize_error", "recall_miss",
    "rank_error", "gate_over_accept", "gate_over_reject", "completion_error", "assembly_error",
)


def attribute(rec: dict) -> tuple[str | None, str]:
    """rec 是 run_eval 组装的一条样本记录。返回 (标签, 说明)；端到端正确时标签为 None。"""
    if rec.get("quality", "ok") != "ok":
        return "skip", f"quality={rec.get('quality')}"
    e2e = rec["e2e"]
    a, b, c, d, e, f = (rec.get(k) or {} for k in ("stage_a", "stage_b", "stage_c", "stage_d", "stage_e", "stage_f"))
    if e2e["exact"]:
        # 地址对了但被拦下来让人确认：不是错送，但也是系统的失败（多余确认），按 §5 第 7 步记
        if e.get("quadrant") == "FR":
            return "gate_over_reject", f"gate={e.get('gate_attrib')} decision={e.get('decision')}（地址正确）"
        return None, ""

    # 2. 库缺：真值里有库中没有的地名，且输出里没有按规范名给出它
    #    （原样保住了的不算——那是流水线该有的行为，错在别处）
    missing = rec.get("gold_missing") or []
    out = rec["pred"] + rec["fields"].get("unverified", "")
    lost = [m for m in missing if m not in out]
    if lost:
        return "db_missing", f"库中无「{'、'.join(lost)}」，输出为「{rec['pred']}」"
    if not rec.get("gold_in_db", True) and not missing:
        return "db_missing", f"库中无「{rec.get('gold_deepest_name', '') or '真值区级条目'}」"
    # 地名原样保住了、但因为库里没有它而拦下（reject/partial），省市区无从回溯：还是库缺，
    # 不是补全错。补全只在有候选链时才谈得上——评测自检曾把这种情形误归到 completion_error。
    if missing and rec.get("decision") in ("reject", "partial", "empty") and not e2e.get("admin_all", True):
        return "db_missing", f"库中无「{'、'.join(missing)}」，决策 {rec.get('decision')}，省市区无法回溯"

    # 3. 声学层错到拼音也救不回
    if a.get("name_phon_dist_max") is not None and a["name_phon_dist_max"] > rank_mod.MAX_DIST:
        worst = max(a.get("names", []), key=lambda p: p["dist_phon_w"], default={})
        return "asr_unrecoverable", (
            f"「{worst.get('name')}」在 ASR 输出里最近片段「{worst.get('seg')}」音距离 {worst.get('dist_phon_w')} > {rank_mod.MAX_DIST}"
        )

    # 4. 归一化：B 错，且下游 C/D/E（可评的部分）都对
    c_ok = c.get("ok", True) if c.get("evaluable") else True
    d_ok = d.get("ok", True) if d.get("evaluable") else True
    e_ok = e.get("quadrant_geo") in ("TP", "TR") if e else True
    if b and not b.get("ok", True) and c_ok and d_ok and e_ok:
        why = []
        if not b.get("lexicon_hit", True):
            why.append(f"方言词泄漏 {b.get('lexicon_leaked')}")
        if not b.get("number_ok", True):
            why.append("数字转换错")
        if not b.get("tail_exact", True):
            bad = [k for k, v in b.get("tail_fields", {}).items() if not v]
            why.append(f"门牌字段错 {bad}")
        if not b.get("illegal_number_caught", True):
            why.append(f"非法数串被猜成数字 {b.get('illegal_sequences')}")
        if b.get("filler_leak", 0) > 0:
            why.append(f"闲话泄漏「{b.get('filler_leaked_chars')}」")
        return "normalize_error", "；".join(why)

    # 5. 召回
    if c.get("evaluable") and not c.get("recall_chain@all"):
        return "recall_miss", f"{c.get('miss_reason')}: {c.get('miss_detail', '')}"

    # 6. 排序
    if d.get("evaluable") and not d.get("top1_correct"):
        return "rank_error", (
            f"dominant={d.get('dominant_term')} Δ={d.get('delta')} 真值「{d.get('gold_chain')}」"
            f"({d.get('gold_total')}) vs 选中「{d.get('top_chain')}」({d.get('top_total')})"
        )

    # 7. 闸门
    if e.get("quadrant_geo") == "FA":
        return "gate_over_accept", f"gate={e.get('gate_attrib')} decision={e.get('decision')}"
    if e.get("quadrant_geo") == "FR":
        return "gate_over_reject", f"gate={e.get('gate_attrib')} decision={e.get('decision')}"

    # 8. 补全
    if f.get("evaluable") and not f.get("backfill_correct"):
        return "completion_error", f"回溯层级 {f.get('backfilled_levels')} 与真值不符"

    # 9. 兜底
    return "assembly_error", f"各阶段均通过但输出「{rec['pred']}」≠ 真值；需人工看拼装/字段映射"
