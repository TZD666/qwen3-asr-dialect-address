"""错误归因决策树的单测（评测体系设计 §5）。

纯 assert，`python tests/test_attribution.py` 直接跑，装了 pytest 也能收。

归因表是给优化方向排序用的：哪个标签多就先改哪个环节。所以这里要保住两件事——
**每个标签都能被触发**（不然那类错永远看不见），以及**标签互斥且顺序固定**
（同一条样本不能今天算排序错、明天算库缺，否则前后两份报告没法比）。

互斥性用一条"所有触发条件同时成立"的记录验证：从上往下逐条拆掉触发条件，
标签必须沿着文档写的顺序一格一格往下走。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr.address_db import AddressDB  # noqa: E402

from attribution import LABELS, attribute  # noqa: E402
from run_eval import run_once  # noqa: E402

EVAL_JSON = ROOT / "data" / "eval" / "xinan_guanhua.json"

_CACHE: dict = {}


def base_rec(**overrides) -> dict:
    """一条"端到端错了，但每个阶段都过"的记录 → 兜底标签 assembly_error。

    每个用例只覆盖它要触发的那一格，其余保持通过，标签才归得干净。
    """
    rec = {
        "id": "T00",
        "quality": "ok",
        "pred": "四川省成都市武侯区玉林南路7栋",
        "fields": {"unverified": ""},
        "truth": "四川省成都市武侯区玉林南路玉林小区7栋",
        "e2e": {"exact": False, "deliverable": False, "geo_ok": True, "admin_all": True},
        "gold_in_db": True,
        "gold_missing": [],
        "gold_deepest_name": "玉林小区",
        "stage_a": {"name_phon_dist_max": 0.05,
                    "names": [{"name": "玉林南路", "seg": "玉林南路", "dist_phon_w": 0.05}]},
        "stage_b": {"ok": True, "lexicon_hit": True, "number_ok": True, "tail_exact": True,
                    "illegal_number_caught": True, "filler_leak": 0.0, "tail_fields": {}},
        "stage_c": {"evaluable": True, "recall_chain@all": True, "ok": True},
        "stage_d": {"evaluable": True, "top1_correct": True, "mrr": 1.0, "ok": True},
        "stage_e": {"decision": "confident", "quadrant": "TP", "quadrant_geo": "TP", "ok": True},
        "stage_f": {"evaluable": True, "backfilled_levels": ["province", "city"],
                    "backfill_correct": True, "ok": True},
    }
    rec.update(overrides)
    return rec


# 每个触发条件对应的字段覆盖，顺序 = attribution.py 文档里的决策树顺序
TRIGGERS = {
    "db_missing": {"gold_missing": ["交通巷"]},
    "asr_unrecoverable": {"stage_a": {"name_phon_dist_max": 0.82,
                                      "names": [{"name": "交通巷", "seg": "高通巷", "dist_phon_w": 0.82}]}},
    "stage_b_failed": {"stage_b": {"ok": False, "lexicon_hit": True, "number_ok": False,
                                   "tail_exact": True, "illegal_number_caught": True,
                                   "filler_leak": 0.0, "tail_fields": {}}},
    "recall_miss": {"stage_c": {"evaluable": True, "recall_chain@all": False,
                                "miss_reason": "dist_over_max",
                                "miss_detail": "最近窗口「高通巷」vs「交通巷」dist=0.500 > 0.4", "ok": False}},
    "rank_error": {"stage_d": {"evaluable": True, "top1_correct": False, "mrr": 0.5,
                               "dominant_term": "conflict", "delta": {"conflict": -0.2},
                               "gold_chain": "云南省昆明市呈贡区雨花路", "top_chain": "云南省昆明市官渡区",
                               "gold_total": 0.71, "top_total": 0.80, "ok": False}},
    "gate_over_accept": {"stage_e": {"decision": "confident", "quadrant": "FA", "quadrant_geo": "FA",
                                     "gate_attrib": "margin", "ok": False}},
    "completion_error": {"stage_f": {"evaluable": True, "backfilled_levels": ["province", "city"],
                                     "backfill_correct": False, "ok": False}},
}


# ==========================================================================
# 每个标签至少一条构造用例
# ==========================================================================


def test_skip_when_recording_quality_is_not_ok():
    label, detail = attribute(base_rec(quality="silent"))
    assert label == "skip"
    assert "silent" in detail


def test_correct_and_auto_accepted_sample_gets_no_label():
    label, detail = attribute(base_rec(e2e={"exact": True}))
    assert label is None
    assert detail == ""


def test_correct_but_held_back_sample_is_still_gate_over_reject():
    """地址对了却被拦下来让人确认：不是错送，但也是系统的失败（多余确认），要计。

    这一条不走下面的决策树——树是给"端到端错了"的样本用的，FR 是唯一的例外。
    """
    rec = base_rec(e2e={"exact": True},
                   stage_e={"decision": "ambiguous", "quadrant": "FR", "quadrant_geo": "FR",
                            "gate_attrib": "margin", "ok": False})
    label, detail = attribute(rec)
    assert label == "gate_over_reject"
    assert "gate=margin" in detail and "地址正确" in detail


def test_db_missing_when_the_missing_name_is_not_in_the_output():
    rec = base_rec(gold_missing=["交通巷"], pred="四川省成都市金牛区花牌坊街38号")
    label, detail = attribute(rec)
    assert label == "db_missing"
    assert "交通巷" in detail


def test_db_missing_when_gold_deepest_level_is_not_in_db_at_all():
    """真值只到区级、区级条目也不在库里：missing 为空，靠 gold_in_db 兜住。"""
    label, detail = attribute(base_rec(gold_in_db=False, gold_missing=[], gold_deepest_name="交通巷"))
    assert label == "db_missing"
    assert "交通巷" in detail


def test_oov_name_preserved_verbatim_is_not_db_missing():
    """流水线对库里没有的段是原样保留的。保住了就不是库的锅，错在别处。

    这是 attribution.py 相对 §5 原文的那处细化，丢了它会把归一化的问题算到库头上。
    """
    rec = base_rec(gold_missing=["交通巷"], pred="四川省成都市金牛区花牌坊街交通巷38号")
    label, _ = attribute(rec)
    assert label != "db_missing"
    assert label == "assembly_error"
    # 落在 unverified 字段里也算保住了
    rec2 = base_rec(gold_missing=["交通巷"], pred="四川省成都市金牛区花牌坊街38号",
                    fields={"unverified": "交通巷"})
    assert attribute(rec2)[0] != "db_missing"


def test_asr_unrecoverable_when_name_phon_distance_exceeds_max():
    label, detail = attribute(base_rec(**TRIGGERS["asr_unrecoverable"]))
    assert label == "asr_unrecoverable"
    assert "交通巷" in detail and "高通巷" in detail


def test_normalize_error_only_when_downstream_is_clean():
    label, detail = attribute(base_rec(**TRIGGERS["stage_b_failed"]))
    assert label == "normalize_error"
    assert "数字转换错" in detail


def test_normalize_error_detail_lists_every_failed_sub_check():
    rec = base_rec(stage_b={
        "ok": False, "lexicon_hit": False, "lexicon_leaked": ["屋头"], "number_ok": False,
        "tail_exact": False, "tail_fields": {"house_no": False, "room": True},
        "illegal_number_caught": False, "illegal_sequences": ["八百一百八号"],
        "filler_leak": 0.2, "filler_leaked_chars": "巴适的板",
    })
    label, detail = attribute(rec)
    assert label == "normalize_error"
    for fragment in ("屋头", "数字转换错", "house_no", "八百一百八号", "巴适的板"):
        assert fragment in detail, fragment


def test_recall_miss_carries_the_miss_reason():
    label, detail = attribute(base_rec(**TRIGGERS["recall_miss"]))
    assert label == "recall_miss"
    assert detail.startswith("dist_over_max")


def test_rank_error_carries_the_dominant_term():
    label, detail = attribute(base_rec(**TRIGGERS["rank_error"]))
    assert label == "rank_error"
    assert "dominant=conflict" in detail
    assert "云南省昆明市呈贡区雨花路" in detail


def test_gate_over_accept_for_false_accept():
    label, detail = attribute(base_rec(**TRIGGERS["gate_over_accept"]))
    assert label == "gate_over_accept"
    assert "gate=margin" in detail and "decision=confident" in detail


def test_gate_over_reject_for_false_reject():
    rec = base_rec(stage_e={"decision": "ambiguous", "quadrant": "FR", "quadrant_geo": "FR",
                            "gate_attrib": "margin", "ok": False})
    label, detail = attribute(rec)
    assert label == "gate_over_reject"
    assert "decision=ambiguous" in detail


def test_completion_error_when_backfilled_admin_levels_are_wrong():
    label, detail = attribute(base_rec(**TRIGGERS["completion_error"]))
    assert label == "completion_error"
    assert "province" in detail and "city" in detail


def test_assembly_error_is_the_fallback():
    label, detail = attribute(base_rec())
    assert label == "assembly_error"
    assert "各阶段均通过" in detail


def test_every_label_in_the_tuple_is_reachable():
    """LABELS 里的每一格都要有构造用例，否则那类错在报告里永远是 0。"""
    produced = {
        attribute(base_rec(quality="silent"))[0],
        attribute(base_rec(gold_missing=["交通巷"], pred="四川省成都市金牛区花牌坊街38号"))[0],
        attribute(base_rec(**TRIGGERS["asr_unrecoverable"]))[0],
        attribute(base_rec(**TRIGGERS["stage_b_failed"]))[0],
        attribute(base_rec(**TRIGGERS["recall_miss"]))[0],
        attribute(base_rec(**TRIGGERS["rank_error"]))[0],
        attribute(base_rec(**TRIGGERS["gate_over_accept"]))[0],
        attribute(base_rec(stage_e={"decision": "reject", "quadrant": "FR", "quadrant_geo": "FR",
                                    "gate_attrib": "sim", "ok": False}))[0],
        attribute(base_rec(**TRIGGERS["completion_error"]))[0],
        attribute(base_rec())[0],
    }
    assert produced == set(LABELS)


def test_attribute_returns_a_label_and_a_detail_string():
    for rec in (base_rec(), base_rec(e2e={"exact": True}), base_rec(quality="clipped")):
        out = attribute(rec)
        assert isinstance(out, tuple) and len(out) == 2
        label, detail = out
        assert label is None or label in LABELS
        assert isinstance(detail, str)


# ==========================================================================
# 互斥性：所有触发条件同时成立时，标签沿决策树逐级下移
# ==========================================================================


def all_triggers_rec() -> dict:
    rec = base_rec(pred="四川省成都市金牛区花牌坊街38号")
    for ov in TRIGGERS.values():
        rec.update(copy.deepcopy(ov))
    return rec


def test_all_triggers_at_once_yields_the_first_label_in_the_tree():
    assert attribute(all_triggers_rec())[0] == "db_missing"


def test_labels_walk_down_the_tree_in_the_documented_order():
    """从上往下逐条拆掉触发条件，标签必须一格一格往下走，中间不跳、不回头。

    normalize_error 排在第 4 步，但它额外要求 C/D/E 都对——所以在这条路径上
    它出现在闸门被修好之后，而不是紧跟 asr_unrecoverable。这正是决策树想要的：
    下游还坏着的时候，B 的错是传导，不单独计。
    """
    rec = all_triggers_rec()
    clean = base_rec()
    walk = [
        ("db_missing", None),
        ("asr_unrecoverable", "gold_missing"),
        ("recall_miss", "stage_a"),
        ("rank_error", "stage_c"),
        ("gate_over_accept", "stage_d"),
        ("normalize_error", "stage_e"),
        ("completion_error", "stage_b"),
        ("assembly_error", "stage_f"),
    ]
    seen = []
    for expected, fix in walk:
        if fix is not None:
            rec[fix] = copy.deepcopy(clean[fix])
        label = attribute(rec)[0]
        assert label == expected, f"修好 {fix} 之后期望 {expected}，实得 {label}"
        seen.append(label)
    assert len(set(seen)) == len(seen)      # 每一格只经过一次
    # skip 在树顶被 quality 单独挡掉；gate_over_reject 与 gate_over_accept 是第 7 步的
    # 两个互斥分支（FA / FR），同一条记录只可能走其中一条
    assert set(seen) == set(LABELS) - {"skip", "gate_over_reject"}


def test_skip_wins_over_every_other_trigger():
    rec = all_triggers_rec()
    rec["quality"] = "silent"
    assert attribute(rec)[0] == "skip"


def test_an_exact_answer_short_circuits_the_whole_tree():
    """端到端对了就不进决策树，哪怕每个阶段的中间量都写着"错了"。

    唯一的例外是被闸门拦下的 FR——见 test_correct_but_held_back_sample_...。
    这里的记录停在 FA，所以标签必须是 None。
    """
    rec = all_triggers_rec()
    rec["e2e"] = {"exact": True}
    assert rec["stage_e"]["quadrant"] == "FA"
    assert attribute(rec)[0] is None


# ==========================================================================
# 真流水线：故意从库里删条目（§10 步骤 2 的破坏性验证）
# ==========================================================================


def eval_items() -> list[dict]:
    data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))["items"]
    items = copy.deepcopy(data)
    for it in items:
        it.setdefault("transcript_gold", it["spoken"])
        it.setdefault("scored", True)
    return items


def text_run(dropped_names: frozenset) -> list[dict]:
    if dropped_names not in _CACHE:
        full = AddressDB.default()
        db = (full if not dropped_names else
              AddressDB([e for e in full.entries if e.name not in dropped_names], provider=full.provider))
        _CACHE[dropped_names] = run_once("text", eval_items(), db, None, None, False, set(), verbose=False)
    return _CACHE[dropped_names]


def test_full_db_text_mode_has_no_errors_to_attribute():
    """朗读稿 = 完美 ASR。库完整时 24/24，一个归因标签都不该出现。"""
    rows = text_run(frozenset())
    assert len(rows) == 24
    assert all(r["e2e"]["exact"] for r in rows), [r["id"] for r in rows if not r["e2e"]["exact"]]
    assert all(r["attribution"] is None for r in rows)


def test_dropping_a_sibling_road_is_preserved_not_corrected():
    """天府三街 从库里删掉后，同区还有 天府一街/天府五街（音距离 0.25）。
    兄弟路守门（rank.is_sibling_mismatch）：只差一个序数字且两个音毫不相干，说明说的是库里没有的
    那条，名字原样保留而不是"纠"成邻居——输出仍然精确，没有归因标签。"""
    rows = {r["id"]: r for r in text_run(frozenset({"天府三街"}))}
    assert all(r["attribution"] is None for r in rows.values()), \
        {i: r["attribution"] for i, r in rows.items() if r["attribution"]}
    r = rows["024"]
    assert "天府三街" in r["pred"] and "天府一街" not in r["pred"]
    assert r["gold_missing"] == ["天府三街"]
    assert r["oov_preserved"] is True
    assert r["over_correction"] is False


def test_dropping_an_entry_whose_spoken_form_differs_is_attributed_db_missing():
    """犀浦街道 删掉后，005 说的是"犀浦镇"：没有库条目就没法规范成"犀浦街道"，输出保留原文，
    与真值不符 → db_missing；但名字词干保住了，不算过度纠正。"""
    rows = {r["id"]: r for r in text_run(frozenset({"犀浦街道"}))}
    bad = {i: r["attribution"] for i, r in rows.items() if r["attribution"]}
    assert bad == {"005": "db_missing"}
    r = rows["005"]
    assert "犀浦镇" in r["pred"] and "犀浦街道" not in r["pred"]
    assert r["gold_missing"] == ["犀浦街道"]
    assert r["oov_preserved"] is True
    assert r["over_correction"] is False
    assert "犀浦街道" in r["attribution_detail"]


def test_dropping_two_entries_attributes_both_and_leaves_the_rest_exact():
    """解放碑街道（说的是"解放碑"）和 犀浦街道（说的是"犀浦镇"）同时删：两条都是库缺，其余 22 条精确。"""
    dropped = frozenset({"解放碑街道", "犀浦街道"})
    rows = {r["id"]: r for r in text_run(dropped)}
    labelled = {i: r["attribution"] for i, r in rows.items() if r["attribution"]}
    assert labelled == {"003": "db_missing", "005": "db_missing"}
    assert all(r["e2e"]["exact"] for i, r in rows.items() if i not in labelled)
    assert rows["003"]["over_correction"] is False
    assert rows["005"]["over_correction"] is False


def test_every_attribution_produced_by_the_real_pipeline_is_a_known_label():
    for dropped in (frozenset(), frozenset({"天府三街"}), frozenset({"解放碑街道", "犀浦街道"})):
        for r in text_run(dropped):
            assert r["attribution"] is None or r["attribution"] in LABELS, r


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
