"""六个阶段打分函数 + 小样本统计的单测（评测体系设计 §3 / §9）。

纯 assert，`python tests/test_stages.py` 直接跑，装了 pytest 也能收。

这些函数是评测的量尺。量尺自己错了，报告上的每个数都不可信，而且错得**看不出来**——
所以每个阶段至少一条"应该对"和一条"应该错"，再加真流水线跑出来的端到端锚点。
真值链、召回、排序三段用 24 条评测集的真实结果做锚，闸门与统计用构造数据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr.address_db import AddressDB  # noqa: E402
from dialect_addr.pipeline import Pipeline  # noqa: E402

import stages as S  # noqa: E402
import stats as ST  # noqa: E402

EVAL_JSON = ROOT / "data" / "eval" / "xinan_guanhua.json"

_CACHE: dict = {}


def db() -> AddressDB:
    if "db" not in _CACHE:
        _CACHE["db"] = AddressDB.default()
    return _CACHE["db"]


def items() -> dict[str, dict]:
    if "items" not in _CACHE:
        data = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
        _CACHE["items"] = {it["id"]: it for it in data["items"]}
    return _CACHE["items"]


def run(text: str):
    """跑一遍文本模式流水线（不需要模型），按文本缓存——24 条全跑一遍要几秒。"""
    cache = _CACHE.setdefault("runs", {})
    if text not in cache:
        if "pipe" not in _CACHE:
            _CACHE["pipe"] = Pipeline(db=db(), asr=None, two_pass=False)
        cache[text] = _CACHE["pipe"].process_text(text, None)
    return cache[text]


# ==========================================================================
# cer
# ==========================================================================


def test_cer_identical_is_zero():
    assert S.cer("民权路18号", "民权路18号") == 0.0
    assert S.cer("", "") == 0.0


def test_cer_empty_ref():
    # 参考为空：假设也为空才算对，否则整条都是错的
    assert S.cer("", "") == 0.0
    assert S.cer("民权路", "") == 1.0


def test_cer_single_substitution():
    # 三个字错一个 → 1/3
    assert abs(S.cer("民全路", "民权路") - 1 / 3) < 1e-9
    # 参考长度做分母，不是假设长度
    assert abs(S.cer("民权", "民权路") - 1 / 3) < 1e-9


# ==========================================================================
# 真值链 gold_chain / chain_matches
# ==========================================================================


def test_gold_chain_deepest_is_road_level_entry_in_db():
    g = S.gold_chain(db(), items()["003"]["fields"])
    assert g.deepest is not None and g.deepest.name == "民权路"
    assert g.deepest_field == "road"
    assert g.deepest_name == "民权路"
    assert g.in_db is True
    assert g.missing == []
    assert g.evaluable is True
    assert {lv: e.name for lv, e in g.admin.items()} == {
        "province": "重庆市", "city": "重庆市", "district": "渝中区",
    }


def test_gold_chain_missing_name_keeps_deeper_entry_evaluable():
    """库里没有的层级记进 missing、in_db 转 false，但更浅的路级条目仍是 deepest。

    这是 C/D 阶段"库缺也要能评"的前提：库缺的是小区名，路名还在库里，
    召回和排序照样能打分，错才不会一股脑记到库头上。
    """
    fields = dict(items()["003"]["fields"])
    fields["community"] = "不存在的公寓"
    g = S.gold_chain(db(), fields)
    assert g.in_db is False
    assert g.missing == ["不存在的公寓"]
    assert g.deepest_name == "不存在的公寓"      # 最深非空层级，不论在不在库
    assert g.deepest is not None and g.deepest.name == "民权路"
    assert g.evaluable is True


def test_gold_chain_short_name_flag_for_two_syllable_deepest():
    # 双楠（武侯区 510107-10）是库里的 2 音节全名，落在"短名"派生轴上
    short = S.gold_chain(db(), {"province": "四川省", "city": "成都市", "district": "武侯区", "road": "双楠"})
    assert short.deepest is not None and short.deepest.adcode == "510107-10"
    assert short.short_name is True
    # 三音节的不算短名
    assert S.gold_chain(db(), items()["003"]["fields"]).short_name is False


def test_chain_matches_accepts_pipeline_top_chain_and_rejects_other_district():
    it = items()["003"]
    gold = S.gold_chain(db(), it["fields"])
    top = run(it["spoken"]).final.ranking.nbest[0]
    assert S.chain_matches(top, gold) is True
    # 另一个区的链不能算真值链（民权路 不在链上）
    other = run(items()["001"]["spoken"]).final.ranking.nbest[0]
    assert S.chain_matches(other, gold) is False


# ==========================================================================
# Stage A  裸 ASR
# ==========================================================================

_A_REF = "天津和平区滨江道一百六十八号劝业场十四楼"
_A_NAMES = ["滨江道", "劝业场"]


def test_stage_a_perfect_transcript_scores_zero():
    r = S.stage_a(_A_REF, _A_REF, _A_NAMES, None)
    for k in ("cer_char", "cer_phon", "cer_name_char", "cer_name_phon", "name_phon_dist_max"):
        assert r[k] == 0.0, f"{k} 应为 0，实得 {r[k]}"
    # 字级本来就没错，"可恢复比例"无意义，必须是 None 而不是 0（0 会被均值统计当成"救不回来"）
    assert r["recoverable_share"] is None


def test_stage_a_homophone_error_is_fully_recoverable():
    """劝业场 → 全叶厂：字全错、音全对，拼音匹配救得回来。"""
    r = S.stage_a(_A_REF.replace("劝业场", "全叶厂"), _A_REF, _A_NAMES, None)
    assert r["cer_name_char"] > 0
    assert r["cer_name_phon"] == 0.0
    assert r["recoverable_share"] == 1.0
    assert r["name_phon_dist_max"] == 0.0


def test_stage_a_unrelated_name_exceeds_max_dist():
    """劝业场 → 雪山湖：音也对不上，超过 0.40 就是声学层的错，后处理够不着。"""
    r = S.stage_a(_A_REF.replace("劝业场", "雪山湖"), _A_REF, _A_NAMES, None)
    assert r["name_phon_dist_max"] > S.MAX_DIST
    assert r["recoverable_share"] == 0.0
    worst = max(r["names"], key=lambda p: p["dist_phon_w"])
    assert worst["name"] == "劝业场"


def test_stage_a_picks_closest_name_variant():
    """口语说"解放碑"不说"解放碑街道"，别名组里取最接近的那个，不能算 ASR 错。"""
    text = items()["003"]["spoken"]
    with_alias = S.stage_a(text, "", [["解放碑街道", "解放碑"], "民权路"], None)
    assert with_alias["name_phon_dist_max"] == 0.0
    assert [p["name"] for p in with_alias["names"]] == ["解放碑", "民权路"]
    # 只给正名时同一句话就要被判成有错——证明上面的 0 是别名带来的
    only_formal = S.stage_a(text, "", [["解放碑街道"], "民权路"], None)
    assert only_formal["name_phon_dist_max"] > 0


# ==========================================================================
# Stage B  归一化与抽取
# ==========================================================================

_B_TRUTH = {
    "ground_truth": "四川省成都市武侯区红牌楼街道二环路南四段30号",
    "fields": {"province": "四川省", "city": "成都市", "district": "武侯区",
               "street": "红牌楼街道", "road": "二环路南四段",
               "house_no": "30号", "building": "", "unit": "", "room": ""},
    "transcript_gold": "我屋头在成都市武侯区红牌楼街道二环路南四段三十号",
}
_B_RAW = _B_TRUTH["transcript_gold"]
_B_TAIL = {"house_no": "30号", "building": "", "unit": "", "room": "", "unverified": ""}


def test_stage_b_all_green_on_correct_output():
    r = S.stage_b(_B_RAW, [("屋头", "家")], _B_TRUTH["ground_truth"], _B_TAIL, _B_TRUTH)
    assert r["lexicon_present"] is True and r["lexicon_hit"] is True
    assert r["number_ok"] is True
    assert r["tail_exact"] is True
    assert r["filler_leak"] == 0.0
    assert r["ok"] is True


def test_stage_b_number_and_tail_mismatch_are_caught():
    bad = dict(_B_TAIL, house_no="31号")
    r = S.stage_b(_B_RAW, [("屋头", "家")], "四川省成都市武侯区红牌楼街道二环路南四段31号", bad, _B_TRUTH)
    assert r["number_ok"] is False
    assert r["tail_fields"]["house_no"] is False
    assert r["tail_exact"] is False
    assert r["ok"] is False


def test_stage_b_dialect_word_leaking_into_output_fails_lexicon_hit():
    """输入里有"屋头"，输出地址里还留着 → 方言归一没做。"""
    leaked = "四川省成都市武侯区屋头红牌楼街道二环路南四段30号"
    r = S.stage_b(_B_RAW, [], leaked, _B_TAIL, _B_TRUTH)
    assert r["lexicon_hit"] is False
    assert r["lexicon_leaked"] == ["屋头"]
    assert r["ok"] is False


_B_ILLEGAL_TRUTH = {
    "ground_truth": "上海市静安区南京西路",
    "fields": {"house_no": "", "building": "", "unit": "", "room": ""},
    "transcript_gold": "喏侬听好上海市静安区南京西路八百一百八号",
}
_B_ILLEGAL_RAW = _B_ILLEGAL_TRUTH["transcript_gold"]


def test_stage_b_illegal_number_preserved_is_caught():
    """"八百一百八号"不是合法数字，原样保留才对。"""
    empty_tail = {"house_no": "", "building": "", "unit": "", "room": "", "unverified": ""}
    r = S.stage_b(_B_ILLEGAL_RAW, [], "上海市静安区南京西路八百一百八号", empty_tail, _B_ILLEGAL_TRUTH)
    assert r["illegal_present"] is True
    assert r["illegal_sequences"] == ["八百一百八号"]
    assert r["illegal_number_caught"] is True


def test_stage_b_illegal_number_guessed_into_digits_is_not_caught():
    """旧解析器把它算成 908号——自信地把门牌编错，这条必须报 False。"""
    guessed = {"house_no": "908号", "building": "", "unit": "", "room": "", "unverified": ""}
    r = S.stage_b(_B_ILLEGAL_RAW, [], "上海市静安区南京西路908号", guessed, _B_ILLEGAL_TRUTH)
    assert r["illegal_present"] is True
    assert r["illegal_number_caught"] is False
    assert r["ok"] is False


def test_stage_b_filler_leak_counts_non_address_chars():
    """"巴适的板"是闲话，进了输出就要按字符比例记漏。"""
    truth = {
        "ground_truth": "四川省成都市武侯区玉林南路",
        "fields": {"house_no": "", "building": "", "unit": "", "room": ""},
        "transcript_gold": "武侯区玉林南路巴适的板",
    }
    tail = {"house_no": "", "building": "", "unit": "", "room": "", "unverified": ""}
    r = S.stage_b(truth["transcript_gold"], [], "四川省成都市武侯区玉林南路巴适的板", tail, truth)
    assert r["filler_leak"] > 0
    assert r["filler_leaked_chars"] == "巴适的板"
    assert r["ok"] is False
    # 不泄漏时归零
    clean = S.stage_b(truth["transcript_gold"], [], truth["ground_truth"], tail, truth)
    assert clean["filler_leak"] == 0.0 and clean["ok"] is True


# ==========================================================================
# Stage C 召回 / Stage D 排序
# ==========================================================================


def test_stage_c_and_d_perfect_on_real_pipeline_items():
    for iid in ("001", "003", "010"):
        it = items()[iid]
        res = run(it["spoken"])
        gold = S.gold_chain(db(), it["fields"])
        c = S.stage_c(res.final.ranking, gold, db(), None)
        d = S.stage_d(res.final.ranking, gold, c)
        assert c["evaluable"] is True, iid
        assert c["recall_hit"] is True, iid
        assert c["recall_chain@1"] is True, iid
        assert c["recall_chain@all"] is True, iid
        assert c["gold_rank"] == 1, iid
        assert d["evaluable"] is True and d["top1_correct"] is True, iid
        assert d["mrr"] == 1.0, iid


def test_stage_c_not_evaluable_when_no_road_level_entry_in_db():
    """010 的真值只有 road=翠湖北路。库里删掉它，真值链就没有最深层条目——
    这条样本不该进 C/D 的分母，否则库缺会被算成召回率下降。"""
    it = items()["010"]
    db_wo = AddressDB([e for e in db().entries if e.name != "翠湖北路"], provider=db().provider)
    gold = S.gold_chain(db_wo, it["fields"])
    assert gold.evaluable is False and gold.deepest is None
    assert gold.missing == ["翠湖北路"] and gold.in_db is False
    c = S.stage_c(run(it["spoken"]).final.ranking, gold, db_wo, None)
    assert c == {"evaluable": False}
    assert S.stage_d(run(it["spoken"]).final.ranking, gold, c) == {"evaluable": False}


def test_stage_c_reports_miss_reason_when_road_is_garbled_beyond_max_dist():
    """真值链在库里（可评），但输入里的路名被糊到 0.40 之外 → 真召回失败，附原因。"""
    it = items()["010"]
    gold = S.gold_chain(db(), it["fields"])
    assert gold.evaluable is True
    res = run("昆明市五华区乌鸦石头二十三号")
    c = S.stage_c(res.final.ranking, gold, db(), None)
    assert c["evaluable"] is True
    assert c["recall_hit"] is False
    assert c["recall_chain@all"] is False
    assert c["gold_rank"] is None
    assert c["miss_reason"] in ("dist_over_max", "weak_filtered", "window_miss")
    assert c["miss_reason"] == "dist_over_max"
    assert "翠湖北" in c["miss_detail"]
    # C 没过，D 不进分母
    assert S.stage_d(res.final.ranking, gold, c) == {"evaluable": False}


# ==========================================================================
# Stage E  闸门重放与四格
# ==========================================================================


def test_replay_reproduces_live_decision_for_every_eval_item():
    """存下来的几个数字必须能原样重放出 rank.decide 的决策，
    否则 risk-coverage 网格扫描扫的是另一套逻辑。"""
    for iid, it in items().items():
        res = run(it["spoken"])
        feats = S.replay_features(res.final.ranking)
        assert S.replay_decision(feats) == res.final.decision, iid


def test_replay_sim_gate_rejects_when_threshold_raised():
    # 真流水线：翠湖播路 音近但不完全，top_sim ≈ 0.945
    feats = S.replay_features(run("昆明市五华区翠湖播路二十三号").final.ranking)
    assert 0.62 < feats["top_sim"] < 0.99
    assert S.replay_decision(feats) == "confident"
    assert S.replay_decision(feats, sim_min=0.99) == "reject"


def test_replay_margin_gate_needs_a_differing_admin():
    """分差闸门只有在 Top-2 指向**另一个行政区**时才拦——同区不同路不算歧义。"""
    diff = S.replay_features(run(items()["003"]["spoken"]).final.ranking)
    assert diff["second_admin_differs"] is True
    assert S.replay_decision(diff, margin_min=1.0) == "ambiguous"
    same = S.replay_features(run(items()["001"]["spoken"]).final.ranking)
    assert same["second_admin_differs"] is False
    assert S.replay_decision(same, margin_min=1.0) == "confident"


def test_replay_decision_on_empty_features_passes_through():
    assert S.replay_decision({"empty": True, "decision": "reject"}) == "reject"


def test_replay_gate_precedence_is_sim_then_deep_hit_then_margin():
    """三道闸门的**先后**要和 rank.decide 一致，不只是各自的阈值。

    顺序换了，网格扫描出来的每个点都还是"合法"的决策，只是把 reject 记成了
    ambiguous——报告里 over_reject 与 risk 会互相串味，而且看不出来。
    """
    tripped = {"empty": False, "top_total": 0.5, "top_sim": 0.50, "second_total": 0.49,
               "second_admin_differs": True, "deep_hit": True, "margin": 0.01}
    # sim 和 margin 两道闸门同时该拦：sim 在前，记 reject
    assert S.replay_decision(tripped) == "reject"
    # sim 过了但没有深层命中：partial 在 margin 之前
    shallow = dict(tripped, top_sim=0.95)
    shallow["deep_hit"] = False
    assert S.replay_decision(shallow) == "partial"
    # 前两道都过，才轮到 margin
    assert S.replay_decision(dict(tripped, top_sim=0.95)) == "ambiguous"


def test_quadrant_four_cells():
    assert S.quadrant("confident", True) == "TP"
    assert S.quadrant("confident", False) == "FA"
    assert S.quadrant("reject", True) == "FR"
    assert S.quadrant("ambiguous", False) == "TR"


_FEATS = {"empty": False, "top_total": 0.90, "top_sim": 0.90,
          "second_total": 0.80, "second_admin_differs": False, "deep_hit": True, "margin": 0.10}


def test_stage_e_fa_caused_by_tail_only_is_not_blamed_on_a_gate():
    """地名链对、门牌错：三道闸门都不看门牌，不能记在 margin/sim 头上。"""
    r = S.stage_e("confident", ok_deliverable=False, ok_geo=True, feats=_FEATS, tail_ok=False)
    assert r["quadrant"] == "FA"
    assert r["quadrant_geo"] == "TP"     # 归因看地名链，门牌不进闸门的账
    assert r["gate_attrib"] == "tail"
    assert r["ok"] is False


def test_stage_e_fr_gate_attrib_follows_the_decision():
    for decision, attrib in (("reject", "sim"), ("ambiguous", "margin"), ("partial", "partial")):
        r = S.stage_e(decision, ok_deliverable=True, ok_geo=True, feats=_FEATS, tail_ok=True)
        assert r["quadrant"] == "FR"
        assert r["gate_attrib"] == attrib, decision
        assert r["ok"] is False


def test_stage_e_ok_only_for_tp_and_tr():
    assert S.stage_e("confident", True, True, _FEATS, True)["ok"] is True     # TP
    assert S.stage_e("reject", False, False, _FEATS, True)["ok"] is True      # TR
    assert S.stage_e("confident", False, True, _FEATS, False)["ok"] is False  # FA
    assert S.stage_e("reject", True, True, _FEATS, True)["ok"] is False       # FR


# ==========================================================================
# Stage F  补全
# ==========================================================================


def test_stage_f_backfill_correct_when_province_and_city_are_inserted():
    """002 只说了"锦江区"，四川省/成都市 是回溯出来的，两级都要对得上。"""
    it = items()["002"]
    res = run(it["spoken"])
    e2e = S.score_e2e(res.fields, res.address, it)
    depth = S.address_depth_of(it["spoken"], it["fields"])
    assert depth == "district"
    f = S.stage_f(res.fields, it, it["spoken"], res.segments, depth, e2e)
    assert f["evaluable"] is True
    assert f["backfilled_levels"] == ["province", "city"]
    assert f["backfill_correct"] is True
    # 回溯出来的层级在 segments 里标成 inserted，与命中层分开报
    assert f["backfill_confidence_gap"] is True
    assert {s["level"] for s in res.segments if s.get("kind") == "inserted"} >= {"province", "city"}


def test_stage_f_wrong_backfilled_province_fails():
    it = items()["002"]
    res = run(it["spoken"])
    e2e = S.score_e2e(res.fields, res.address, it)
    wrong = dict(res.fields, province="云南省")
    f = S.stage_f(wrong, it, it["spoken"], res.segments, "district", e2e)
    assert f["backfill_correct"] is False
    assert f["ok"] is False


def test_stage_f_not_evaluable_when_speaker_gave_the_full_address():
    it = items()["001"]
    res = run(it["spoken"])
    e2e = S.score_e2e(res.fields, res.address, it)
    assert S.address_depth_of(it["spoken"], it["fields"]) == "full"
    assert S.stage_f(res.fields, it, it["spoken"], res.segments, "full", e2e) == {"evaluable": False}


# ==========================================================================
# 端到端
# ==========================================================================


def test_score_e2e_all_green_on_a_correct_answer():
    it = items()["002"]
    res = run(it["spoken"])
    r = S.score_e2e(res.fields, res.address, it)
    assert r["exact"] is True and r["deliverable"] is True and r["geo_ok"] is True
    assert r["admin_all"] is True and r["tail_all"] is True
    assert r["geo_recall"] == 1.0 and r["cer"] == 0.0
    assert r["severity"] == "正确"


def test_score_e2e_tail_only_error_is_minor():
    it = items()["002"]
    res = run(it["spoken"])
    fields = dict(res.fields, house_no="6号附2号")
    r = S.score_e2e(fields, "四川省成都市锦江区春熙路步行街6号附2号", it)
    assert r["exact"] is False and r["deliverable"] is False
    assert r["admin_all"] is True and r["geo_ok"] is True     # 地名链没错
    assert r["tail_all"] is False
    assert r["severity"] == "轻微(门牌以下错)"


def test_score_e2e_admin_error_is_severe():
    it = items()["002"]
    res = run(it["spoken"])
    fields = dict(res.fields, district="武侯区")
    r = S.score_e2e(fields, "四川省成都市武侯区春熙路步行街5号附2号", it)
    assert r["admin_all"] is False
    assert r["geo_ok"] is False
    assert r["severity"] == "严重(行政区错)"


def test_address_depth_of():
    assert S.address_depth_of("成都市武侯区红牌楼街道", items()["001"]["fields"]) == "full"
    assert S.address_depth_of("锦江区春熙路步行街五号附二号", items()["002"]["fields"]) == "district"
    assert S.address_depth_of("重庆渝中区解放碑民权路", items()["003"]["fields"]) == "full"
    assert S.address_depth_of(
        "花牌坊交通巷三十八号",
        {"province": "四川省", "city": "成都市", "district": "金牛区",
         "street": "花牌坊街", "community": "交通巷"},
    ) == "street_only"


# ==========================================================================
# 小样本统计（§2.6 / §9）
# ==========================================================================


def test_wilson_upper_bound_for_zero_errors_in_24():
    lo, hi = ST.wilson(0, 24)
    assert lo == 0.0
    assert 0.13 < hi < 0.14        # 24 条全对也只能说"错误率 ≤ 约 14%"
    assert ST.wilson(0, 0) == (0.0, 1.0)


def test_fmt_rate_hides_percentages_below_the_sample_threshold():
    assert ST.fmt_rate(7, 8) == "7/8"
    assert ST.fmt_rate(0, 0) == "-"
    s = ST.fmt_rate(24, 24)
    assert s.startswith("100.0%") and "[" in s      # n ≥ 20 才给百分比，且必须带区间


def test_mcnemar_without_discordant_pairs_gives_p_one():
    a = [True, False, True, False]
    assert ST.mcnemar(a, a) == {"b01": 0, "b10": 0, "p": 1.0}
    flipped = ST.mcnemar([True, False], [False, True])
    assert flipped["b01"] == 1 and flipped["b10"] == 1 and flipped["p"] == 1.0


def test_bootstrap_diff_of_identical_sequences_is_zero():
    a = [True, False, True, True]
    r = ST.bootstrap_diff(a, a)
    assert r["diff"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0 and r["n"] == 4
    assert ST.bootstrap_diff([], []) == {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}


def test_rule_of_three():
    assert ST.rule_of_three(24) == 0.125
    assert ST.rule_of_three(0) is None


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
