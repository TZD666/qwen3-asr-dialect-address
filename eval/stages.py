"""六个阶段的打分函数（评测体系设计 §3）。全部是纯函数：输入流水线中间产物 + 真值，输出 dict。

    Stage A  裸 ASR         字级 / 音级 CER，地名片段的可恢复比例
    Stage B  归一化与抽取   方言词、口语数字、门牌四字段、非法数串、闲话泄漏
    Stage C  召回           真值链最深层条目是否进了 all_hits / nbest，未召回的原因
    Stage D  排序           Top-1 是否真值链、MRR、排错时五项分数差里谁占主导
    Stage E  闸门与决策     TP / FA / FR / TR 四格 + 是哪道闸门放行/拦截的
    Stage F  补全           回溯出的省市区对不对、路名命中错导致城市错的比例

"真值链"由 fields_gold 反查 AddressDB 得到。库里没有的层级不参与 C/D，这一点
决定了"库缺"和"排错"能分开计——中华中路那种案例是库缺，不是权重的锅。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dialect_addr.address_db import AddressDB, AddressEntry
from dialect_addr.dialect_lexicon import LEXICON
from dialect_addr.normalize import _CN_NUM, _UNITS, cn_to_int, extract_tail
from dialect_addr.pinyin_dialect import syllable_edit_distance
from dialect_addr import rank as rank_mod
from dialect_addr.rank import (
    Chain, RankResult, admin_key, best_window, effective_max_dist, has_deep_hit, rank,
)
from dialect_addr.romanize import resolve_space

ADMIN = ("province", "city", "district")
GEO = ("street", "road", "community")
TAIL = ("house_no", "building", "unit", "room")
MAX_DIST = 0.40
_HAN = re.compile(r"[一-鿿]")
_ILLEGAL_NUM = re.compile(f"({_CN_NUM})({_UNITS})")


def _han(s: str) -> str:
    return "".join(_HAN.findall(s))


def _norm_admin(v: str) -> str:
    return v.replace("市辖区", "市")


# --------------------------------------------------------------------------
# 真值链
# --------------------------------------------------------------------------


@dataclass
class GoldChain:
    admin: dict[str, AddressEntry] = field(default_factory=dict)     # province/city/district → 条目
    deepest: AddressEntry | None = None       # 库里能找到的最深路级条目（street/road/community 之一）
    deepest_field: str = ""                   # deepest 来自哪个真值字段
    road_codes: set[str] = field(default_factory=set)   # 真值里所有能在库里找到的路级条目
    in_db: bool = False                       # §2.1 gold_in_db：最深非空层级（community>road>street）是否在库
    deepest_name: str = ""                    # 最深非空层级的名字（不论在不在库）
    missing: list[str] = field(default_factory=list)    # street/road/community 里库中缺的名字
    short_name: bool = False                  # 派生轴：最深层名称 ≤ 2 音节

    @property
    def evaluable(self) -> bool:
        """能不能参与 C/D 阶段：至少有一个路级条目在库里。"""
        return self.deepest is not None

    def codes(self) -> set[str]:
        s = {e.adcode for e in self.admin.values()}
        s |= self.road_codes
        return s


def lookup_name(db: AddressDB, name: str, levels: tuple[str, ...] | None = None) -> list[AddressEntry]:
    """精确名或别名查找。"""
    if not name:
        return []
    out = []
    for e in db.entries:
        if levels and e.level not in levels:
            continue
        if e.name == name or name in e.aliases:
            out.append(e)
    return out


def _ancestor_codes(db: AddressDB, e: AddressEntry) -> set[str]:
    codes = set()
    cur = e.parent
    guard = 0
    while cur and guard < 8:
        codes.add(cur)
        p = db.by_adcode.get(cur)
        if p is None:
            break
        cur = p.parent
        guard += 1
    return codes


def gold_chain(db: AddressDB, fields: dict[str, str]) -> GoldChain:
    g = GoldChain()
    # 行政区：省 → 市 → 区，逐级用父链约束消歧（全国有多个"鼓楼区"）
    prov = lookup_name(db, fields.get("province", ""), ("province",))
    if prov:
        g.admin["province"] = prov[0]
    city_name = fields.get("city", "")
    cities = lookup_name(db, city_name, ("city",))
    if "province" in g.admin:
        cities = [c for c in cities if c.parent == g.admin["province"].adcode] or cities
    if cities:
        g.admin["city"] = cities[0]
    elif prov and city_name == fields.get("province"):
        # 直辖市：省=市，库里只有省级条目
        g.admin["city"] = prov[0]
    dists = lookup_name(db, fields.get("district", ""), ("district",))
    anchor = g.admin.get("city") or g.admin.get("province")
    if anchor:
        dists = [d for d in dists if d.parent == anchor.adcode or anchor.adcode in _ancestor_codes(db, d)] or dists
    if dists:
        g.admin["district"] = dists[0]

    # 路级：community > road > street，每个字段都查，记录缺的
    anchor_codes = {e.adcode for e in g.admin.values()}
    deepest_name = ""
    for fld in ("community", "road", "street"):
        name = fields.get(fld, "")
        if not name:
            continue
        if not deepest_name:
            deepest_name = name
        cands = lookup_name(db, name, ("street", "road", "poi"))
        if anchor_codes:
            cands = [c for c in cands if _ancestor_codes(db, c) & anchor_codes] or []
        if cands:
            g.road_codes.add(cands[0].adcode)
            if g.deepest is None:
                g.deepest, g.deepest_field = cands[0], fld
        else:
            g.missing.append(name)
    g.deepest_name = deepest_name
    g.in_db = bool(deepest_name) and deepest_name not in g.missing
    if not deepest_name:
        # 真值只到区级（没有任何路级字段）：区在库里就算在库
        g.in_db = "district" in g.admin
    name_for_len = g.deepest.name if g.deepest else deepest_name
    g.short_name = bool(name_for_len) and len(_han(name_for_len)) <= 2
    return g


def chain_matches(chain: Chain, gold: GoldChain) -> bool:
    """候选链是否就是真值链：真值最深路级条目在链上（主路或印证），且行政区不冲突。"""
    if gold.deepest is None:
        return False
    codes = {e.adcode for e in chain.entries.values()} | {x.entry.adcode for x in chain.extra}
    if gold.deepest.adcode not in codes:
        return False
    for lv in ADMIN:
        if lv in gold.admin and lv in chain.entries and chain.entries[lv].adcode != gold.admin[lv].adcode:
            return False
    return True


# --------------------------------------------------------------------------
# Stage A  裸 ASR
# --------------------------------------------------------------------------


def cer(hyp: str, ref: str) -> float:
    """字符错误率 = 编辑距离 / 参考长度。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m] / n


def _syl_cer(hyp_syls, ref_syls, profile=None) -> float:
    """音节级 CER。profile=None 用 0/1 代价（音节 raw 相等即 0），给了 profile 用加权代价。"""
    if not ref_syls:
        return 0.0 if not hyp_syls else 1.0
    if profile is None:
        a = [s.raw for s in ref_syls]
        b = [s.raw for s in hyp_syls]
        n, m = len(a), len(b)
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            cur = [i] + [0] * m
            for j in range(1, m + 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            prev = cur
        return prev[m] / n
    return syllable_edit_distance(hyp_syls, ref_syls, profile) / len(ref_syls)


def _best_substring(hyp: str, name: str) -> tuple[float, str]:
    """hyp 里与 name 字级最接近的等长（±1）窗口：(cer, 片段)。"""
    L = len(name)
    best = (1.0, "")
    for w in (L - 1, L, L + 1):
        if w < 1 or w > len(hyp):
            continue
        for s in range(len(hyp) - w + 1):
            seg = hyp[s:s + w]
            c = cer(seg, name)
            if c < best[0]:
                best = (c, seg)
    return best


def stage_a(hyp_text: str, transcript_gold: str, gold_names: list[str | list[str]], dialect: str | None) -> dict:
    """裸 ASR 质量。transcript_gold 为空时只算地名片段部分（那部分只需要 fields_gold）。

    gold_names 每项可以是一个名字，也可以是一组等价写法（正名 + 别名："解放碑街道"/"解放碑"），
    取最接近的那个——口语里说的常常是别名，不能因为没说"街道"两个字就算 ASR 错。
    """
    space = resolve_space(dialect)
    hyp_h, ref_h = _han(hyp_text), _han(transcript_gold)
    r: dict = {"space": space.name}
    if ref_h:
        r["cer_char"] = round(cer(hyp_h, ref_h), 4)
        hs, rs = space.romanizer(hyp_h), space.romanizer(ref_h)
        if not hs or not rs:      # 该音系罗马化不出来，退回普通话
            m = resolve_space(None)
            hs, rs = m.romanizer(hyp_h), m.romanizer(ref_h)
        r["cer_phon"] = round(_syl_cer(hs, rs), 4)
        r["cer_phon_w"] = round(_syl_cer(hs, rs, space.profile), 4)
    groups = [([g] if isinstance(g, str) else [x for x in g if x]) for g in gold_names]
    groups = [g for g in groups if g]
    if groups:
        hyp_syls = space.romanizer(hyp_h)
        per = []
        for variants in groups:
            best = None
            for nm in variants:
                c_char, seg = _best_substring(hyp_h, nm)
                entry = AddressEntry(name=nm, level="road", adcode="gold")
                bw = best_window(hyp_syls, entry, space, prefilter=False) if hyp_syls else None
                d_phon = bw[0] if bw else 1.0
                # 无权 0/1 音节 CER：同一窗口按音节 raw 相等算
                if bw:
                    seg_syls = hyp_syls[bw[1][0]:bw[1][1]]
                    d_phon01 = _syl_cer(seg_syls, space.romanizer(nm))
                else:
                    d_phon01 = 1.0
                cand = {"name": nm, "seg": seg, "cer_char": round(c_char, 3),
                        "dist_phon_w": round(d_phon, 3), "cer_phon": round(d_phon01, 3)}
                if best is None or (cand["dist_phon_w"], cand["cer_char"]) < (best["dist_phon_w"], best["cer_char"]):
                    best = cand
            per.append(best)
        tot = sum(len(p["name"]) for p in per)
        r["cer_name_char"] = round(sum(p["cer_char"] * len(p["name"]) for p in per) / tot, 4)
        r["cer_name_phon"] = round(sum(p["cer_phon"] * len(p["name"]) for p in per) / tot, 4)
        r["cer_name_phon_w"] = round(sum(p["dist_phon_w"] * len(p["name"]) for p in per) / tot, 4)
        r["name_phon_dist_max"] = round(max(p["dist_phon_w"] for p in per), 3)
        cc = r["cer_name_char"]
        r["recoverable_share"] = round((cc - r["cer_name_phon"]) / cc, 3) if cc > 0 else None
        r["recoverable_share_w"] = round((cc - r["cer_name_phon_w"]) / cc, 3) if cc > 0 else None
        r["names"] = per
    return r


# --------------------------------------------------------------------------
# Stage B  归一化与抽取
# --------------------------------------------------------------------------

_DIALECT_FORMS = sorted({e.dialect for e in LEXICON if e.dialect and len(e.dialect) >= 2}, key=len, reverse=True)


def stage_b(raw_text: str, lex_subs: list, pred_addr: str, pred_fields: dict, truth: dict) -> dict:
    """参照 fields_gold 的 TAIL 四字段和 transcript_gold（文本模式即朗读稿）。"""
    tf = truth["fields"]
    gt = truth["ground_truth"]
    transcript = truth.get("transcript_gold") or truth.get("spoken") or ""
    r: dict = {}
    # 方言词：输入里有词表词形 → 输出地址里不能再出现，且确实做了替换
    present = [w for w in _DIALECT_FORMS if w in raw_text]
    r["lexicon_present"] = bool(present)
    if present:
        leaked = [w for w in present if w in pred_addr or w in pred_fields.get("unverified", "")]
        r["lexicon_hit"] = not leaked and bool(lex_subs)
        r["lexicon_leaked"] = leaked
    else:
        r["lexicon_hit"] = True
    # 口语数字：输出里的数字序列必须和真值一致（和字段归属无关）
    r["number_ok"] = re.findall(r"\d+", pred_addr) == re.findall(r"\d+", gt)
    # 门牌四字段精确匹配
    r["tail_fields"] = {k: pred_fields.get(k, "") == tf.get(k, "") for k in TAIL}
    r["tail_exact"] = all(r["tail_fields"].values())
    # 非法数串："八百一百八号" 必须原样保留、不得猜成数字
    illegal = [m.group(0) for m in _ILLEGAL_NUM.finditer(raw_text) if cn_to_int(m.group(1)) is None]
    r["illegal_present"] = bool(illegal)
    r["illegal_number_caught"] = all(s in pred_addr or s in pred_fields.get("unverified", "") or
                                     not re.search(r"\d+" + re.escape(s[-1]), pred_addr) for s in illegal)
    r["illegal_sequences"] = illegal
    # 闲话泄漏：transcript 里不属于规范地址的字，出现在了输出里
    out_text = pred_addr + pred_fields.get("unverified", "")
    non_addr = set(_han(transcript)) - set(_han(gt))
    leaked_chars = [c for c in _han(out_text) if c in non_addr]
    r["filler_leak"] = round(len(leaked_chars) / max(1, len(_han(out_text))), 3)
    r["filler_leaked_chars"] = "".join(leaked_chars)
    r["ok"] = r["lexicon_hit"] and r["number_ok"] and r["tail_exact"] and r["illegal_number_caught"] and r["filler_leak"] == 0
    return r


# --------------------------------------------------------------------------
# Stage C  召回
# --------------------------------------------------------------------------


def _miss_reason(ranking: RankResult, gold: GoldChain, db: AddressDB, dialect: str | None) -> tuple[str, str]:
    """未召回时的原因。三选一：dist_over_max / weak_filtered / window_miss。"""
    space = resolve_space(dialect)
    han = _han(ranking.geo_text)
    syls = space.romanizer(han)
    e = gold.deepest
    assert e is not None
    hit = next((h for h in ranking.all_hits if h.entry.adcode == e.adcode), None)
    if hit is not None:
        # 命中了但没成链：全弱命中 / 2 字别名撞车 / 无锚定路名
        if hit.weak:
            return "weak_filtered", f"命中「{hit.matched_name}」dist={hit.dist:.3f} 但为弱命中（≤2 音节），未成链"
        return "weak_filtered", f"命中「{hit.matched_name}」dist={hit.dist:.3f} 但链被过滤（无行政区锚定且 dist>0.15）"
    if not syls:
        return "window_miss", "输入无可罗马化汉字"
    bw = best_window(syls, e, space, prefilter=False)
    if bw is None:
        return "window_miss", "条目无法罗马化"
    d, span, nm = bw
    eff, L = effective_max_dist(nm, space, MAX_DIST)
    seg = han[span[0]:span[1]]
    if d > MAX_DIST:
        return "dist_over_max", f"最近窗口「{seg}」vs「{nm}」dist={d:.3f} > {MAX_DIST}"
    if d > eff:
        return "weak_filtered", f"「{seg}」vs「{nm}」dist={d:.3f} > 短名阈值 {eff:.3f}（{L} 音节）"
    return "window_miss", f"「{seg}」vs「{nm}」dist={d:.3f} 在阈值内却未命中（预筛/籍贯排除/同层抑制）"


def stage_c(ranking: RankResult, gold: GoldChain, db: AddressDB, dialect: str | None) -> dict:
    r: dict = {"evaluable": gold.evaluable}
    if not gold.evaluable:
        return r
    e = gold.deepest
    r["recall_hit"] = any(h.entry.adcode == e.adcode for h in ranking.all_hits)
    nb = ranking.nbest
    r["recall_chain@1"] = bool(nb) and chain_matches(nb[0], gold)
    r["recall_chain@3"] = any(chain_matches(c, gold) for c in nb[:3])
    in_nbest = any(chain_matches(c, gold) for c in nb)
    r["gold_rank"] = next((i + 1 for i, c in enumerate(nb) if chain_matches(c, gold)), None)
    if in_nbest:
        r["recall_chain@all"] = True
    else:
        # nbest 只截了前 5；真值链可能在更后面——那是排序问题，不是召回问题
        full = rank(ranking.geo_text, db, dialect, topk=200)
        pos = next((i + 1 for i, c in enumerate(full.nbest) if chain_matches(c, gold)), None)
        r["recall_chain@all"] = pos is not None
        r["gold_rank"] = pos
        if pos is not None:
            r["gold_chain_obj"] = full.nbest[pos - 1]
    if not r["recall_chain@all"]:
        r["miss_reason"], r["miss_detail"] = _miss_reason(ranking, gold, db, dialect)
    r["ok"] = r["recall_chain@all"]
    return r


# --------------------------------------------------------------------------
# Stage D  排序
# --------------------------------------------------------------------------


def _terms(c: Chain) -> dict[str, float]:
    # 权重从模块里现取：评测可能临时改权重做故意破坏测试
    return {"sim": rank_mod.W_SIM * c.sim, "cov": rank_mod.W_COV * c.coverage, "prior": rank_mod.W_PRIOR * c.prior,
            "depth": rank_mod.W_DEPTH * c.depth, "conflict": -rank_mod.W_CONFLICT * c.conflict}


def stage_d(ranking: RankResult, gold: GoldChain, c_res: dict) -> dict:
    """只在 recall_chain@all = true 的子集上算。"""
    r: dict = {"evaluable": bool(c_res.get("recall_chain@all"))}
    if not r["evaluable"]:
        return r
    nb = ranking.nbest
    top = nb[0]
    r["top1_correct"] = chain_matches(top, gold)
    rank_pos = c_res.get("gold_rank")
    r["mrr"] = round(1.0 / rank_pos, 3) if rank_pos else 0.0
    if not r["top1_correct"]:
        g = next((c for c in nb if chain_matches(c, gold)), None) or c_res.get("gold_chain_obj")
        if g is not None:
            tg, tt = _terms(g), _terms(top)
            delta = {k: round(tg[k] - tt[k], 4) for k in tg}
            # 真值链输在哪一项：差值最负的那项（绝对值最大且为负）
            losing = {k: v for k, v in delta.items() if v < 0}
            dom = min(losing, key=lambda k: losing[k]) if losing else max(delta, key=lambda k: abs(delta[k]))
            r["dominant_term"] = dom
            r["delta"] = delta
            r["gold_chain"] = g.full_name()
            r["top_chain"] = top.full_name()
            r["gold_total"], r["top_total"] = round(g.total, 4), round(top.total, 4)
    r["ok"] = r["top1_correct"]
    return r


# --------------------------------------------------------------------------
# Stage E  闸门与决策
# --------------------------------------------------------------------------


def replay_features(ranking: RankResult) -> dict:
    """重放三道闸门所需的量，存进 JSON 以便不重跑检索就能扫阈值。"""
    nb = ranking.nbest
    if not nb:
        return {"empty": True, "decision": ranking.decision}
    top = nb[0]
    ev = list(top.hits.values()) + top.extra
    return {
        "empty": False,
        "top_total": round(top.total, 4), "top_sim": round(top.sim, 4),
        "second_total": round(nb[1].total, 4) if len(nb) > 1 else None,
        "second_admin_differs": (admin_key(nb[1]) != admin_key(top)) if len(nb) > 1 else False,
        "deep_hit": has_deep_hit(top),
        "margin": round(top.total - nb[1].total, 4) if len(nb) > 1 else 1.0,
        "coverage": round(top.coverage, 4),
        "single_hit_dist": round(ev[0].dist, 4) if len(ev) == 1 else None,
    }


def replay_decision(f: dict, margin_min: float | None = None, sim_min: float | None = None) -> str:
    """与 rank.decide 同一套逻辑，但只用存下来的数字。"""
    margin_min = rank_mod.MARGIN_MIN if margin_min is None else margin_min
    sim_min = rank_mod.SIM_MIN if sim_min is None else sim_min
    if f.get("empty"):
        return f.get("decision", "empty")
    if f["top_sim"] < sim_min:
        return "reject"
    shd = f.get("single_hit_dist")
    if shd is not None and shd > 1e-9 and f.get("coverage", 1.0) < rank_mod.SINGLE_HIT_MIN_COV:
        return "reject"
    if not f["deep_hit"]:
        return "partial"
    if f["second_total"] is not None and f["margin"] < margin_min and f["second_admin_differs"]:
        return "ambiguous"
    return "confident"


def quadrant(decision: str, ok: bool) -> str:
    auto = decision == "confident"
    if auto:
        return "TP" if ok else "FA"
    return "FR" if ok else "TR"


def stage_e(decision: str, ok_deliverable: bool, ok_geo: bool, feats: dict, tail_ok: bool) -> dict:
    r: dict = {
        "decision": decision,
        "quadrant": quadrant(decision, ok_deliverable),        # 与损失对齐：错送 = 交付地址错
        "quadrant_geo": quadrant(decision, ok_geo),            # 归因用：闸门只管地名链，不管门牌
    }
    q = r["quadrant"]
    if q == "FA":
        if ok_geo and not tail_ok:
            r["gate_attrib"] = "tail"          # 地名链对、门牌错：三道闸门都不看门牌
        elif ok_geo:
            r["gate_attrib"] = "assembly"      # 链对、门牌对，串还是不对：拼装/闲话泄漏，不是闸门的事
        elif feats.get("empty"):
            r["gate_attrib"] = "sim"
        else:
            # 两道闸门谁离拦下它最近，就记在谁头上
            mm, sm = rank_mod.MARGIN_MIN, rank_mod.SIM_MIN
            m_slack = (feats["margin"] - mm) / mm if feats["second_total"] is not None else 9.0
            s_slack = (feats["top_sim"] - sm) / sm
            if feats["second_total"] is not None and feats["margin"] < mm and not feats["second_admin_differs"]:
                r["gate_attrib"] = "margin(same_admin_exempt)"
            else:
                r["gate_attrib"] = "margin" if m_slack < s_slack else "sim"
    elif q == "FR":
        r["gate_attrib"] = {"reject": "sim", "ambiguous": "margin", "partial": "partial", "empty": "empty"}.get(decision, decision)
    r["ok"] = q in ("TP", "TR")
    return r


# --------------------------------------------------------------------------
# Stage F  补全
# --------------------------------------------------------------------------


def stage_f(pred_fields: dict, truth: dict, spoken: str, segments: list[dict], address_depth: str, e2e: dict) -> dict:
    """只在 address_depth ∈ {district, street_only} 上算。"""
    r: dict = {"evaluable": address_depth in ("district", "street_only")}
    if not r["evaluable"]:
        return r
    tf = truth["fields"]
    backfilled = [lv for lv in ADMIN if tf.get(lv) and _norm_admin(tf[lv]) not in spoken and tf[lv].rstrip("省市区") not in spoken]
    r["backfilled_levels"] = backfilled
    r["backfill_correct"] = all(_norm_admin(pred_fields.get(lv, "")) == _norm_admin(tf.get(lv, "")) for lv in backfilled)
    r["wrong_hit_wrong_city"] = (not e2e["admin_all"]) and e2e["geo_recall"] < 1.0
    inserted = {s.get("level") for s in segments if s.get("kind") == "inserted"}
    if "province" in inserted and tf.get("city") == tf.get("province"):
        inserted.add("city")      # 直辖市：省=市，拼装时只插一次
    r["backfill_confidence_gap"] = all(lv in inserted for lv in backfilled) if backfilled else True
    r["ok"] = r["backfill_correct"]
    return r


# --------------------------------------------------------------------------
# 端到端
# --------------------------------------------------------------------------


def score_e2e(pred_fields: dict, pred_addr: str, truth: dict) -> dict:
    tf = truth["fields"]
    r: dict = {}
    for k in ADMIN:
        r[k] = _norm_admin(pred_fields.get(k, "")) == _norm_admin(tf.get(k, ""))
    geo_names = [tf[k] for k in GEO if tf.get(k)]
    r["geo_recall"] = (sum(1 for g in geo_names if g in pred_addr) / len(geo_names)) if geo_names else 1.0
    for k in TAIL:
        r[k] = pred_fields.get(k, "") == tf.get(k, "")
    r["tail_all"] = all(r[k] for k in TAIL)
    r["admin_all"] = all(r[k] for k in ADMIN)
    r["exact"] = pred_addr == truth["ground_truth"]
    # 投递等价：接 POI 前按精确匹配算（§3 端到端）
    r["deliverable"] = r["exact"]
    # 地名链层面（去掉门牌以下）是否一致——闸门归因用
    r["geo_ok"] = extract_tail(pred_addr).geo_text == extract_tail(truth["ground_truth"]).geo_text
    r["cer"] = round(cer(pred_addr, truth["ground_truth"]), 4)
    if not r["admin_all"]:
        r["severity"] = "严重(行政区错)"
    elif r["geo_recall"] < 1.0 or not r["geo_ok"]:
        r["severity"] = "中等(地名漏/错)"
    elif not r["tail_all"]:
        r["severity"] = "轻微(门牌以下错)"
    else:
        r["severity"] = "正确"
    return r


def address_depth_of(spoken: str, fields: dict[str, str]) -> str:
    """说话人给到哪一层：full（提了省或市）/ district（只提了区）/ street_only。"""
    def said(lv: str) -> bool:
        v = fields.get(lv, "")
        if not v:
            return False
        stem = v.rstrip("省市区县")
        return v in spoken or (len(stem) >= 2 and stem in spoken)
    if said("province") or said("city"):
        return "full"
    if said("district"):
        return "district"
    return "street_only"
