"""端到端流水线：音频 → 方言识别 → 转写 → 归一化 → 检索排序 → 二次解码 → 结构化地址。

    音频
     ↓  Qwen3-ASR 第 1 遍（裸转写，顺带给出方言标签）
    粗文本 + 方言
     ↓  dialect_lexicon   方言词汇 → 标准中文（喺→在、屋头→家）
     ↓  normalize         口语数字 → 阿拉伯数字；剥离门牌/楼栋/单元/室
    地名部分 + 门牌字段
     ↓  rank              在该方言的音系空间里检索地址库、打分、决策
    候选链 Top-K
     ↓  Qwen3-ASR 第 2 遍（把 Top-K 写进 system prompt 重新解码）   ← 可开关
    精文本 → 再走一遍归一化 + 排序
     ↓
    取两遍中决策更强的一方，拼装最终地址

每一步的中间结果都保留在 Result 里——出问题时要能回答"哪一步把它弄错的"。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .address_db import AddressDB
from .asr import ASROutput, Qwen3ASR, normalize_dialect_label
from .dialect_lexicon import normalize as lex_normalize
from .normalize import TailFields, normalize as num_normalize
from .rank import Chain, RankResult, rank

_DECISION_RANK = {"confident": 4, "ambiguous": 3, "partial": 2, "reject": 1, "empty": 0}


@dataclass
class PassResult:
    """一遍解码 + 后处理的全部中间产物。"""

    asr: ASROutput | None
    raw_text: str
    lex_text: str
    lex_subs: list[Any]
    norm_text: str
    tail: TailFields
    ranking: RankResult

    @property
    def decision(self) -> str:
        return self.ranking.decision


@dataclass
class Result:
    audio: str
    dialect: str | None
    pass1: PassResult
    pass2: PassResult | None
    final: PassResult
    chosen: str                       # "pass1" | "pass2"
    address: str                      # 拼装后的规范地址
    fields: dict[str, str]            # 结构化字段
    segments: list[dict] = field(default_factory=list)   # 原文逐段标注：进地址/未核验/闲话/参照/落选
    nbest: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    def summary(self) -> str:
        lines = [
            f"音频: {self.audio}",
            f"方言: {self.dialect or '未识别'}",
            f"第1遍: {self.pass1.raw_text}",
        ]
        if self.pass2:
            lines.append(f"第2遍: {self.pass2.raw_text}   (context_used={self.pass2.asr.context_used if self.pass2.asr else '-'})")
        lines.append(f"采用: {self.chosen}  决策: {self.final.decision}  ({self.final.ranking.reason})")
        lines.append(f"地址: {self.address}")
        if self.final.decision == "ambiguous" and self.nbest:
            lines.append("候选: " + " | ".join(self.nbest[:3]))
        lines.append(f"耗时: {self.elapsed:.1f}s")
        return "\n".join(lines)


# --------------------------------------------------------------------------


def _postprocess(text: str, db: AddressDB, dialect: str | None, asr: ASROutput | None) -> PassResult:
    lex_text, subs = lex_normalize(text)
    norm_text, tail = num_normalize(lex_text)
    ranking = rank(tail.geo_text, db, dialect)
    return PassResult(asr, text, lex_text, subs, norm_text, tail, ranking)


_HAN = re.compile(r"[一-鿿]")
# "X那个Y" / "X旁边Y" / "X附近的Y"：X 是**参照地标**，Y 才是目的地。
# 参照地标不进最终地址（放进 landmark 字段留档），否则会拼出
# "三峡广场磁器口古镇" 这种把路标当地址的结果。
_REF_MARKERS = ("那个", "那边的", "那边", "旁边的", "旁边", "附近的", "附近", "对面的", "对面")


def _split_reference_landmarks(chain: Chain, geo_text: str) -> tuple[list, list]:
    """把链上的路级命中分成 (目的地, 参照地标)。

    判据：命中片段后面**紧跟**参照词，且后面还有另一个路级命中。
    "万象城对面那个院坝头" 里 万象城 后面虽有"对面"，但再往后没有地名，
    所以它仍是目的地——这就是要求"后面还有命中"的原因。
    """
    han = "".join(_HAN.findall(geo_text))
    roads = sorted([h for lv, h in chain.hits.items() if lv in ("road", "street", "poi")] + chain.extra,
                   key=lambda h: h.span[0])
    targets, refs = [], []
    for i, h in enumerate(roads):
        tail_txt = han[h.span[1]: h.span[1] + 3]
        has_marker = any(tail_txt.startswith(m) for m in _REF_MARKERS)
        has_later = any(r.span[0] >= h.span[1] for r in roads[i + 1:])
        # 路名后面的"那个"是口语填充，不是参照："玉林南路那个玉林小区" 是正常的
        # 路+小区链。只有广场/湖/巷子/楼这类**地标**后跟参照词才算参照地标。
        is_road = h.matched_name.endswith(("路", "街", "道", "段"))
        is_ref = has_marker and has_later and not is_road
        (refs if is_ref else targets).append(h)
    return targets, refs


# 只出现在地址开头的口语引导语
_LEAD_FILLER = re.compile(
    r"^(?:我家住在|我家在|我家住|我住在|我家|我在|家在|住在|送到|寄到|送去|寄去|地址是|地址在|地址|就是|就在|在|那个|那边|那儿|那里|这个|这边|这里|是|的)+"
)
# 未命中段里允许删除的口语填充（多字的可以出现在任何位置；单字只删段首段尾——
# "步高里""四公里"这种真地名里的字不能被误伤）
_FILLER_MULTI = ["那边的", "旁边的", "附近的", "对面的", "那栋楼", "那幢楼", "那座楼",
                 "那个", "那边", "那儿", "那里", "这个", "这边", "这里", "旁边", "附近", "对面",
                 "楼上", "楼下", "院子里", "院子", "那栋", "那幢", "那座", "就是", "就在"]
_FILLER_EDGE = "的里头上下在是"
_PUNCT = re.compile(r"[。，,、！!？?；;：:\s「」『』()（）]+")
# 整段只剩这些泛指名词时视为无信息（"那个巷巷头"→"巷子"）；有专名前缀的（宽窄巷子）不受影响
_GENERIC_ONLY = {"巷子", "巷", "院子", "院坝", "楼", "房子", "房", "家", "家里", "屋", "屋里",
                 "那栋", "那幢", "小区", "街", "路", "门口", "门", "对门", "隔壁"}
# 紧跟在命中片段之后的孤立类型后缀（"观音桥"匹配成"观音桥街道"之后剩下的"步行街"）
_GENERIC_SUFFIX = {"街道", "步行街", "古镇", "广场", "大道", "路", "街", "巷", "小区", "花园",
                   "大厦", "商场", "中心", "镇", "村", "新区", "区", "市", "省", "机场", "火车站", "站"}


def _clean_free(seg: str, is_lead: bool, after_hit: bool = False) -> str:
    """清理一段未被地址库命中的文本：去标点、去口语填充，剩下的原样保留。"""
    s = _PUNCT.sub("", seg)
    if is_lead:
        s = _LEAD_FILLER.sub("", s)
    changed = True
    while changed and s:
        changed = False
        for w in _FILLER_MULTI:
            if w in s:
                s = s.replace(w, "")
                changed = True
        while s and s[0] in _FILLER_EDGE:
            s = s[1:]
            changed = True
        while s and s[-1] in _FILLER_EDGE:
            s = s[:-1]
            changed = True
    if s in _GENERIC_ONLY:
        return ""
    if after_hit and s in _GENERIC_SUFFIX:
        return ""
    return s


def _assemble(
    chain: Chain | None, tail: TailFields, fallback_geo: str, norm_text: str = "",
    all_hits: list | None = None,
) -> tuple[str, dict[str, str]]:
    """候选链 + 门牌字段 → 规范地址字符串 + 字段字典。

    以**原文为锚**重建，而不是从候选链"生成"：
      * 命中的片段 → 换成地址库的规范名（或用户说的更长别名）
      * 参照地标（"三峡广场那个磁器口"里的三峡广场）→ 连同后面的参照词一起删
      * 门牌/楼栋/单元/室 → 原位保留
      * **没命中的地名段 → 原样保留**，并记入 unverified 字段
      * 链上有但原文没说的上级行政区（只说"成都"没说"四川省"）→ 补插到正确位置
      * 段首的"我家在/送到"、段内的"那个/旁边"等口语填充 → 删

    第一版是"从链生成"：只输出命中的层级，没命中的一律丢——用户说了
    "花牌坊交通巷碧园公寓"，库里没有，输出就成了"成都市38号"。
    宁可把没核验的原文交出去并标明，也不能悄悄吞掉。
    """
    t = tail.as_dict()
    if not norm_text:
        norm_text = fallback_geo
    n = len(norm_text)

    # ---- 字符分类：tail 字段 / 命中片段 / 自由文本 ----
    tail_spans = sorted((s, e) for s, e, _ in tail.spans)
    kind = ["free"] * n
    owner = [-1] * n
    for s, e in tail_spans:
        for i in range(s, min(e, n)):
            kind[i] = "tail"

    # 排序层的 span 是"纯汉字且不含 tail 字段"的索引空间，映射回 norm_text 索引
    han_pos = [i for i, ch in enumerate(norm_text) if _HAN.match(ch) and kind[i] != "tail"]

    edits: list[tuple[int, int, str, str]] = []   # (起, 止, 替换文本, 层级)
    landmark_names: list[str] = []
    demoted_admin: set[str] = set()                # 从原文位置删除、改为前插的行政区层级
    ADMIN = ("province", "city", "district")
    if chain is not None:
        targets, refs = _split_reference_landmarks(chain, fallback_geo)
        ref_ids = {id(h) for h in refs}
        chain_hits = list(chain.hits.items()) + [(h.entry.level, h) for h in chain.extra]
        first_road_pos = min(
            (h.span[0] for lv, h in chain_hits if lv not in ADMIN), default=None
        )
        for lv, h in chain_hits:
            s, e = h.span
            if s >= e or e > len(han_pos):
                continue
            ns, ne = han_pos[s], han_pos[e - 1] + 1
            if id(h) in ref_ids:
                ext = next((len(m) for m in _REF_MARKERS if norm_text[ne:ne + 4].startswith(m)), 0)
                edits.append((ns, ne + ext, "", "ref"))
                landmark_names.append(h.matched_name)
                continue
            e_ = h.entry
            disp = h.matched_name if (len(h.matched_name) > len(e_.name) and e_.name in h.matched_name) else e_.name
            # 行政区名出现在路名**之后**（"航空路1号成都双流机场"里的"成都"）：
            # 它是后面专名的一部分，不是在报行政区。原位删掉，行政区改为按层级前插。
            if lv in ADMIN and first_road_pos is not None and s > first_road_pos:
                edits.append((ns, ne, "", "demoted"))
                demoted_admin.add(lv)
                continue
            edits.append((ns, ne, disp, lv))

        # 落选的竞争行政区（"官渡区呈贡新区"里被排序层否掉的官渡区）：
        # 它在库里，不是"未核验"，是"已判定不采用"——从原文删除，不能当成未知地名放行。
        chain_codes = {x.adcode for x in chain.entries.values()}
        chain_levels = {lv for lv in ADMIN if lv in chain.hits}
        taken = [(ns, ne) for ns, ne, _, _ in edits]
        for h in (all_hits or []):
            # 只删"和本链同一层级正面竞争、且本链在该层级有真实命中"的强命中。
            # 弱命中（2 音节撞县名）不删；本链该层级没命中的也不删——那不是竞争，是本链不知道。
            if h.entry.level not in ADMIN or h.entry.adcode in chain_codes or h.dist > 0.15:
                continue
            if h.weak or h.entry.level not in chain_levels:
                continue
            s, e = h.span
            if s >= e or e > len(han_pos):
                continue
            ns, ne = han_pos[s], han_pos[e - 1] + 1
            if any(not (ne <= a or ns >= b) for a, b in taken):
                continue
            edits.append((ns, ne, "", "rejected"))
            taken.append((ns, ne))
        edits.sort()
        for k, (ns, ne, _, _) in enumerate(edits):
            for i in range(ns, min(ne, n)):
                if kind[i] != "tail":
                    kind[i] = "hit"
                    owner[i] = k

    # ---- 按顺序切成片段 ----
    parts: list[tuple[str, str, str, str]] = []   # (kind, text, level, 原文)
    i = 0
    while i < n:
        if kind[i] == "hit":
            k = owner[i]
            ns, ne, rep, lv = edits[k]
            parts.append(("hit", rep, lv, norm_text[ns:ne]))
            i = ne
        elif kind[i] == "tail":
            s, e = next((s, e) for s, e in tail_spans if s <= i < e)
            parts.append(("tail", norm_text[s:e], "tail", norm_text[s:e]))
            i = e
        else:
            j = i
            while j < n and kind[j] == "free":
                j += 1
            parts.append(("free", norm_text[i:j], "free", norm_text[i:j]))
            i = j

    # ---- 清理自由段；收集未核验地名；同时给每一段打标签供界面着色 ----
    #   hit        进了地址（规范名）
    #   tail       门牌/楼栋/单元/室
    #   unverified 库里没有、按原文保留的地名段（橙）
    #   filler     非地址的话（引导语/口语填充/结尾闲话），不进地址（灰）
    #   ref        参照地标，不进地址（灰）
    #   dropped    落选的竞争行政区 / 位置错乱的行政区名，不进地址（灰）
    unverified: list[str] = []
    cleaned: list[tuple[str, str, str]] = []
    segments: list[dict] = []
    has_hit = any(k == "hit" and txt for k, txt, _, _ in parts)
    hit_indices = [idx for idx, (k, txt, _, _) in enumerate(parts) if k == "hit" and txt]
    first_hit_idx = hit_indices[0] if hit_indices else -1
    last_content_idx = max(
        (idx for idx, (k, txt, _, _) in enumerate(parts) if (k == "hit" and txt) or k == "tail"), default=-1
    )
    first_content_seen = False
    prev_kind = ""
    seg_of_cleaned: list[int] = []     # cleaned[i] 对应 segments 里的下标，补插时用来定位
    for idx, (k, txt, lv, orig) in enumerate(parts):
        if k == "hit" and not txt:
            segments.append({"kind": "ref" if lv == "ref" else "dropped", "text": orig, "level": lv})
            continue
        if k == "free":
            # 位置规则：有行政区/路命中时，第一个命中之前的自由文本一律是引导语——
            # 地址从大到小说，省市前面不会有真地名，只会有"那我来句武汉腔""地址是"这种话。
            if has_hit and idx < first_hit_idx:
                segments.append({"kind": "filler", "text": orig})
                continue
            cleaned_txt = _clean_free(txt, is_lead=not first_content_seen, after_hit=(prev_kind == "hit"))
            # 门牌以下都说完之后的自由文本，没有地名特征字的当结尾闲话
            if cleaned_txt and idx > last_content_idx and last_content_idx >= 0 and not re.search(
                r"[路街巷道村镇号栋幢座楼苑园区厦场城店馆院寓库站港口湾桥门]", cleaned_txt
            ):
                cleaned_txt = ""
            if not cleaned_txt:
                segments.append({"kind": "filler", "text": orig})
                continue
            unverified.append(cleaned_txt)
            segments.append({"kind": "unverified", "text": cleaned_txt, "orig": orig})
            txt = cleaned_txt
        else:
            segments.append({"kind": k, "text": txt, "orig": orig, "level": lv})
        first_content_seen = True
        prev_kind = k
        cleaned.append((k, txt, lv))
        seg_of_cleaned.append(len(segments) - 1)

    # ---- 补插原文没说（或被降级删除）的上级行政区 ----
    if chain is not None:
        rank_of = {lv: i for i, lv in enumerate(("province", "city", "district", "street", "road", "poi"))}
        present = {lv for lv in ADMIN if lv in chain.hits} - demoted_admin
        present_texts = {txt for k, txt, _ in cleaned if k == "hit"}
        for lv in ADMIN:
            if lv not in chain.entries or lv in present:
                continue
            disp = chain.display_name(lv)
            # 直辖市省=市：同名只出现一次，不管是哪一级先落在文本里
            if disp in present_texts:
                continue
            pos = next(
                (idx for idx, (k, _, l) in enumerate(cleaned) if k == "hit" and rank_of.get(l, 9) > rank_of[lv]),
                None,
            )
            if pos is None:
                pos = next((idx for idx, (k, _, _) in enumerate(cleaned) if k in ("hit", "tail")), 0)
            cleaned.insert(pos, ("hit", disp, lv))
            present_texts.add(disp)
            # 标注行里也插到同一位置，让"湖北省"出现在"武汉市"前面而不是末尾
            seg_pos = seg_of_cleaned[pos] if pos < len(seg_of_cleaned) else len(segments)
            segments.insert(seg_pos, {"kind": "inserted", "text": disp, "level": lv})
            seg_of_cleaned = [i + 1 if i >= seg_pos else i for i in seg_of_cleaned]
            seg_of_cleaned.insert(pos, seg_pos)

    address = "".join(txt for _, txt, _ in cleaned)

    # ---- 字段 ----
    if chain is not None:
        f = chain.fields()
        road_parts = [txt for k, txt, lv in cleaned if k == "hit" and lv in ("road", "street", "poi")]
        f["road"] = road_parts[0] if road_parts else ""
        community = "".join(road_parts[1:])
    else:
        f = {lv: "" for lv in ("province", "city", "district", "street", "road")}
        community = ""
    fields = {
        **f, "community": community, "landmark": "、".join(landmark_names),
        "unverified": "、".join(unverified), **t,
    }
    return address, fields, segments


def _better(a: PassResult, b: PassResult) -> bool:
    """b 是否比 a 更好：决策等级优先，同级比 Top-1 总分。"""
    ra, rb = _DECISION_RANK[a.decision], _DECISION_RANK[b.decision]
    if rb != ra:
        return rb > ra
    ta = a.ranking.top.total if a.ranking.top else 0.0
    tb = b.ranking.top.total if b.ranking.top else 0.0
    return tb > ta + 1e-6


class Pipeline:
    def __init__(
        self,
        db: AddressDB | None = None,
        asr: Qwen3ASR | None = None,
        two_pass: bool = True,
        context_topk: int = 8,
    ):
        self.db = db or AddressDB.default()
        self.asr = asr or Qwen3ASR()
        self.two_pass = two_pass
        self.context_topk = context_topk

    # ------------------------------------------------------------------
    def process_text(self, text: str, dialect: str | None = None) -> Result:
        """纯文本入口：跳过 ASR，用于离线测试和演示。"""
        t0 = time.time()
        p1 = _postprocess(text, self.db, dialect, None)
        address, fields, segs = _assemble(p1.ranking.top, p1.tail, p1.tail.geo_text, p1.norm_text, p1.ranking.all_hits)
        return Result(
            audio="(text)", dialect=dialect, pass1=p1, pass2=None, final=p1, chosen="pass1",
            address=address, fields=fields, segments=segs,
            nbest=[c.full_name() for c in p1.ranking.nbest], elapsed=time.time() - t0,
        )

    def process(self, audio: str | Path, dialect_hint: str | None = None, language: str | None = None) -> Result:
        t0 = time.time()
        audio = str(audio)

        # ---- 第 1 遍：裸转写 ----
        a1 = self.asr.transcribe(audio, language=language)
        dialect = normalize_dialect_label(a1.language) or dialect_hint
        p1 = _postprocess(a1.text, self.db, dialect, a1)

        # ---- 第 2 遍：上下文注入 ----
        p2: PassResult | None = None
        if self.two_pass and p1.ranking.nbest:
            ctx = [c.full_name() for c in p1.ranking.nbest[:self.context_topk]]
            # 把印证地标也放进去，模型看到"红牌楼街道"和"二环路南四段"都在候选里
            for c in p1.ranking.nbest[:self.context_topk]:
                for x in c.extra:
                    ctx.append(self.db.full_name(x.entry))
            a2 = self.asr.transcribe(audio, language=language, context=ctx)
            if a2.context_used:
                p2 = _postprocess(a2.text, self.db, dialect, a2)

        # ---- 择优 ----
        if p2 is not None and _better(p1, p2):
            final, chosen = p2, "pass2"
        else:
            final, chosen = p1, "pass1"

        # partial（只命中省/市）也要用候选链：省市要规范化、要补插，只是其余段标未核验。
        # 第一版漏了 partial，结果连命中的"河南省"都没用上，整句原样吐了出去。
        top = final.ranking.top if final.decision in ("confident", "ambiguous", "partial") else None
        address, fields, segs = _assemble(top, final.tail, final.tail.geo_text, final.norm_text, final.ranking.all_hits)
        return Result(
            audio=audio, dialect=dialect, pass1=p1, pass2=p2, final=final, chosen=chosen,
            address=address, fields=fields, segments=segs,
            nbest=[c.full_name() for c in final.ranking.nbest], elapsed=time.time() - t0,
        )
