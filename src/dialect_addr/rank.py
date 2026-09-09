"""候选地址打分排序 + 兜底决策。回答"识别不准时，怎么确定概率最大的候选"。

理论骨架
--------
要的不是 ASR 在做的 argmax P(文字|音频)，而是 argmax P(真实地址 A|音频 x)：

    P(A|x) ∝ P(x|A) · P(A)
              ↑声学似然   ↑地址先验（ASR 完全没有这一项）

ASR 错，不是听错了，是它的语言模型先验是"通用中文"——里面"五后去"
这种常见字组合的概率天然高于"武侯区"这个低频专名。地名是长尾，
通用先验在这里是负作用。解法不是修 ASR，是把 P(A) 补进来。

工程落地
--------
对每条候选地址链 A 打分：

    score(A) = w_sim  · 音相似度        ← P(x|A) 的代理：在该方言的音系空间里算
             + w_cov  · 文本覆盖率      ← 候选能解释多少输入文本
             + w_prior· 地址先验        ← P(A)：POI 热度 / 用户定位 / 历史频次
             + w_depth· 层级深度        ← 匹配到路级比只匹配到市级可信
             - w_conf · 层级冲突惩罚    ← 文本里有强命中的**另一个**区 → 扣分

其中层级合法性是**硬约束**而非软分：候选链一律由"路→区→市→省"回溯父链
生成，天然合法，"成都市+江北区"这种链根本不会被构造出来。

兜底（防"自信地答错"）
----------------------
最危险的失败模式是幻觉性纠正：把说对的地址"纠正"成库里那个相似但错的。
三道确定性阈值：
    1. 分差阈值   Top-1 与 Top-2 太接近 → 不自动决策，吐 N-best
    2. 相似度阈值 Top-1 的音相似度太低 → 不采信，保留原文
    3. 门牌绝不猜 门牌/单元/室号不经过本模块，由 normalize.py 纯规则抽取
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .address_db import LEVEL_ORDER, AddressDB, AddressEntry, Level
from .pinyin_dialect import Syllable, syllable_edit_distance
from .romanize import PhonSpace, family_of, resolve_space

_HAN = re.compile(r"[一-鿿]")
_DEMONYM_SUFFIX = set("话腔人佬妹仔音调味菜")   # 河南话/武汉腔/四川人/湘菜/川味

# 每个层级的"特异性权重"：越具体的层级，命中越有说服力
_LEVEL_W: dict[str, float] = {
    "province": 0.6, "city": 0.8, "district": 1.0, "street": 1.1, "road": 1.2, "poi": 1.2,
}

# 打分权重。总和 1，便于把 total 当成 0~1 的置信度读。
W_SIM, W_COV, W_PRIOR, W_DEPTH = 0.50, 0.20, 0.15, 0.15
W_CONFLICT = 0.25          # 冲突惩罚（减项）

# 决策阈值
MARGIN_MIN = 0.06          # Top1-Top2 分差低于此 → ambiguous
SIM_MIN = 0.62             # Top1 音相似度低于此 → reject
STRONG_HIT = 0.15          # 距离低于此视为"强命中"，用于冲突检测


@dataclass
class Hit:
    entry: AddressEntry
    dist: float                 # 归一化音距离 0~1
    span: tuple[int, int]       # 在纯汉字串里的 [起, 止)
    matched_name: str           # 命中的是正名还是哪个别名
    # 弱命中：命中的名字只有 ≤2 个音节。全国 2800 个区县里任何两个音节都能撞到一个
    # （"中山路"→中山市、"广场"→广昌县、"南话"→南华县），所以弱命中**不能单独成链**、
    # 不能当冲突去扣别人的分、不能触发"落选删除"——只有同链上有强命中印证时才算数。
    weak: bool = False

    @property
    def sim(self) -> float:
        return 1.0 - self.dist


@dataclass
class Chain:
    """一条候选地址链：从省到路的一串条目，以及支撑它的文本命中。"""

    entries: dict[str, AddressEntry]          # level -> entry
    hits: dict[str, Hit]                      # level -> 文本命中（可能缺某些层）
    # 同一个区下额外命中的路/地标（"解放碑" + "民权路"）。它们是**互相印证**
    # 的证据，参与相似度/覆盖率/深度计分，但字段里只报主路（hits["road"]）。
    extra: list[Hit] = field(default_factory=list)
    sim: float = 0.0
    coverage: float = 0.0
    prior: float = 0.0
    depth: float = 0.0
    conflict: float = 0.0
    total: float = 0.0

    @property
    def deepest(self) -> AddressEntry:
        for lv in reversed(LEVEL_ORDER):
            if lv in self.entries:
                return self.entries[lv]
        raise ValueError("empty chain")

    def display_name(self, lv: str) -> str:
        """某层级的输出名：命中的别名比正名更长且包含正名时，用别名。

        "春熙路步行街" 命中了条目 "春熙路" 的别名——用户说的是全称，
        输出就该是全称；反过来 "武侯" 命中 "武侯区"，输出要补全成正名。
        """
        e = self.entries[lv]
        h = self.hits.get(lv)
        if h and len(h.matched_name) > len(e.name) and e.name in h.matched_name:
            return h.matched_name
        return e.name

    def full_name(self) -> str:
        parts: list[str] = []
        for lv in LEVEL_ORDER:
            if lv in self.entries:
                nm = self.display_name(lv)
                if parts and parts[-1] == nm:   # 直辖市：省=市，只显示一次
                    continue
                parts.append(nm)
        return "".join(parts)

    def fields(self) -> dict[str, str]:
        f = {lv: "" for lv in ("province", "city", "district", "street", "road")}
        for lv in self.entries:
            if lv in f:
                f[lv] = self.display_name(lv)
        # 直辖市：省=市
        if f["province"] and not f["city"] and f["province"].endswith("市"):
            f["city"] = f["province"]
        return f


@dataclass
class RankResult:
    decision: str                        # confident | ambiguous | reject | empty
    top: Chain | None
    nbest: list[Chain]
    geo_text: str
    dialect: str | None
    space_name: str
    reason: str = ""
    all_hits: list[Hit] = field(default_factory=list)


# --------------------------------------------------------------------------
# 文本内定位：在输入里找每个库条目最像的片段
# --------------------------------------------------------------------------


def _han_only(text: str) -> str:
    return "".join(_HAN.findall(text))


def find_hits(
    han: str,
    syls: tuple[Syllable, ...],
    entries: list[AddressEntry],
    space: PhonSpace,
    max_dist: float = 0.40,
) -> list[Hit]:
    """滑窗对齐：对每个条目，在输入音节序列上找距离最小的窗口。

    窗口宽度取条目音节数 ±1，容忍口语多说/少说一个字。
    这一步不需要事先分词——分词本身在方言误识文本上就不可靠。
    """
    n = len(syls)
    if n == 0:
        return []
    # 预筛：候选名至少要和输入共享一个完全相同的音节，否则不可能在 0.4 距离内。
    # 3500 条全国表逐条滑窗要 1.2s，预筛后大部分条目一次集合查询就跳过。
    text_raw = {s.raw for s in syls}
    hits: list[Hit] = []
    for e in entries:
        best: tuple[float, tuple[int, int], str] | None = None
        for nm in e.all_names():
            esyl = space.romanizer(nm)
            L = len(esyl)
            if L == 0:
                continue
            if not any(s.raw in text_raw for s in esyl):
                continue
            for w in (L - 1, L, L + 1):
                if w < 1 or w > n:
                    continue
                # 太短的别名（单字）不允许 ±1 宽度，否则到处乱命中
                if L <= 1 and w != L:
                    continue
                for s in range(0, n - w + 1):
                    d = syllable_edit_distance(syls[s:s + w], esyl, space.profile) / max(w, L)
                    # 距离相同时偏向**更长的名字**："玉林小区" 比别名 "玉林" 更具体，
                    # 且不会和相邻的 "玉林南路" 抢同一片文本。
                    if best is None or d < best[0] - 1e-9 or (
                        abs(d - best[0]) < 1e-9 and len(nm) > len(best[2])
                    ):
                        best = (d, (s, s + w), nm)
        if best is None:
            continue
        # 短名字更容易撞车：两音节的名字在任意文本里找到距离 0.33 的窗口太容易了。
        # 名字越短，要求的距离越严。三音节及以上用满额阈值。
        L_hit = len(space.romanizer(best[2]))
        eff_max = max_dist * min(1.0, L_hit / 3.0)
        if best[0] > eff_max:
            continue
        # "河南话""武汉腔""四川人"：后面紧跟 话/腔/人 的省市名是在说方言或籍贯，不是在报地址。
        # 只对**别名**形式（没带 省/市/区 后缀）做这个排除——"卧龙区人民路"里的
        # 卧龙区后面跟"人"，那是下一个词的开头，全名不受此规则影响。
        if e.level in ("province", "city", "district") and best[2] != e.name:
            nxt = han[best[1][1]] if best[1][1] < n else ""
            if nxt in _DEMONYM_SUFFIX:
                continue
        hits.append(Hit(e, best[0], best[1], best[2], weak=(L_hit <= 2)))
    hits.sort(key=lambda h: (h.dist, -h.entry.prior))

    # 非极大值抑制：同层级、片段重叠、且明显更弱的命中直接丢弃。
    # 否则一个 0.39 的"玉林街道"会以 0.00 的"红牌楼街道"同一片文本为根
    # 另起一条链，再吸收旁边的强证据，把分数抬到和正确答案接近。
    kept: list[Hit] = []
    for h in hits:
        suppressed = any(
            k.entry.level == h.entry.level
            and _spans_overlap(k.span, h.span)
            and h.dist - k.dist > 0.15
            for k in kept
        )
        if not suppressed:
            kept.append(h)
    return kept


# --------------------------------------------------------------------------
# 链构造：自底向上回溯父链
# --------------------------------------------------------------------------


def _ancestors(db: AddressDB, e: AddressEntry) -> dict[str, AddressEntry]:
    chain = {e.level: e}
    cur = e.parent
    guard = 0
    while cur and guard < 8:
        p = db.by_adcode.get(cur)
        if p is None:
            break
        chain[p.level] = p
        cur = p.parent
        guard += 1
    return chain


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def build_chains(db: AddressDB, hits: list[Hit], n_chars: int) -> list[Chain]:
    """从每个命中出发回溯父链，再把父链上其它层级的命中挂上来。"""
    by_adcode: dict[str, Hit] = {}
    for h in hits:
        # 同一条目可能通过不同别名命中多次，留距离最小的
        if h.entry.adcode not in by_adcode or h.dist < by_adcode[h.entry.adcode].dist:
            by_adcode[h.entry.adcode] = h

    chains: dict[str, Chain] = {}
    for h in by_adcode.values():
        entries = _ancestors(db, h.entry)
        ch = Chain(entries=entries, hits={h.entry.level: h})
        # 把父链上有命中的层级也挂上（要求 span 不与已有命中重叠）
        for lv, e in entries.items():
            if lv == h.entry.level:
                continue
            ph = by_adcode.get(e.adcode)
            if ph and not any(_spans_overlap(ph.span, x.span) for x in ch.hits.values()):
                ch.hits[lv] = ph

        # 路级：吸收同一父节点下其它不重叠的路/地标命中作为印证。
        # "解放碑 + 民权路" 两个都在渝中区下，应合成一条链而不是两条互相竞争。
        if h.entry.level in ("road", "street", "poi"):
            taken = [x.span for x in ch.hits.values()]
            for sib in sorted(by_adcode.values(), key=lambda x: x.dist):
                if sib.entry.adcode == h.entry.adcode or sib.entry.level not in ("road", "street", "poi"):
                    continue
                if sib.entry.parent != h.entry.parent:
                    continue
                if any(_spans_overlap(sib.span, s) for s in taken):
                    continue
                ch.extra.append(sib)
                taken.append(sib.span)
            # 主路取文本里最靠前的那个——地址是从大到小说的，先说的是主干
            all_roads = [h] + ch.extra
            all_roads.sort(key=lambda x: x.span[0])
            ch.hits[h.entry.level] = all_roads[0]
            ch.entries[h.entry.level] = all_roads[0].entry
            ch.extra = all_roads[1:]

        # 去重键：最深行政区 + 全部路级条目集合。
        # 从雨花路出发和从万科城出发吸收完彼此后是同一条链，只留一份。
        # 全是弱命中的链不成立：两个音节撞上某个县名不是证据
        ch_hits = list(ch.hits.values()) + ch.extra
        if all(x.weak for x in ch_hits):
            continue
        # 最深的那个命中如果是靠 2 字**别名**撞上的，链也不成立。
        # 父节点不能替它担保：南阳市下面 12 个县，随便哪个县的 2 字别名撞上文本，
        # 南阳市这个强命中都会"印证"它——"阳河"→唐河县就是这么来的。
        # 2 字**全名**（双楠、南坪）不在此列：它们是完整地名，有上级强命中就够。
        deepest = max(ch_hits, key=lambda x: LEVEL_ORDER.index(x.entry.level))
        if deepest.weak and deepest.matched_name != deepest.entry.name:
            continue
        # 没有任何行政区命中（省/市/区都是回溯出来的）的孤立路名，必须近乎精确。
        # 全国重名路太多（人民路/中山路/建设路），没有城市锚定时靠 ±1 宽度容错
        # 匹配到的路名是在猜（"人民路"→民权路 就是这么来的）。
        if not any(lv in ch.hits for lv in ("province", "city", "district")):
            if all(x.dist > STRONG_HIT for x in ch_hits):
                continue

        admin = tuple(ch.entries[lv].adcode for lv in ("province", "city", "district") if lv in ch.entries)
        roadset = frozenset(x.entry.adcode for x in [ch.hits.get(h.entry.level)] + ch.extra if x)
        key = (admin, roadset if h.entry.level in ("road", "street", "poi") else frozenset())
        if key not in chains or len(ch.hits) + len(ch.extra) > len(chains[key].hits) + len(chains[key].extra):
            chains[key] = ch

    # 去重：若链 A 的条目集合是链 B 的真子集（B 更深且包含 A 的全部命中），丢 A
    keys = list(chains)
    keep: list[Chain] = []
    for k in keys:
        a = chains[k]
        dominated = False
        for k2 in keys:
            if k2 == k:
                continue
            b = chains[k2]
            a_codes = {e.adcode for e in a.entries.values()}
            b_codes = {e.adcode for e in b.entries.values()}
            if a_codes < b_codes and set(a.hits) <= set(b.hits):
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return keep


# --------------------------------------------------------------------------
# 打分
# --------------------------------------------------------------------------


def score_chain(ch: Chain, all_hits: list[Hit], n_chars: int) -> None:
    evidence = list(ch.hits.items()) + [(x.entry.level, x) for x in ch.extra]

    # 音相似度：各层命中的相似度按特异性加权平均
    ws = [(_LEVEL_W[lv], h.sim) for lv, h in evidence]
    ch.sim = sum(w * s for w, s in ws) / sum(w for w, _ in ws) if ws else 0.0

    # 覆盖率：命中片段覆盖了输入里多少字
    covered = set()
    for _, h in evidence:
        covered.update(range(*h.span))
    ch.coverage = len(covered) / n_chars if n_chars else 0.0

    # 先验：链上各条目的先验取几何平均意味的加权——这里用简单平均，
    # 但越深的层级权重越高（路级热度比省级热度更有区分度）
    pw = [(_LEVEL_W[lv], e.prior) for lv, e in ch.entries.items()]
    ch.prior = sum(w * p for w, p in pw) / sum(w for w, _ in pw) if pw else 0.0

    # 深度：命中了几处证据（不是链有几层——父链是回溯出来的，不算证据）
    ch.depth = min(len(ch.hits) + len(ch.extra), 3) / 3.0

    # 冲突：输入里有"强命中"的**另一个行政区**（省/市/区级），且不在本链上 → 惩罚
    # 例：文本同时强命中 官渡区 和 呈贡区，选了呈贡的链要为官渡的存在付出代价，
    # 但如果呈贡链有路级证据支撑（雨花路/万科城），总分仍能胜出。
    #
    # 只看行政区层级，**不看路级**：一句地址里同时出现"解放碑"和"民权路"
    # 是互相印证（同一个区下的两个地标），不是冲突。第一版把这当冲突，
    # 直接把正确答案从 Top-1 打到了 Top-3。
    conflict = 0.0
    own = {e.adcode for e in ch.entries.values()}
    own_spans = [h.span for h in ch.hits.values()] + [h.span for h in ch.extra]
    for h in all_hits:
        if h.entry.adcode in own or h.dist > STRONG_HIT or h.weak:
            continue
        if h.entry.level not in ("province", "city", "district"):
            continue
        if h.entry.level in ch.entries and not any(_spans_overlap(h.span, s) for s in own_spans):
            conflict = max(conflict, h.sim * _LEVEL_W[h.entry.level] / 1.2)
    ch.conflict = conflict

    ch.total = (
        W_SIM * ch.sim
        + W_COV * ch.coverage
        + W_PRIOR * ch.prior
        + W_DEPTH * ch.depth
        - W_CONFLICT * ch.conflict
    )


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def rank(
    geo_text: str,
    db: AddressDB,
    dialect: str | None = None,
    topk: int = 5,
    max_dist: float = 0.40,
) -> RankResult:
    """地名文本 → 排好序的候选链 + 决策。

    geo_text 应是 normalize.extract_tail 之后的"地名部分"——门牌以下已剥离。
    """
    space = resolve_space(dialect)
    han = _han_only(geo_text)
    syls = space.romanizer(han)
    if not syls:
        return RankResult("empty", None, [], geo_text, dialect, space.name, "无可匹配的汉字")

    # 一次性在全库所有层级找命中；库只有几百条时这比逐级查更简单也更稳
    pool = [e for lv in LEVEL_ORDER for e in db.by_level[lv]]
    hits = find_hits(han, syls, pool, space, max_dist=max_dist)
    if not hits:
        return RankResult("reject", None, [], geo_text, dialect, space.name,
                          "地址库中无音近候选", all_hits=[])

    chains = build_chains(db, hits, len(han))
    if not chains:
        # 有命中但全被过滤（只剩 2 字别名撞车、或没有城市锚定的模糊路名）——
        # 这是"不知道"，不是"没有"，按 reject 把原文交出去
        return RankResult("reject", None, [], geo_text, dialect, space.name,
                          "只有弱命中（2 字别名/无锚定的模糊路名），不构成可信候选", all_hits=hits)
    for ch in chains:
        score_chain(ch, hits, len(han))
    chains.sort(key=lambda c: -c.total)
    nbest = chains[:topk]
    top = nbest[0]

    # ---- 决策 ----
    def _admin_key(c: Chain) -> tuple:
        return tuple(c.entries[lv].adcode for lv in ("province", "city", "district") if lv in c.entries)

    margin = top.total - nbest[1].total if len(nbest) > 1 else 1.0
    deep_hit = any(lv in top.hits for lv in ("district", "street", "road", "poi")) or bool(top.extra)
    if top.sim < SIM_MIN:
        decision, reason = "reject", f"Top-1 音相似度 {top.sim:.2f} < {SIM_MIN}，不采信"
    elif not deep_hit:
        # 只命中省/市级：地址库对这条地址的其余部分一无所知，不能叫 confident。
        # 未命中的地名段会原样保留（pipeline 里做），但要明确标出"未经核验"。
        decision, reason = "partial", "仅匹配到省/市级，区/路/小区未在地址库中命中，地名段原样保留"
    elif margin < MARGIN_MIN and len(nbest) > 1 and _admin_key(nbest[1]) != _admin_key(top):
        # 只有 Top-2 指向**另一个行政区**才算真歧义。
        # 同一个区下"主路选哪条"的分歧不影响送达，不该拦下来让人确认。
        decision, reason = "ambiguous", (
            f"Top-1/Top-2 分差 {margin:.3f} < {MARGIN_MIN} 且行政区不同，需人工确认"
        )
    else:
        decision, reason = "confident", f"分差 {margin:.3f}"

    return RankResult(decision, top, nbest, geo_text, dialect, space.name, reason, hits)


def explain(res: RankResult, db: AddressDB, n: int = 3) -> str:
    lines = [
        f"输入地名: {res.geo_text}",
        f"方言: {res.dialect or '未指定'}  家族: {family_of(res.dialect)}  音系: {res.space_name}",
        f"决策: {res.decision}  ({res.reason})",
    ]
    for i, ch in enumerate(res.nbest[:n], 1):
        mark = "★" if i == 1 else " "
        lines.append(
            f"{mark} #{i} {ch.full_name():<26} total={ch.total:.3f}  "
            f"sim={ch.sim:.2f} cov={ch.coverage:.2f} prior={ch.prior:.2f} "
            f"depth={ch.depth:.2f} conflict={ch.conflict:.2f}"
        )
        for lv in LEVEL_ORDER:
            h = ch.hits.get(lv)
            if h:
                seg = res.geo_text  # 展示用，span 是纯汉字索引
                lines.append(
                    f"      {lv:<8} {h.entry.name:<10} ← 命中「{h.matched_name}」 距离 {h.dist:.3f}"
                )
    return "\n".join(lines)
