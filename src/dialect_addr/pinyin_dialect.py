"""方言模糊音加权拼音距离。

整套方案的核心假设：**方言误识错的是字，不是音。**

    真值      成都武侯区红牌楼      wu hou qu hong pai lou
    ASR 输出  成都五后区红排楼      wu hou qu hong pai lou
    汉字编辑距离 = 3（看着全错）
    拼音编辑距离 = 0（完全一致）

所以匹配要在拼音空间做。但光转拼音还不够——方言音变是**系统性**的，
标准 Levenshtein 把每个替换都算代价 1，会把"n/l 不分"这种规律性音变
误判成"差很远"。因此需要一张带权代价矩阵，让系统性音变的替换代价接近 0。

实现上是两层距离：
    外层  音节序列做 Levenshtein 对齐
    内层  音节间距离 = 声母代价 + 韵母代价（各自查混淆表）

比"把整串拼音拼成字符串再算编辑距离"准得多：后者会因为音节边界错位
产生大量虚假代价（"chang an" vs "cha ngan"）。
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field

try:
    from pypinyin import Style, lazy_pinyin
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 pypinyin：pip install pypinyin") from exc


# --------------------------------------------------------------------------
# 混淆表
#
# 每条 = (可互换的音素集合, 替换代价, 说明, 主要方言区)
# 代价 0.0~1.0，0 表示完全等价，1 表示与随机替换无异（标准编辑距离的默认值）。
# 数值依据：音变越普遍、越规律，代价越低。
# --------------------------------------------------------------------------

INITIAL_CONFUSIONS: list[tuple[frozenset[str], float, str, tuple[str, ...]]] = [
    (frozenset({"n", "l"}), 0.10, "边鼻音不分", ("西南官话", "湘", "赣", "江淮官话")),
    (frozenset({"zh", "z"}), 0.12, "平翘舌不分", ("西南官话", "江淮官话", "闽", "湘")),
    (frozenset({"ch", "c"}), 0.12, "平翘舌不分", ("西南官话", "江淮官话", "闽", "湘")),
    (frozenset({"sh", "s"}), 0.12, "平翘舌不分", ("西南官话", "江淮官话", "闽", "湘")),
    (frozenset({"h", "f"}), 0.20, "h/f 不分", ("西南官话", "湘", "闽")),
    (frozenset({"r", "l"}), 0.25, "r 声母边音化", ("西南官话",)),
    (frozenset({"r", "y", ""}), 0.30, "r 声母脱落/半元音化", ("吴", "粤", "闽")),
    (frozenset({"n", "r"}), 0.35, "n/r 相混", ("西南官话",)),
    # 尖团音：吴语及部分官话中 j/q/x 与 z/c/s 分立，普通话已合并
    (frozenset({"j", "z"}), 0.30, "尖团音分立", ("吴", "部分官话")),
    (frozenset({"q", "c"}), 0.30, "尖团音分立", ("吴", "部分官话")),
    (frozenset({"x", "s"}), 0.30, "尖团音分立", ("吴", "部分官话")),
    # 浊音清化：吴语保留浊声母，普通话已清化
    (frozenset({"b", "p"}), 0.35, "浊音清化", ("吴", "老湘")),
    (frozenset({"d", "t"}), 0.35, "浊音清化", ("吴", "老湘")),
    (frozenset({"g", "k"}), 0.35, "浊音清化", ("吴", "老湘")),
    (frozenset({"zh", "j"}), 0.40, "知组与见组相混", ("闽", "客")),
    # 见系不颚化：普通话里 g/k/h 在细音前已颚化成 j/q/x，西南官话/湘/赣/客大量保留古读。
    # 街=gai、鞋=hai、交=gao、解=gai、去=kê。实测：四川话「交通巷」被 ASR 听成「高通巷」。
    (frozenset({"g", "j"}), 0.20, "见系不颚化（交=gao 街=gai）", ("西南官话", "湘", "赣", "客")),
    (frozenset({"k", "q"}), 0.20, "溪群不颚化（去=kê）", ("西南官话", "湘", "赣", "客")),
    (frozenset({"h", "x"}), 0.20, "晓匣不颚化（鞋=hai 下=ha）", ("西南官话", "湘", "赣", "客")),
    (frozenset({"w", ""}), 0.30, "零声母/w 交替", ("粤", "吴")),
    (frozenset({"m", "b"}), 0.40, "明母塞化", ("闽",)),
]

FINAL_CONFUSIONS: list[tuple[frozenset[str], float, str, tuple[str, ...]]] = [
    # 前后鼻音——南方方言最普遍的音变，代价给最低
    (frozenset({"an", "ang"}), 0.10, "前后鼻音不分", ("西南官话", "吴", "湘", "赣", "闽")),
    (frozenset({"en", "eng"}), 0.10, "前后鼻音不分", ("西南官话", "吴", "湘", "赣", "闽")),
    (frozenset({"in", "ing"}), 0.08, "前后鼻音不分", ("西南官话", "吴", "湘", "赣", "闽", "江淮官话")),
    (frozenset({"ian", "iang"}), 0.12, "前后鼻音不分", ("西南官话", "湘", "赣")),
    (frozenset({"uan", "uang"}), 0.12, "前后鼻音不分", ("西南官话", "湘", "赣")),
    (frozenset({"un", "ong"}), 0.25, "un/ong 相混", ("西南官话", "闽")),
    (frozenset({"uen", "ueng"}), 0.15, "前后鼻音不分", ("西南官话",)),
    # 单元音与复元音
    (frozenset({"e", "o"}), 0.25, "e/o 相混", ("西南官话", "晋")),
    (frozenset({"o", "uo"}), 0.15, "o/uo 相混", ("西南官话", "中原官话")),
    (frozenset({"e", "uo"}), 0.25, "e/uo 相混", ("西南官话",)),
    (frozenset({"ui", "uei"}), 0.05, "同音异写", ("通用",)),
    (frozenset({"iu", "iou"}), 0.05, "同音异写", ("通用",)),
    (frozenset({"un", "uen"}), 0.05, "同音异写", ("通用",)),
    (frozenset({"ui", "ei"}), 0.25, "介音脱落", ("西南官话", "闽")),
    (frozenset({"iu", "ou"}), 0.25, "介音脱落", ("西南官话", "闽")),
    # 与见系不颚化伴生：交 jiao→gao、街 jie→gai 时 i 介音一并脱落
    (frozenset({"iao", "ao"}), 0.20, "介音脱落（见系不颚化伴生）", ("西南官话", "湘", "赣", "客")),
    (frozenset({"ie", "ai"}), 0.30, "街=gai 型韵母对应", ("西南官话", "湘", "赣", "客")),
    (frozenset({"ia", "a"}), 0.25, "介音脱落（下=ha）", ("西南官话", "湘", "赣", "客")),
    (frozenset({"ai", "ei"}), 0.30, "ai/ei 相混", ("晋", "中原官话")),
    (frozenset({"ao", "ou"}), 0.30, "ao/ou 相混", ("晋",)),
    (frozenset({"ie", "ian"}), 0.35, "鼻音韵尾脱落", ("吴", "闽")),
    (frozenset({"v", "u"}), 0.20, "撮口呼混同", ("西南官话", "闽", "粤")),
    (frozenset({"ve", "ue"}), 0.05, "同音异写", ("通用",)),
    (frozenset({"i", "er"}), 0.40, "儿化差异", ("北方官话",)),
    # 入声：吴/粤/闽/客保留 -p/-t/-k 韵尾，普通话已消失
    (frozenset({"a", "ai"}), 0.35, "入声派入", ("吴", "粤")),
    (frozenset({"e", "ei"}), 0.35, "入声派入", ("吴", "粤")),
]


@dataclass(frozen=True)
class DialectProfile:
    """一个方言区的混淆代价配置。

    name         方言名
    initial_cost (声母a, 声母b) -> 代价
    final_cost   (韵母a, 韵母b) -> 代价
    w_initial    声母在音节距离中的权重
    w_final      韵母在音节距离中的权重（韵母承载更多辨义信息，权重更高）
    """

    name: str
    initial_cost: dict[tuple[str, str], float] = field(default_factory=dict)
    final_cost: dict[tuple[str, str], float] = field(default_factory=dict)
    w_initial: float = 0.45
    w_final: float = 0.55


def _build_cost_table(
    confusions: list[tuple[frozenset[str], float, str, tuple[str, ...]]],
    dialects: tuple[str, ...] | None,
) -> dict[tuple[str, str], float]:
    """把混淆组展开成两两代价字典。

    dialects=None 表示收录全部规则（通用档）。
    同一对音素被多条规则命中时取**最低**代价——只要有任一方言解释得通，
    就不该重罚。
    """
    table: dict[tuple[str, str], float] = {}
    for group, cost, _desc, dia in confusions:
        if dialects is not None and not (set(dia) & set(dialects) or "通用" in dia):
            continue
        members = sorted(group)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                for key in ((a, b), (b, a)):
                    if key not in table or cost < table[key]:
                        table[key] = cost
    return table


def make_profile(name: str, dialects: tuple[str, ...] | None = None) -> DialectProfile:
    return DialectProfile(
        name=name,
        initial_cost=_build_cost_table(INITIAL_CONFUSIONS, dialects),
        final_cost=_build_cost_table(FINAL_CONFUSIONS, dialects),
    )


# 通用档收录全部混淆规则。
#
# 设计取舍：不知道说话人是哪种方言时，通用档是正确默认——漏掉一条音变
# 会让正确候选排不进 Top-K（不可恢复），而多收一条只是让错误候选分数略高
# （后续打分函数里还有地址先验、层级合法性等项可以纠正）。
# 已知方言时切到专用档更准，但收益远小于"用对拼音空间"本身。
GENERIC = make_profile("通用", None)
SICHUAN = make_profile("四川话", ("西南官话",))
WU = make_profile("吴语", ("吴",))
YUE = make_profile("粤语", ("粤",))
MIN = make_profile("闽语", ("闽",))
XIANG = make_profile("湘语", ("湘",))
JIANGHUAI = make_profile("江淮官话", ("江淮官话",))
ZHONGYUAN = make_profile("中原官话", ("中原官话",))

PROFILES: dict[str, DialectProfile] = {
    p.name: p
    for p in (GENERIC, SICHUAN, WU, YUE, MIN, XIANG, JIANGHUAI, ZHONGYUAN)
}


# --------------------------------------------------------------------------
# 音节
# --------------------------------------------------------------------------

_HAN = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class Syllable:
    initial: str  # 声母，零声母为 ''
    final: str    # 韵母
    raw: str      # 完整拼音（无声调）

    def __str__(self) -> str:  # pragma: no cover
        return self.raw


def _norm_final(f: str) -> str:
    """统一韵母写法，消除 pypinyin 的书写变体。"""
    return f.replace("ü", "v").replace("ǖ", "v").replace("ń", "n")


@functools.lru_cache(maxsize=8192)
def to_syllables(text: str) -> tuple[Syllable, ...]:
    """汉字串 → 音节序列（不含声调）。

    非汉字（数字/字母/标点）直接丢弃：地址里的门牌号走独立的规则归一化
    通道处理，不参与拼音匹配。混进来只会污染距离。
    """
    han = "".join(_HAN.findall(text))
    if not han:
        return ()
    fulls = lazy_pinyin(han, style=Style.NORMAL, errors="ignore")
    inits = lazy_pinyin(han, style=Style.INITIALS, strict=False, errors="ignore")
    finals = lazy_pinyin(han, style=Style.FINALS, strict=False, errors="ignore")
    n = min(len(fulls), len(inits), len(finals))
    return tuple(
        Syllable(initial=inits[i], final=_norm_final(finals[i]), raw=fulls[i])
        for i in range(n)
    )


def syllable_distance(a: Syllable, b: Syllable, profile: DialectProfile = GENERIC) -> float:
    """两个音节的距离，归一化到 0~1。"""
    if a.raw == b.raw:
        return 0.0

    if a.initial == b.initial:
        ic = 0.0
    else:
        ic = profile.initial_cost.get((a.initial, b.initial), 1.0)

    if a.final == b.final:
        fc = 0.0
    else:
        fc = profile.final_cost.get((a.final, b.final), 1.0)

    return profile.w_initial * ic + profile.w_final * fc


def syllable_edit_distance(
    xs: tuple[Syllable, ...],
    ys: tuple[Syllable, ...],
    profile: DialectProfile = GENERIC,
    indel_cost: float = 1.0,
) -> float:
    """音节序列的加权 Levenshtein 距离（未归一化）。"""
    n, m = len(xs), len(ys)
    if n == 0:
        return m * indel_cost
    if m == 0:
        return n * indel_cost

    prev = [j * indel_cost for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [i * indel_cost] + [0.0] * m
        xi = xs[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j] + indel_cost,                                  # 删除
                cur[j - 1] + indel_cost,                               # 插入
                prev[j - 1] + syllable_distance(xi, ys[j - 1], profile),  # 替换
            )
        prev = cur
    return prev[m]


def pinyin_distance(a: str, b: str, profile: DialectProfile = GENERIC) -> float:
    """两个中文串的方言加权拼音距离，归一化到 0~1。"""
    xs, ys = to_syllables(a), to_syllables(b)
    if not xs and not ys:
        return 0.0
    if not xs or not ys:
        return 1.0
    return syllable_edit_distance(xs, ys, profile) / max(len(xs), len(ys))


def pinyin_similarity(a: str, b: str, profile: DialectProfile = GENERIC) -> float:
    """1 - 距离。越接近 1 越像。"""
    return 1.0 - pinyin_distance(a, b, profile)


def explain(a: str, b: str, profile: DialectProfile = GENERIC) -> str:
    """人类可读的逐音节对比，用于调试和演示。"""
    xs, ys = to_syllables(a), to_syllables(b)
    lines = [
        f"[{profile.name}] {a}  vs  {b}",
        f"  拼音A: {' '.join(s.raw for s in xs)}",
        f"  拼音B: {' '.join(s.raw for s in ys)}",
        f"  加权距离: {pinyin_distance(a, b, profile):.4f}"
        f"   相似度: {pinyin_similarity(a, b, profile):.4f}",
    ]
    if len(xs) == len(ys):
        diffs = [
            f"    {x.raw}→{y.raw}  代价 {syllable_distance(x, y, profile):.2f}"
            for x, y in zip(xs, ys)
            if x.raw != y.raw
        ]
        if diffs:
            lines.append("  逐音节差异:")
            lines.extend(diffs)
    return "\n".join(lines)
