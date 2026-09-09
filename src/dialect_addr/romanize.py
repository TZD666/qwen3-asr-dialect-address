"""多音系罗马化路由：为不同方言选择正确的「音的坐标系」。

这是对纯普通话拼音方案的关键修正。

普通话拼音方案的隐含前提是「方言 = 普通话的音变」。这对**官话系**成立
（四川话 wu hou qu 和普通话 wu hou qu 是同一套音系的变体），
对**非官话系**根本不成立：

    地名   普通话           粤拼            关系
    广州   guang zhou      gwong zau       勉强对应
    番禺   pan yu          pun jyu         声母都不同
    深圳   shen zhen       sam zan         有 -m 韵尾，普通话没有

粤语有 -p/-t/-k 入声韵尾和 6~9 个声调，普通话一个都没有。
拿普通话拼音去比对粤语误识，等于**用错误的坐标系量距离**。

因此按方言路由到不同的音系空间：

    官话系(12)      → 普通话拼音   pypinyin
    粤语(2)         → 粤拼         pycantonese
    闽南/闽(2)      → 台罗         taibun
    吴/赣/湘/晋等   → 无成熟库，跳过罗马化，交给声学重打分兜底

方言库都是**可选依赖**：没装就把对应路径标记为不可用并降级，
不让 import 失败拖垮整个流程。这不是防御性编程洁癖——
生产环境里少数派方言的库经常装不上，主链路必须活着。
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Callable

from .pinyin_dialect import (
    DialectProfile,
    GENERIC,
    JIANGHUAI,
    MIN as MIN_PROFILE,
    SICHUAN,
    Syllable,
    WU as WU_PROFILE,
    YUE as YUE_PROFILE,
    ZHONGYUAN,
    _build_cost_table,
    syllable_edit_distance,
    to_syllables as mandarin_syllables,
)

# --------------------------------------------------------------------------
# 粤拼（Jyutping）
# --------------------------------------------------------------------------

# 粤语**内部**的音变（不是与普通话的对应关系）。
# 这些是广州/香港粤语正在发生的懒音现象，ASR 误识主要沿这些方向。
JYUTPING_INITIAL_CONFUSIONS = [
    (frozenset({"n", "l"}), 0.10, "n/l 懒音（现代广州话普遍）", ("粤",)),
    (frozenset({"ng", ""}), 0.12, "ng 声母脱落", ("粤",)),
    (frozenset({"gw", "g"}), 0.20, "gw→g 圆唇丢失", ("粤",)),
    (frozenset({"kw", "k"}), 0.20, "kw→k 圆唇丢失", ("粤",)),
    (frozenset({"z", "j"}), 0.35, "z/j 相混", ("粤",)),
]

JYUTPING_FINAL_CONFUSIONS = [
    (frozenset({"ang", "an"}), 0.15, "-ng/-n 韵尾合并", ("粤",)),
    (frozenset({"ong", "on"}), 0.15, "-ng/-n 韵尾合并", ("粤",)),
    (frozenset({"eng", "en"}), 0.15, "-ng/-n 韵尾合并", ("粤",)),
    (frozenset({"ing", "in"}), 0.15, "-ng/-n 韵尾合并", ("粤",)),
    (frozenset({"ung", "un"}), 0.15, "-ng/-n 韵尾合并", ("粤",)),
    (frozenset({"ak", "at"}), 0.18, "-k/-t 入声韵尾合并", ("粤",)),
    (frozenset({"ok", "ot"}), 0.18, "-k/-t 入声韵尾合并", ("粤",)),
    (frozenset({"ik", "it"}), 0.18, "-k/-t 入声韵尾合并", ("粤",)),
    (frozenset({"uk", "ut"}), 0.18, "-k/-t 入声韵尾合并", ("粤",)),
    (frozenset({"am", "an"}), 0.25, "-m/-n 韵尾合并", ("粤",)),
    (frozenset({"im", "in"}), 0.25, "-m/-n 韵尾合并", ("粤",)),
    (frozenset({"ap", "at"}), 0.25, "-p/-t 韵尾合并", ("粤",)),
    (frozenset({"eoi", "eoy"}), 0.05, "同音异写", ("粤",)),
    (frozenset({"yu", "jyu"}), 0.10, "书写变体", ("粤",)),
]

JYUTPING_PROFILE = DialectProfile(
    name="粤拼",
    initial_cost=_build_cost_table(JYUTPING_INITIAL_CONFUSIONS, None),
    final_cost=_build_cost_table(JYUTPING_FINAL_CONFUSIONS, None),
)

# 粤拼声母表，按长度降序以便最长匹配（ng/gw/kw 是双字母声母）
_JP_INITIALS = [
    "ng", "gw", "kw", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "z", "c", "s", "j", "w",
]


def _split_jyutping(syl: str) -> tuple[str, str]:
    """粤拼音节 → (声母, 韵母)，输入需已去声调。"""
    for ini in _JP_INITIALS:
        if syl.startswith(ini):
            rest = syl[len(ini):]
            if rest:                     # 避免把整个音节当成声母
                return ini, rest
    return "", syl


class JyutpingRomanizer:
    """汉字 → 粤拼音节序列。依赖 pycantonese（可选）。"""

    name = "粤拼"
    profile = JYUTPING_PROFILE

    def __init__(self) -> None:
        try:
            import pycantonese  # noqa: F401

            self._pc = pycantonese
            self.available = True
            self.reason = ""
        except ImportError as exc:
            self._pc = None
            self.available = False
            self.reason = f"pycantonese 未安装: {exc}"

    def __call__(self, text: str) -> tuple[Syllable, ...]:
        if not self.available:
            return ()
        out: list[Syllable] = []
        # characters_to_jyutping 返回 [(词, 粤拼串)]，粤拼串是整词连写带声调
        for _word, jp in self._pc.characters_to_jyutping(text):
            if not jp:
                continue
            # 拆成音节：每个音节以声调数字结尾
            cur = ""
            for ch in jp:
                cur += ch
                if ch.isdigit():
                    syl = cur[:-1]          # 去掉声调
                    ini, fin = _split_jyutping(syl)
                    out.append(Syllable(initial=ini, final=fin, raw=syl))
                    cur = ""
            if cur:                          # 无声调残余
                ini, fin = _split_jyutping(cur)
                out.append(Syllable(initial=ini, final=fin, raw=cur))
        return tuple(out)


# --------------------------------------------------------------------------
# 台罗（闽南语）
# --------------------------------------------------------------------------

TAILO_INITIAL_CONFUSIONS = [
    # 送气对立：闽南语 t/th、k/kh、p/ph 分立，但 ASR 对送气的判别最脆弱
    (frozenset({"t", "th"}), 0.25, "送气对立弱化", ("闽",)),
    (frozenset({"k", "kh"}), 0.25, "送气对立弱化", ("闽",)),
    (frozenset({"p", "ph"}), 0.25, "送气对立弱化", ("闽",)),
    (frozenset({"n", "l"}), 0.15, "n/l 相混", ("闽",)),
    (frozenset({"b", "m"}), 0.20, "b/m 鼻化交替", ("闽",)),
    (frozenset({"g", "ng"}), 0.20, "g/ng 交替", ("闽",)),
    (frozenset({"j", "l"}), 0.25, "j/l 相混（泉漳差异）", ("闽",)),
    (frozenset({"ts", "tsh"}), 0.30, "送气对立弱化", ("闽",)),
]

TAILO_FINAL_CONFUSIONS = [
    # 泉州腔 vs 漳州腔的系统性差异
    (frozenset({"ir", "i"}), 0.15, "泉漳腔差异", ("闽",)),
    (frozenset({"er", "e"}), 0.15, "泉漳腔差异", ("闽",)),
    (frozenset({"ue", "e"}), 0.20, "泉漳腔差异", ("闽",)),
    (frozenset({"ing", "in"}), 0.15, "鼻韵尾合并", ("闽",)),
    (frozenset({"ang", "an"}), 0.15, "鼻韵尾合并", ("闽",)),
    (frozenset({"ik", "it"}), 0.20, "入声韵尾合并", ("闽",)),
    (frozenset({"ak", "at"}), 0.20, "入声韵尾合并", ("闽",)),
]

TAILO_PROFILE = DialectProfile(
    name="台罗",
    initial_cost=_build_cost_table(TAILO_INITIAL_CONFUSIONS, None),
    final_cost=_build_cost_table(TAILO_FINAL_CONFUSIONS, None),
)

_TL_INITIALS = [
    "tsh", "ts", "ng", "ph", "th", "kh", "b", "p", "m", "t", "n",
    "l", "k", "g", "h", "j", "s",
]


def _split_tailo(syl: str) -> tuple[str, str]:
    for ini in _TL_INITIALS:
        if syl.startswith(ini):
            rest = syl[len(ini):]
            if rest:
                return ini, rest
    return "", syl


class TailoRomanizer:
    """汉字 → 台罗音节序列。依赖 taibun（可选）。

    注意闽南语有**文白异读**：同一个字在不同词里读音完全不同
    （人 = jîn 文读 / lâng 白读）。taibun 会按词选择，但地名
    往往两读并存，属于本路径的已知不确定性。
    """

    name = "台罗"
    profile = TAILO_PROFILE

    def __init__(self) -> None:
        try:
            from taibun import Converter

            self._conv = Converter(system="Tailo", punctuation="none")
            self.available = True
            self.reason = ""
        except Exception as exc:          # taibun 初始化可能因数据文件失败
            self._conv = None
            self.available = False
            self.reason = f"taibun 不可用: {exc}"

    def __call__(self, text: str) -> tuple[Syllable, ...]:
        if not self.available:
            return ()
        try:
            romanized = self._conv.get(text)
        except Exception:
            return ()
        out: list[Syllable] = []
        for token in romanized.replace("-", " ").split():
            # 台罗用附加符号标声调（tâi / kóo / lōng）。带调字母 isalpha() 为 True，
            # 必须先 NFD 分解再丢掉组合符号，否则 tâi 和 tai 会被判成不同音节。
            decomposed = unicodedata.normalize("NFD", token.lower())
            syl = "".join(
                c for c in decomposed
                if c.isalpha() and not unicodedata.combining(c)
            )
            if not syl:
                continue
            ini, fin = _split_tailo(syl)
            out.append(Syllable(initial=ini, final=fin, raw=syl))
        return tuple(out)


# --------------------------------------------------------------------------
# 音系空间与方言路由
# --------------------------------------------------------------------------


@dataclass
class PhonSpace:
    """一个「音的坐标系」：罗马化器 + 该音系内部的混淆代价表。"""

    name: str
    romanizer: Callable[[str], tuple[Syllable, ...]]
    profile: DialectProfile
    available: bool = True
    reason: str = ""


class _MandarinRomanizer:
    name = "普通话拼音"
    available = True
    reason = ""

    def __call__(self, text: str) -> tuple[Syllable, ...]:
        return mandarin_syllables(text)


_jyut = JyutpingRomanizer()
_tailo = TailoRomanizer()
_mand = _MandarinRomanizer()


def _space(name: str, rom, profile: DialectProfile) -> PhonSpace:
    return PhonSpace(
        name=name,
        romanizer=rom,
        profile=profile,
        available=getattr(rom, "available", True),
        reason=getattr(rom, "reason", ""),
    )


SPACES: dict[str, PhonSpace] = {
    "mandarin":  _space("普通话拼音", _mand, GENERIC),
    "mandarin_xinan":     _space("普通话拼音·西南官话", _mand, SICHUAN),
    "mandarin_zhongyuan": _space("普通话拼音·中原官话", _mand, ZHONGYUAN),
    "mandarin_jianghuai": _space("普通话拼音·江淮官话", _mand, JIANGHUAI),
    "mandarin_wu":        _space("普通话拼音·吴语近似", _mand, WU_PROFILE),
    "mandarin_min":       _space("普通话拼音·闽语近似", _mand, MIN_PROFILE),
    "mandarin_yue":       _space("普通话拼音·粤语近似", _mand, YUE_PROFILE),
    "jyutping":  _space("粤拼", _jyut, JYUTPING_PROFILE),
    "tailo":     _space("台罗", _tailo, TAILO_PROFILE),
}


# 模型卡列出的 22 种中文方言 → 音系空间。
# fallback 是**必需字段**：主空间不可用（库没装）时退到它。
# 吴/赣/湘/晋没有成熟罗马化库，主空间直接用"普通话近似"，
# 真正的准确性靠声学重打分补，而不是假装拼音能救。
DIALECT_ROUTING: dict[str, tuple[str, str, str]] = {
    # 方言名          主空间              兜底空间       家族
    "Sichuan":   ("mandarin_xinan", "mandarin", "官话系"),
    "Guizhou":   ("mandarin_xinan", "mandarin", "官话系"),
    "Yunnan":    ("mandarin_xinan", "mandarin", "官话系"),
    "Hubei":     ("mandarin_xinan", "mandarin", "官话系"),
    "Henan":     ("mandarin_zhongyuan", "mandarin", "官话系"),
    "Shaanxi":   ("mandarin_zhongyuan", "mandarin", "官话系"),
    "Gansu":     ("mandarin_zhongyuan", "mandarin", "官话系"),
    "Ningxia":   ("mandarin_zhongyuan", "mandarin", "官话系"),
    "Dongbei":   ("mandarin", "mandarin", "官话系"),
    "Hebei":     ("mandarin", "mandarin", "官话系"),
    "Tianjin":   ("mandarin", "mandarin", "官话系"),
    "Shandong":  ("mandarin", "mandarin", "官话系"),
    # 过渡区：晋语有入声，安徽跨江淮/中原/吴/徽多片
    "Shanxi":    ("mandarin_zhongyuan", "mandarin", "过渡区"),
    "Anhui":     ("mandarin_jianghuai", "mandarin", "过渡区"),
    # 非官话，有专用库
    "Cantonese (Hong Kong accent)": ("jyutping", "mandarin_yue", "非官话·有库"),
    "Cantonese (Guangdong accent)": ("jyutping", "mandarin_yue", "非官话·有库"),
    "Minnan language":              ("tailo",    "mandarin_min", "非官话·有库"),
    "Fujian":                       ("tailo",    "mandarin_min", "非官话·有库"),
    # 非官话，无成熟库 —— 主要靠声学兜底
    "Wu language": ("mandarin_wu", "mandarin", "非官话·无库"),
    "Zhejiang":    ("mandarin_wu", "mandarin", "非官话·无库"),
    "Hunan":       ("mandarin", "mandarin", "非官话·无库"),
    "Jiangxi":     ("mandarin", "mandarin", "非官话·无库"),
}

# 中文别名，方便按中文方言名路由
ALIASES: dict[str, str] = {
    "四川": "Sichuan", "四川话": "Sichuan", "成都": "Sichuan", "重庆": "Sichuan",
    "贵州": "Guizhou", "贵阳": "Guizhou", "云南": "Yunnan", "昆明": "Yunnan",
    "湖北": "Hubei", "武汉": "Hubei", "河南": "Henan", "郑州": "Henan",
    "陕西": "Shaanxi", "西安": "Shaanxi", "甘肃": "Gansu", "宁夏": "Ningxia",
    "东北": "Dongbei", "河北": "Hebei", "天津": "Tianjin", "山东": "Shandong",
    "山西": "Shanxi", "晋语": "Shanxi", "安徽": "Anhui",
    "粤语": "Cantonese (Guangdong accent)", "广东话": "Cantonese (Guangdong accent)",
    "广州话": "Cantonese (Guangdong accent)", "香港": "Cantonese (Hong Kong accent)",
    "闽南": "Minnan language", "闽南语": "Minnan language", "台语": "Minnan language",
    "福建": "Fujian", "福州": "Fujian",
    "吴语": "Wu language", "上海": "Wu language", "上海话": "Wu language",
    "苏州": "Wu language", "浙江": "Zhejiang", "杭州": "Zhejiang",
    "湖南": "Hunan", "长沙": "Hunan", "湘语": "Hunan",
    "江西": "Jiangxi", "赣语": "Jiangxi", "南昌": "Jiangxi",
}


def resolve_space(dialect: str | None) -> PhonSpace:
    """方言名 → 可用的音系空间。

    dialect 为 None 或不认识时返回通用普通话空间——
    这是安全默认：官话系覆盖 22 个里的 12 个，且对其余方言
    至少能提供部分信号，比直接放弃强。
    """
    if not dialect:
        return SPACES["mandarin"]
    key = ALIASES.get(dialect, dialect)
    entry = DIALECT_ROUTING.get(key)
    if entry is None:
        return SPACES["mandarin"]
    primary, fallback, _family = entry
    sp = SPACES[primary]
    if sp.available:
        return sp
    return SPACES[fallback]


def family_of(dialect: str | None) -> str:
    if not dialect:
        return "未知"
    key = ALIASES.get(dialect, dialect)
    entry = DIALECT_ROUTING.get(key)
    return entry[2] if entry else "未知"


def distance(a: str, b: str, dialect: str | None = None) -> float:
    """在**该方言对应的音系空间**里算归一化距离。"""
    sp = resolve_space(dialect)
    xs, ys = sp.romanizer(a), sp.romanizer(b)
    if not xs or not ys:
        # 该音系罗马化失败（库缺失或转换不出），退回普通话空间。
        # 返回 1.0 会让上层误以为"完全不像"，比降级更糟。
        mand = SPACES["mandarin"]
        xs, ys = mand.romanizer(a), mand.romanizer(b)
        if not xs or not ys:
            return 1.0
        return syllable_edit_distance(xs, ys, mand.profile) / max(len(xs), len(ys))
    return syllable_edit_distance(xs, ys, sp.profile) / max(len(xs), len(ys))


def similarity(a: str, b: str, dialect: str | None = None) -> float:
    return 1.0 - distance(a, b, dialect)


def status() -> str:
    """各音系空间的可用性报告——部署时先看这个。"""
    lines = ["音系空间可用性:"]
    for key, sp in SPACES.items():
        mark = "可用" if sp.available else "不可用"
        extra = f"  ({sp.reason})" if sp.reason else ""
        lines.append(f"  [{mark}] {key:<20} {sp.name}{extra}")
    lines.append("")
    lines.append("22 种方言路由:")
    fam_order = ["官话系", "过渡区", "非官话·有库", "非官话·无库"]
    for fam in fam_order:
        names = [d for d, v in DIALECT_ROUTING.items() if v[2] == fam]
        lines.append(f"  {fam} ({len(names)}): {', '.join(names)}")
    return "\n".join(lines)
