"""数字口语归一化 + 地址结构化字段抽取。纯规则，零模型。

为什么这一层必须是确定性代码而不是交给 LLM：

门牌号/楼栋/单元/室号是地址里**唯一不能靠地址库检索**的部分——
库里不可能穷举每一扇门。这部分只能从语音里"听"出来，然后按规则转写。
一旦让模型"猜"门牌号，就是自信地把快递送错门。

同时数字读法在口语里有两套系统，必须区分：
    位值读法   三十号 / 一百一十六号 / 两千六百号   → 30 / 116 / 2600
    逐位读法   二零一 / 幺零幺 / 二零零八           → 201 / 101 / 2008
判据很简单：含 十/百/千/万 就是位值读法，否则逐位。

以及一条**绝不能踩**的红线：只转换紧跟地址量词的数字。
"五一广场""三峡广场""二七广场""八一路""天府三街"里的数字是专名的一部分，
转成 51广场/3峡广场 就是破坏地名。所以只认 数字+(号|栋|幢|座|单元|室|楼|层|弄)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DIGITS: dict[str, int] = {
    "零": 0, "〇": 0, "○": 0,
    "一": 1, "幺": 1, "么": 1,      # 幺/么 = 1，门牌与电话号码的口语读法
    "二": 2, "两": 2, "俩": 2,
    "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_NUM_CHARS = "".join(DIGITS) + "十百千万"
_CN_NUM = f"[{_CN_NUM_CHARS}]+"

# 只有这些量词后面的数字才会被转换（红线，见模块说明）
_UNITS = "号|栋|幢|座|单元|室|楼|层|弄"


def cn_to_int(s: str) -> int | None:
    """中文数字串 → 整数。自动区分位值读法与逐位读法。"""
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if any(c not in _CN_NUM_CHARS for c in s):
        return None

    if any(c in s for c in "十百千万"):
        return _place_value(s)

    # 逐位读法
    return int("".join(str(DIGITS[c]) for c in s))


_UNIT_VALUE = {"十": 10, "百": 100, "千": 1000}


def _place_value(s: str) -> int | None:
    """位值读法。ASR 听错会产出"八百一百八"这种不合法的串，旧版累加器把它算成
    908 并自信地写进门牌——门牌绝不猜，所以合法性校验不过就返回 None，调用方原样保留。

    校验：十/百/千 位值必须严格递减；万最多一次；两个非零数字不能相邻；百/千前必须有数字。
    补位：单位后面直接跟一个数字且中间没有"零"，按下一位算——
          八百八→880  一百五→150  两千六→2600  一万五→15000；八百零八→808（有零就是个位）。
    已知取舍：补位按普通话口语惯例，前提是 ASR 把"零"听准。方言里"零"轻读被吞掉时
    808 会变成 880——这是转写层的错，不是这里能兜的，只能靠评测集里多放带零门牌盯住。
    """
    result = 0
    pending: int | None = None   # 还没挂到位值上的非零数字
    after_zero = False           # pending 之前出现过"零"
    last_unit = 0                # 上一个单位的位值，0 = 还没出现单位
    seen_wan = False
    for ch in s:
        if ch in DIGITS:
            d = DIGITS[ch]
            if pending is not None:          # "八八百" / "八零百" / "三十二五"
                return None
            if d == 0:
                after_zero = True
            else:
                pending = d
        elif ch == "万":
            if seen_wan or (result == 0 and pending is None):
                return None
            result = (result + (pending or 0)) * 10000
            pending, after_zero, last_unit, seen_wan = None, False, 10000, True
        else:
            u = _UNIT_VALUE[ch]
            if pending is None:
                if ch != "十" or after_zero:  # "百八" / "一百零十" 不合法；"十八" 的十按 1 算
                    return None
                pending = 1
            if last_unit and u >= last_unit:  # "八百一百八" / "二十百"
                return None
            result += pending * u
            pending, after_zero, last_unit = None, False, u

    if pending is None:
        return result
    if after_zero or last_unit <= 10:
        return result + pending
    return result + pending * (last_unit // 10)   # 补位


def normalize_numbers(text: str) -> str:
    """把地址里的口语数字转成阿拉伯数字。只动量词前的数字。"""

    # 第一遍：数字 + 量词（含"附二号"这种，二后面紧跟号）
    def _with_unit(m: re.Match) -> str:
        n = cn_to_int(m.group(1))
        return f"{n}{m.group(2)}" if n is not None else m.group(0)

    out = re.sub(f"({_CN_NUM})({_UNITS})", _with_unit, text)

    # 第二遍：单元/楼 后面裸跟一串数字且没有量词 → 视为室号，补"室"
    #   三单元二零一 → 3单元201室   二十楼二零零八 → 20楼2008室
    # 只在逐位读法时补，位值读法（"三单元三十"）语义不清，不动。
    def _bare_room(m: re.Match) -> str:
        num = m.group(2)
        if any(c in num for c in "十百千万"):
            return m.group(0)
        n = cn_to_int(num)
        if n is None or len(num) < 2:
            return m.group(0)
        return f"{m.group(1)}{n}室"

    out = re.sub(
        f"(单元|楼)({_CN_NUM})(?![{_CN_NUM_CHARS}]|{_UNITS}|号|附)",
        _bare_room,
        out,
    )
    return out


# --------------------------------------------------------------------------
# 结构化字段抽取
# --------------------------------------------------------------------------


@dataclass
class TailFields:
    """地址尾部的结构化字段（门牌以下）。"""

    house_no: str = ""
    building: str = ""
    unit: str = ""
    room: str = ""
    floor: str = ""
    # 去掉上述字段后剩下的文本——是地名部分，交给地址库去匹配
    geo_text: str = ""
    spans: list[tuple[int, int, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, str]:
        return {
            "house_no": self.house_no,
            "building": self.building,
            "unit": self.unit,
            "room": self.room,
            "floor": self.floor,
        }


# 顺序有意：先匹配长的复合门牌（26号附1号），再匹配简单的
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("house_no", re.compile(r"\d+号附\d+号")),
    ("house_no", re.compile(r"\d+号(?!楼)")),
    ("building", re.compile(r"(?:\d+|[A-Za-z])(?:栋|幢|座|号楼)")),
    ("unit", re.compile(r"\d+单元")),
    ("room", re.compile(r"\d+室")),
    ("floor", re.compile(r"\d+(?:楼|层)")),
]


def extract_tail(text: str) -> TailFields:
    """从（已数字归一化的）文本里抽出门牌/楼栋/单元/室/楼层。

    同一字段出现多次时取**第一个**——地址是从大到小说的，
    第一个更可能是真的；后面的多半是复述或口误。
    """
    tf = TailFields()
    taken: list[tuple[int, int]] = []

    def _overlaps(s: int, e: int) -> bool:
        return any(not (e <= a or s >= b) for a, b in taken)

    for fname, pat in _PATTERNS:
        if getattr(tf, fname):
            continue
        for m in pat.finditer(text):
            if _overlaps(m.start(), m.end()):
                continue
            setattr(tf, fname, m.group(0))
            taken.append((m.start(), m.end()))
            tf.spans.append((m.start(), m.end(), fname))
            break

    # 剩余文本 = 地名部分
    keep = []
    last = 0
    for s, e in sorted(taken):
        keep.append(text[last:s])
        last = e
    keep.append(text[last:])
    tf.geo_text = "".join(keep)
    return tf


def normalize(text: str) -> tuple[str, TailFields]:
    """一步到位：数字归一化 + 尾部抽取。返回 (归一化文本, 字段)。"""
    norm = normalize_numbers(text)
    return norm, extract_tail(norm)


def explain(text: str) -> str:
    norm, tf = normalize(text)
    lines = [f"原文:   {text}", f"归一:   {norm}"]
    for k, v in tf.as_dict().items():
        if v:
            lines.append(f"  {k:<9} {v}")
    lines.append(f"  地名部分  {tf.geo_text}")
    return "\n".join(lines)
