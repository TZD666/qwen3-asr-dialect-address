"""方言词汇归一化：把方言口语转成标准中文，供地址解析使用。

为什么需要这一层（也是纯拼音方案最大的盲点）：

方言差异不只是**发音**，还有**词汇和语法**。
    粤语   我住喺天河区      喺 = 在
    闽南   阮兜佇台江区      阮兜 = 我家，佇 = 在
    四川   我屋头在武侯区    屋头 = 家
    东北   就在那旮旯        旮旯 = 角落
这些词在普通话里根本不存在，拼音距离再准也救不回来。

**但地址领域有个救命的性质：这里用到的方言词是封闭集，不是开放集。**

人说地址时用到的方言词就四类——居所词、方位词、建筑量词、数字读法，
穷举下来几百条量级。所以可以用确定性查表解决，不需要让 LLM 去猜。
这符合"关键判断用工程学而非 prompt"的工程原则：查表的行为可预测、
可测试、可审计，LLM 改写则可能把说对的地址"纠正"错。

注意方向：本模块处理的是 **ASR 输出的方言文本 → 标准中文**，
不是语音层的事。语音层由 romanize.py 负责。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 方言分组标识
YUE = "粤语"
MIN = "闽南"
WU = "吴语"
XINAN = "西南官话"
DONGBEI = "东北官话"
XIANG = "湘语"
GAN = "赣语"
ZHONGYUAN = "中原官话"
JIANGHUAI = "江淮官话"
JIN = "晋语"
ALL = "通用"


@dataclass(frozen=True)
class LexEntry:
    """一条方言词映射。

    standard 为空字符串表示**直接删除**（语气助词一类，没有对应标准词）。
    """

    dialect: str
    standard: str
    groups: tuple[str, ...]
    category: str


# --------------------------------------------------------------------------
# 词表
#
# 排序无所谓——实际替换时按长度降序做最长匹配，避免"屋企"被"屋"先吃掉。
# --------------------------------------------------------------------------

LEXICON: list[LexEntry] = [
    # ---------- 居所词 ----------
    LexEntry("屋企", "家", (YUE,), "居所"),
    LexEntry("屋歧", "家", (YUE,), "居所"),          # ASR 常见同音误写
    LexEntry("阮兜", "我家", (MIN,), "居所"),
    LexEntry("恁兜", "你家", (MIN,), "居所"),
    LexEntry("伊兜", "他家", (MIN,), "居所"),
    LexEntry("厝内", "家里", (MIN,), "居所"),
    LexEntry("厝", "房子", (MIN,), "居所"),
    LexEntry("屋头", "家", (XINAN,), "居所"),
    LexEntry("屋里头", "家里", (XINAN, XIANG), "居所"),
    LexEntry("屋里", "家", (XIANG, WU), "居所"),
    LexEntry("家儿", "家", (DONGBEI,), "居所"),
    LexEntry("院坝", "院子", (XINAN,), "居所"),
    LexEntry("院坝头", "院子里", (XINAN,), "居所"),
    LexEntry("弄堂", "弄", (WU,), "居所"),
    LexEntry("屋邨", "村", (YUE,), "居所"),
    LexEntry("邨", "村", (YUE,), "居所"),

    # ---------- 方位介词 / 指示词 ----------
    LexEntry("喺度", "在这里", (YUE,), "方位"),
    LexEntry("喺", "在", (YUE,), "方位"),
    LexEntry("係", "在", (YUE,), "方位"),
    LexEntry("呢度", "这里", (YUE,), "方位"),
    LexEntry("嗰度", "那里", (YUE,), "方位"),
    LexEntry("边度", "哪里", (YUE,), "方位"),
    LexEntry("呢便", "这边", (YUE,), "方位"),
    LexEntry("嗰便", "那边", (YUE,), "方位"),
    LexEntry("佇遮", "在这里", (MIN,), "方位"),
    LexEntry("佇遐", "在那里", (MIN,), "方位"),
    LexEntry("佗位", "哪里", (MIN,), "方位"),
    LexEntry("佇", "在", (MIN,), "方位"),
    LexEntry("遮", "这里", (MIN,), "方位"),
    LexEntry("遐", "那里", (MIN,), "方位"),
    LexEntry("头前", "前面", (MIN, XIANG), "方位"),
    LexEntry("后壁", "后面", (MIN,), "方位"),
    LexEntry("彼间", "那间", (MIN,), "方位"),
    LexEntry("这间", "这间", (ALL,), "方位"),
    LexEntry("这嘎达", "这里", (DONGBEI,), "方位"),
    LexEntry("那嘎达", "那里", (DONGBEI,), "方位"),
    LexEntry("旮旯", "角落", (DONGBEI, JIN, ZHONGYUAN), "方位"),
    LexEntry("此地", "这里", (WU,), "方位"),
    LexEntry("迭个", "这个", (WU,), "方位"),
    LexEntry("搿搭", "这里", (WU,), "方位"),
    LexEntry("埃搭", "那里", (WU,), "方位"),
    LexEntry("搭界", "相邻", (WU,), "方位"),
    LexEntry("跟前", "旁边", (ZHONGYUAN, JIN), "方位"),
    LexEntry("边边", "旁边", (XINAN,), "方位"),
    LexEntry("边边上", "旁边", (XINAN,), "方位"),
    LexEntry("对门", "对面", (XINAN, ZHONGYUAN), "方位"),
    LexEntry("街沿", "人行道", (XINAN, WU), "方位"),

    # ---------- 西南官话「头」后缀（本次重点方言，单列） ----------
    # 「X头」= X 里面/上面，是西南官话极高频的方位后缀。
    # 只收录地址场景真实会出现的搭配，不做通用规则——
    # 通用规则会误伤"里头"以外的正常词（如"街头""里头"歧义）。
    LexEntry("巷巷头", "巷子里", (XINAN,), "方位"),
    LexEntry("巷巷", "巷子", (XINAN,), "方位"),
    LexEntry("院子头", "院子里", (XINAN,), "方位"),
    LexEntry("楼底下", "楼下", (XINAN, DONGBEI), "方位"),
    LexEntry("上头", "上面", (XINAN, XIANG), "方位"),
    LexEntry("下头", "下面", (XINAN, XIANG), "方位"),
    LexEntry("后头", "后面", (XINAN, XIANG), "方位"),
    LexEntry("前头", "前面", (XINAN, XIANG), "方位"),
    LexEntry("对过", "对面", (WU, JIANGHUAI), "方位"),

    # ---------- 建筑 / 量词 ----------
    LexEntry("大厦", "大厦", (YUE,), "建筑"),
    LexEntry("商场", "商场", (ALL,), "建筑"),
    LexEntry("街市", "菜市场", (YUE,), "建筑"),
    # 只做繁→简（棟→栋），不做同义改写（幢→栋）：
    # 归一化阶段改写量词会丢信息，"幢/栋/座"的等价关系应在评测阶段定义。
    LexEntry("棟", "栋", (YUE,), "量词"),

    # ---------- 语气助词（删除） ----------
    # 这些词不携带地址信息，留着只会干扰后续的结构解析。
    # 只删**方言特有**的，不碰普通话里也有的（如"吧""吗"），
    # 更不碰"那个"——它常常是地标的前导词，删了会丢信息。
    LexEntry("嘅", "", (YUE,), "助词"),
    LexEntry("咗", "", (YUE,), "助词"),
    LexEntry("㗎", "", (YUE,), "助词"),
    LexEntry("喎", "", (YUE,), "助词"),
    LexEntry("嘞", "", (YUE, WU), "助词"),
    LexEntry("咯", "", (YUE, XINAN), "助词"),
    LexEntry("噻", "", (XINAN,), "助词"),
    LexEntry("撒", "", (XINAN,), "助词"),
    LexEntry("嘛", "", (XINAN, ZHONGYUAN), "助词"),
    LexEntry("哈", "", (XINAN, DONGBEI), "助词"),
    LexEntry("呗", "", (DONGBEI,), "助词"),
    LexEntry("嗯呐", "", (DONGBEI,), "助词"),
    LexEntry("啰", "", (YUE, XINAN), "助词"),

    # ---------- 人称（影响"我家"这类的解析） ----------
    LexEntry("阿拉", "我们", (WU,), "人称"),
    LexEntry("侬", "你", (WU,), "人称"),
    LexEntry("俺", "我", (ZHONGYUAN, JIN), "人称"),
    LexEntry("咱", "我们", (DONGBEI, ZHONGYUAN), "人称"),
    LexEntry("我哋", "我们", (YUE,), "人称"),
    LexEntry("你哋", "你们", (YUE,), "人称"),
]


# 常见方言异体字 → 标准字（单字级，独立于词表）
CHAR_VARIANTS: dict[str, str] = {
    "冇": "没",
    "唔": "不",
    "係": "是",
    "睇": "看",
    "喺": "在",
    "嘢": "东西",
    "咁": "这么",
    "啲": "些",
    "攞": "拿",
    "揾": "找",
}


@dataclass
class Substitution:
    """一次替换记录，用于可解释性和调试。"""

    src: str
    dst: str
    pos: int
    category: str
    groups: tuple[str, ...]


def _entries_for(groups: tuple[str, ...] | None) -> list[LexEntry]:
    """挑出适用于给定方言组的词条。

    groups=None 表示**全量**——不知道说话人方言时的正确默认：
    漏掉一条方言词会让地址解析直接错（不可恢复），
    而多收一条的风险很低，因为这些词形在标准中文里基本不出现。
    """
    if groups is None:
        return LEXICON
    want = set(groups) | {ALL}
    return [e for e in LEXICON if set(e.groups) & want]


def normalize(
    text: str,
    groups: tuple[str, ...] | None = None,
    apply_char_variants: bool = True,
) -> tuple[str, list[Substitution]]:
    """方言文本 → 标准中文。

    返回 (归一化后文本, 替换记录列表)。
    替换记录保留下来是有意的：出问题时要能回答"这个字是怎么变的"，
    而不是只给一个黑箱结果。

    用**最长匹配优先**：先按词形长度降序排，避免"屋企"被"屋"抢先匹配。
    """
    entries = sorted(_entries_for(groups), key=lambda e: -len(e.dialect))
    subs: list[Substitution] = []
    out = text

    for e in entries:
        if e.dialect not in out:
            continue
        # 记录首次出现位置用于溯源；replace 一次性替换全部同形
        pos = out.find(e.dialect)
        n = out.count(e.dialect)
        out = out.replace(e.dialect, e.standard)
        for _ in range(n):
            subs.append(
                Substitution(e.dialect, e.standard, pos, e.category, e.groups)
            )

    if apply_char_variants:
        for src, dst in CHAR_VARIANTS.items():
            if src in out:
                pos = out.find(src)
                n = out.count(src)
                out = out.replace(src, dst)
                for _ in range(n):
                    subs.append(Substitution(src, dst, pos, "异体字", (ALL,)))

    # 删除助词后可能留下多余空白
    out = re.sub(r"\s+", "", out)
    return out, subs


def has_dialect_markers(text: str, groups: tuple[str, ...] | None = None) -> bool:
    """文本里是否含方言词——可用作方言检测的辅助信号。"""
    return any(e.dialect in text for e in _entries_for(groups))


def detect_groups(text: str) -> list[tuple[str, int]]:
    """按命中的方言词数量猜测方言组，返回 [(方言组, 命中数)] 降序。

    这是**文本层**的方言线索，与 ASR 自带的**声学层**语种识别互为补充。
    两者一致时置信度高；不一致时应保守处理（走全量词表 + 声学兜底）。
    """
    counts: dict[str, int] = {}
    for e in LEXICON:
        if e.dialect and e.dialect in text:
            for g in e.groups:
                if g != ALL:
                    counts[g] = counts.get(g, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def explain(text: str, groups: tuple[str, ...] | None = None) -> str:
    """人类可读的归一化过程，用于调试和演示。"""
    out, subs = normalize(text, groups)
    lines = [f"原文: {text}", f"归一: {out}"]
    if subs:
        lines.append("替换:")
        seen: set[tuple[str, str]] = set()
        for s in subs:
            key = (s.src, s.dst)
            if key in seen:
                continue
            seen.add(key)
            tgt = s.dst if s.dst else "（删除）"
            lines.append(f"  {s.src} → {tgt}   [{s.category}] {'/'.join(s.groups)}")
    else:
        lines.append("替换: 无")
    hits = detect_groups(text)
    if hits:
        lines.append("方言线索: " + ", ".join(f"{g}×{n}" for g, n in hits))
    return "\n".join(lines)
