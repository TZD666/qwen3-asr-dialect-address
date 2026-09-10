#!/usr/bin/env python3
"""合成文本扰动集生成器（对应《评测体系设计》§2.4）。

做什么
------
拿 `src/dialect_addr/pinyin_dialect.py` 里的声母混淆表 (INITIAL_CONFUSIONS) 和
韵母混淆表 (FINAL_CONFUSIONS) **反向**用：既然这两张表描述了"方言里哪些音会
混"，那就按同样的规则给 24 句朗读稿的地名片段挑同音/近音字替换，合成出成千条
带标注的"ASR 听错了"文本。每条都记下用了哪条规则、换了哪个字、前后拼音，
可以逐条回放校验。

    真值      重庆渝中区解放碑民权路十八号
    扰动      重庆渝中区解放碑民全路十八号     rule=同音异形  权(quan)→全(quan)
    扰动      重庆渝中区解放碑民权路十八号     rule=见系不颚化 交(jiao)→高(gao)（示意）

扰动只落在地名片段上：数字（一二三…幺零两）、门牌尾部字（号栋幢座单元室楼层附）
和地名末尾的类型后缀（区市省路街道巷段镇村/小区花园广场大道街道）一律不动——
ASR 极少把这些听错，动了反而不像真实错误。多音字（重庆的"重"读 chong 而单字读
zhong）也跳过，避免标注的字级拼音与句中实际读音打架。

适用边界（重要，报告里必须写明）
--------------------------------
**这个集合是用本项目自己的代价矩阵生成的，所以不能用来验证代价矩阵本身。**
用 A 生成的数据去证明 A 是对的，是循环论证。它只能测**下游**环节：

    能测：候选排序、置信度闸门、门牌尾部解析、省市区层级补全
    不能测：加权拼音距离/混淆表的音学正确性（那要靠真实录音，见 §3 音频集）

用法
----
    python scripts/gen_perturbed.py                    # 生成，默认 24×30 条
    python scripts/gen_perturbed.py --per-item 50      # 每句 50 个变体
    python scripts/gen_perturbed.py --verify           # 回放校验已有文件
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from pypinyin import Style, lazy_pinyin
    from pypinyin.constants import PHRASES_DICT
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 pypinyin：pip install pypinyin") from exc

from dialect_addr.pinyin_dialect import (  # noqa: E402
    FINAL_CONFUSIONS,
    INITIAL_CONFUSIONS,
)

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

DEFAULT_SEED = 20260910
DEFAULT_PER_ITEM = 30
PER_ITEM_MIN, PER_ITEM_MAX = 20, 50
MIN_TOTAL = 500

SRC_EVAL = ROOT / "data" / "eval" / "xinan_guanhua.json"
SRC_DB = ROOT / "data" / "addresses" / "cn_subset.json"
OUT_DIR = ROOT / "data" / "eval" / "synthetic"

HOMOPHONE = "同音异形"
COMPOUND = "复合"
SHARE_HOMOPHONE = 0.35  # 同音异形是最常见的 ASR 错误，占比最大
SHARE_COMPOUND = 0.10   # 一句里两处不同规则的错
MAX_CAND_PER_RULE = 6   # 每个位置每条规则最多留几个候选字（取最常用的）
COMMON_TOP_N = 3500     # 常用字代理表规模

HAN = re.compile(r"[一-龥]")
HAN_RUN = re.compile(r"[一-龥]+")

# 数字与数词：门牌号走独立的规则归一化通道，扰动它等于换了个地址，不是听错
NUM_CHARS = set("一二三四五六七八九十百千万亿零两幺〇○")
# 门牌尾部字段用字
TAIL_CHARS = set("号栋幢座单元室楼层附")
# 地名类型后缀：末尾命中就保护，长的优先
TYPE_SUFFIXES = (
    "街道", "小区", "花园", "广场", "大道",
    "区", "市", "省", "路", "街", "道", "巷", "段", "镇", "村",
)
# 从 fields 里取地名的字段顺序
NAME_FIELDS = ("district", "street", "road", "community", "city", "province")
# 从源条目原样复制过来的扩展 schema 字段（可能还不存在）
EXTRA_FIELDS = (
    "dialect_group", "sub_dialect", "address_depth", "noise",
    "orthography_expected", "gold_in_db", "negative_type", "split",
)


# --------------------------------------------------------------------------
# 规则
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """一条可反向应用的音变规则。

    name  规则中文描述（混淆表元组的第 3 项）。多条规则共享同一个名字时
          （如"前后鼻音不分"有 5 组），按规范视为同一条规则，只是组不同。
    kind  'initial' 声母规则 / 'final' 韵母规则
    group 可互换的音素集合
    """

    name: str
    kind: str
    group: frozenset[str]


def build_rules() -> list[Rule]:
    rules = [Rule(desc, "initial", grp) for grp, _c, desc, _d in INITIAL_CONFUSIONS]
    rules += [Rule(desc, "final", grp) for grp, _c, desc, _d in FINAL_CONFUSIONS]
    return rules


RULES = build_rules()
RULE_NAMES = sorted({r.name for r in RULES})
# 名字 -> 同名的所有规则（校验时任一组满足即算通过）
RULES_BY_NAME: dict[str, list[Rule]] = {}
for _r in RULES:
    RULES_BY_NAME.setdefault(_r.name, []).append(_r)


# --------------------------------------------------------------------------
# 拼音
# --------------------------------------------------------------------------


def _norm_final(f: str) -> str:
    """与 pinyin_dialect._norm_final 保持一致，消除 pypinyin 书写变体。"""
    return f.replace("ü", "v").replace("ǖ", "v").replace("ń", "n")


@functools.lru_cache(maxsize=65536)
def char_pinyin(ch: str) -> tuple[str, str, str] | None:
    """单字 -> (声母, 韵母, 完整拼音)，全部不含声调。

    风格与 pinyin_dialect.to_syllables 完全一致：INITIALS/FINALS 都用
    strict=False，韵母做 ü→v 归一。不一致会导致标注的规则与项目内部的
    音节分解对不上，扰动集就白做了。
    """
    if not HAN.fullmatch(ch):
        return None
    full = lazy_pinyin(ch, style=Style.NORMAL, errors="ignore")
    init = lazy_pinyin(ch, style=Style.INITIALS, strict=False, errors="ignore")
    fin = lazy_pinyin(ch, style=Style.FINALS, strict=False, errors="ignore")
    if not full or not fin:
        return None
    # 零声母时 INITIALS 返回 ''，errors='ignore' 会把它当空结果丢掉
    initial = init[0] if init else ""
    return initial, _norm_final(fin[0]), full[0]


def context_pinyin(text: str) -> dict[int, str]:
    """位置 -> 该字在**句中**的拼音。

    按汉字连续段调用 lazy_pinyin，保证下标对齐（整串调用会因为非汉字被
    合并/丢弃而错位）。用来筛掉多音字：单字读 zhong 的"重"在"重庆"里读
    chong，若按单字读音去挑替换字，生成的错字在音上根本不成立。
    """
    out: dict[int, str] = {}
    for m in HAN_RUN.finditer(text):
        run = m.group()
        pys = lazy_pinyin(run, style=Style.NORMAL, errors="ignore")
        if len(pys) != len(run):  # pragma: no cover - 全汉字段不该发生
            continue
        for k, py in enumerate(pys):
            out[m.start() + k] = py
    return out


# --------------------------------------------------------------------------
# 常用字池与反查表
# --------------------------------------------------------------------------


@dataclass
class CharTable:
    """替换字的反查表。

    by_full  完整拼音 -> 候选字（同音异形用）
    by_if    (声母, 韵母) -> 候选字（声母/韵母规则用）
    freq     字频（用于优先挑常用字，让错字像真实 ASR 输出）
    desc     字池来源说明，写进 gen_config.json
    size     字池大小
    """

    by_full: dict[str, list[str]]
    by_if: dict[tuple[str, str], list[str]]
    freq: Counter
    desc: list[str]
    size: int


def _chars_of(obj: object, sink: set[str]) -> None:
    """递归收集 JSON 结构里的所有汉字。"""
    if isinstance(obj, str):
        sink.update(HAN.findall(obj))
    elif isinstance(obj, list):
        for x in obj:
            _chars_of(x, sink)
    elif isinstance(obj, dict):
        for v in obj.values():
            _chars_of(v, sink)


def phrase_freq() -> Counter:
    """用 pypinyin 自带词库统计字频，作为"常用字"的离线代理。

    完全离线、不引新依赖。一个字出现在越多词条里越常用；取前 3500 个，
    量级正好对上《现代汉语常用字表》的 3500 常用字（但不是那张表本身）。
    """
    cnt: Counter = Counter()
    for phrase in PHRASES_DICT:
        for ch in phrase:
            if HAN.fullmatch(ch):
                cnt[ch] += 1
    return cnt


def build_char_table(items: list[dict]) -> CharTable:
    """字池 = 地址库用字 ∪ 24 句评测文本用字 ∪ 常用字代理表前 3500。

    刻意**不**用整个 CJK 区：随便从 U+4E00–U+9FA5 里挑同音字会挑出一堆生僻字，
    ASR 根本不会输出那些字，生成的扰动就不像真实错误了。
    """
    freq = phrase_freq()

    db_chars: set[str] = set()
    db = json.loads(SRC_DB.read_text(encoding="utf-8"))
    for entry in db.get("entries", []):
        _chars_of(entry.get("name", ""), db_chars)
        _chars_of(entry.get("aliases", []), db_chars)

    eval_chars: set[str] = set()
    for it in items:
        _chars_of(it.get("spoken", ""), eval_chars)
        _chars_of(it.get("ground_truth", ""), eval_chars)

    common = {
        c for c, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:COMMON_TOP_N]
    }

    pool = db_chars | eval_chars | common
    by_full: dict[str, list[str]] = {}
    by_if: dict[tuple[str, str], list[str]] = {}
    kept = 0
    for ch in sorted(pool):
        py = char_pinyin(ch)
        if py is None:
            continue
        initial, final, full = py
        kept += 1
        by_full.setdefault(full, []).append(ch)
        by_if.setdefault((initial, final), []).append(ch)

    # 常用字排前面：挑替换字时只取前几个，保证错字是常见字
    def rank(ch: str) -> tuple[int, str]:
        return (-freq.get(ch, 0), ch)

    for lst in by_full.values():
        lst.sort(key=rank)
    for lst in by_if.values():
        lst.sort(key=rank)

    desc = [
        f"地址库 {SRC_DB.relative_to(ROOT)} 的 name+aliases 用字：{len(db_chars)} 字",
        f"评测集 {SRC_EVAL.relative_to(ROOT)} 的 spoken+ground_truth 用字：{len(eval_chars)} 字",
        f"常用字离线代理（pypinyin PHRASES_DICT 词条覆盖数前 {COMMON_TOP_N}）：{len(common)} 字"
        "（无官方 3500 常用字表可用，也不联网，故用词库字频代理，仅取常用字、避开生僻字）",
        f"并集去重并过滤掉 pypinyin 拿不到拼音的字后：{kept} 字",
    ]
    return CharTable(by_full=by_full, by_if=by_if, freq=freq, desc=desc, size=kept)


# --------------------------------------------------------------------------
# 可扰动位置
# --------------------------------------------------------------------------


def trailing_suffix(name: str) -> str:
    """返回地名末尾命中的类型后缀（没有则空串），长后缀优先。"""
    for suf in TYPE_SUFFIXES:
        if len(name) > len(suf) and name.endswith(suf):
            return suf
    return ""


def surface_forms(val: str) -> list[str]:
    """地名在口语里的可能写法。

    朗读稿常把类型后缀省掉（fields.street="解放碑街道"，口语只说"解放碑"），
    所以完整形式找不到时退一步试去掉后缀的形式。
    """
    if not val:
        return []
    forms = [val]
    suf = trailing_suffix(val)
    if suf:
        forms.append(val[: -len(suf)])
    return forms


def name_spans(item: dict) -> list[tuple[int, int, str]]:
    """在 spoken 里定位到的地名片段 [(start, end, surface), ...]。"""
    spoken = item.get("spoken", "")
    fields = item.get("fields") or {}
    spans: list[tuple[int, int, str]] = []
    for key in NAME_FIELDS:
        val = fields.get(key) or ""
        for surface in surface_forms(val):
            hits = []
            start = spoken.find(surface)
            while start >= 0:
                hits.append((start, start + len(surface), surface))
                start = spoken.find(surface, start + 1)
            if hits:  # 长形式命中就不再试短形式
                spans.extend(hits)
                break
    return spans


def perturbable_positions(item: dict) -> list[int]:
    """可以被扰动的字位置：地名片段内、非数字、非门牌尾部字、非末尾类型后缀。"""
    spoken = item.get("spoken", "")
    ok: set[int] = set()
    for start, end, surface in name_spans(item):
        suf = trailing_suffix(surface)
        limit = end - len(suf) if suf else end
        for i in range(start, min(end, limit)):
            ch = spoken[i]
            if not HAN.fullmatch(ch):
                continue
            if ch in NUM_CHARS or ch in TAIL_CHARS:
                continue
            ok.add(i)
    return sorted(ok)


def usable_positions(item: dict) -> list[int]:
    """可扰动位置里再筛掉多音字（句中读音≠单字读音）。"""
    spoken = item.get("spoken", "")
    ctx = context_pinyin(spoken)
    out = []
    for i in perturbable_positions(item):
        py = char_pinyin(spoken[i])
        if py is None:
            continue
        if ctx.get(i) != py[2]:  # 多音字，字级标注会与句中读音矛盾
            continue
        out.append(i)
    return out


# --------------------------------------------------------------------------
# 候选替换字
# --------------------------------------------------------------------------


def candidates(ch: str, table: CharTable) -> dict[str, list[str]]:
    """某个字在各条规则下的替换候选：规则名 -> 候选字（已按字频截断）。"""
    py = char_pinyin(ch)
    if py is None:
        return {}
    initial, final, full = py
    out: dict[str, list[str]] = {}

    # 同音异形：完整拼音完全相同的另一个字
    same = [c for c in table.by_full.get(full, []) if c != ch][:MAX_CAND_PER_RULE]
    if same:
        out[HOMOPHONE] = same

    for rule in RULES:
        if rule.kind == "initial":
            if initial not in rule.group:
                continue
            keys = [(o, final) for o in sorted(rule.group) if o != initial]
        else:
            if final not in rule.group:
                continue
            keys = [(initial, o) for o in sorted(rule.group) if o != final]
        cands: list[str] = []
        for key in keys:
            cands += [c for c in table.by_if.get(key, []) if c != ch]
        cands = sorted(set(cands), key=lambda c: (-table.freq.get(c, 0), c))
        if cands:
            out.setdefault(rule.name, [])
            out[rule.name] += cands[:MAX_CAND_PER_RULE]
    return out


# --------------------------------------------------------------------------
# 生成
# --------------------------------------------------------------------------


def apply_subs(spoken: str, subs: list[tuple[int, str]]) -> tuple[str, list[dict]]:
    """把 [(位置, 替换字)] 应用到句子上，返回新句子与 applied 标注。"""
    chars = list(spoken)
    applied = []
    for pos, to_ch in sorted(subs):
        from_ch = spoken[pos]
        chars[pos] = to_ch
        applied.append({
            "from": from_ch,
            "to": to_ch,
            "pinyin_from": char_pinyin(from_ch)[2],
            "pinyin_to": char_pinyin(to_ch)[2],
        })
    return "".join(chars), applied


def _line(item: dict, idx: int, rule: str, spoken: str, applied: list[dict]) -> dict:
    """组装一条输出。扩展 schema 字段原样带过来，缺的按规范补默认值。"""
    out = {
        "id": f"{item['id']}-{idx:03d}",
        "source_id": item["id"],
        "rule": rule,
        "spoken": spoken,
        "ground_truth": item["ground_truth"],
        "city": item.get("city", ""),
        "fields": item.get("fields", {}),
        "difficulty": item.get("difficulty", []),
        "applied": applied,
        "n_subs": len(applied),
    }
    for key in EXTRA_FIELDS:
        if key in item:
            out[key] = item[key]
    out.setdefault("split", "eval")
    out.setdefault("negative_type", None)
    return out


def generate_for_item(
    item: dict, table: CharTable, per_item: int, rng: random.Random, seen: set[str]
) -> list[dict]:
    """给一条朗读稿生成 per_item 个变体（去重后可能不足）。"""
    spoken = item["spoken"]
    positions = usable_positions(item)

    # 规则名 -> 该句所有可用的单字替换 [(位置, 替换字)]
    pool: dict[str, list[tuple[int, str]]] = {}
    for pos in positions:
        for rule_name, cands in candidates(spoken[pos], table).items():
            pool.setdefault(rule_name, []).extend((pos, c) for c in cands)
    for lst in pool.values():
        rng.shuffle(lst)
    if not pool:
        return []

    lines: list[dict] = []

    def emit(rule: str, subs: list[tuple[int, str]]) -> bool:
        new_spoken, applied = apply_subs(spoken, subs)
        if new_spoken == spoken or new_spoken in seen:
            return False
        seen.add(new_spoken)
        lines.append(_line(item, len(lines) + 1, rule, new_spoken, applied))
        return True

    # 1) 同音异形拿最大份额
    n_homo = round(per_item * SHARE_HOMOPHONE)
    for pos, to_ch in list(pool.get(HOMOPHONE, [])):
        if sum(1 for x in lines if x["rule"] == HOMOPHONE) >= n_homo:
            break
        emit(HOMOPHONE, [(pos, to_ch)])

    # 2) 复合：两个不同位置 + 两条不同规则
    n_comp = round(per_item * SHARE_COMPOUND)
    names = sorted(pool)
    made = 0
    for _ in range(n_comp * 40):
        if made >= n_comp or len(names) < 2:
            break
        a, b = rng.sample(names, 2)
        sa = rng.choice(pool[a])
        sb = rng.choice(pool[b])
        if sa[0] == sb[0]:
            continue
        if emit(COMPOUND, [sa, sb]):
            made += 1

    # 3) 其余份额在各条音变规则间轮转分配
    others = [n for n in names if n != HOMOPHONE]
    rng.shuffle(others)
    cursor = {n: 0 for n in others}
    while len(lines) < per_item and others:
        progressed = False
        for name in list(others):
            if len(lines) >= per_item:
                break
            i = cursor[name]
            if i >= len(pool[name]):
                others.remove(name)
                continue
            cursor[name] = i + 1
            progressed = True
            emit(name, [pool[name][i]])
        if not progressed:
            break

    # 4) 还差就再补同音异形（它的候选最多）
    for pos, to_ch in list(pool.get(HOMOPHONE, [])):
        if len(lines) >= per_item:
            break
        emit(HOMOPHONE, [(pos, to_ch)])

    # id 连号（emit 失败时不会留空洞，这里重排保证 001..N 连续）
    for k, ln in enumerate(lines, 1):
        ln["id"] = f"{item['id']}-{k:03d}"
    return lines


def generate(
    items: list[dict], table: CharTable, per_item: int, seed: int
) -> list[dict]:
    """全量生成。同一个 seed + per_item 必然给出同样的结果。"""
    rng = random.Random(seed)
    seen: set[str] = {it["spoken"] for it in items}  # 真值句本身不算扰动
    out: list[dict] = []
    for item in items:
        out.extend(generate_for_item(item, table, per_item, rng, seen))
    return out


# --------------------------------------------------------------------------
# 回放校验
# --------------------------------------------------------------------------


def _sub_matches(rule: Rule, sub: dict) -> bool:
    """一处替换是否落在这条规则的混淆组上。"""
    pf, pt = char_pinyin(sub["from"]), char_pinyin(sub["to"])
    if pf is None or pt is None:
        return False
    if pf[2] != sub["pinyin_from"] or pt[2] != sub["pinyin_to"]:
        return False
    (ia, fa), (ib, fb) = (pf[0], pf[1]), (pt[0], pt[1])
    if rule.kind == "initial":
        return fa == fb and ia != ib and ia in rule.group and ib in rule.group
    return ia == ib and fa != fb and fa in rule.group and fb in rule.group


def _is_homophone(sub: dict) -> bool:
    pf, pt = char_pinyin(sub["from"]), char_pinyin(sub["to"])
    if pf is None or pt is None:
        return False
    return (
        pf[2] == pt[2] == sub["pinyin_from"] == sub["pinyin_to"]
        and sub["from"] != sub["to"]
    )


def _matching_rule_names(sub: dict) -> set[str]:
    """这处替换能被哪些规则解释（复合行的校验要用）。"""
    names = {HOMOPHONE} if _is_homophone(sub) else set()
    for rule in RULES:
        if _sub_matches(rule, sub):
            names.add(rule.name)
    return names


def check_line(line: dict) -> str | None:
    """校验一行，通过返回 None，否则返回原因。"""
    rule = line.get("rule", "")
    applied = line.get("applied") or []
    if not applied:
        return "applied 为空"
    if line.get("n_subs") != len(applied):
        return f"n_subs={line.get('n_subs')} 与 applied 长度 {len(applied)} 不符"
    for sub in applied:
        for key in ("from", "to", "pinyin_from", "pinyin_to"):
            if not sub.get(key):
                return f"applied 缺字段 {key}"
        if len(sub["from"]) != 1 or len(sub["to"]) != 1:
            return "applied 的 from/to 必须是单字"

    if rule == HOMOPHONE:
        for sub in applied:
            if not _is_homophone(sub):
                return f"{sub['from']}({sub['pinyin_from']})→{sub['to']}({sub['pinyin_to']}) 不是同音异形"
        return None

    if rule == COMPOUND:
        if len(applied) != 2:
            return f"复合行应有 2 处替换，实得 {len(applied)}"
        s1, s2 = (_matching_rule_names(s) for s in applied)
        if not s1 or not s2:
            return "复合行里有替换无法归到任何规则"
        # 两处必须能指派给**不同**的规则
        if len(s1 | s2) < 2:
            return f"复合行两处替换只能归到同一条规则 {s1}"
        return None

    rules = RULES_BY_NAME.get(rule)
    if not rules:
        return f"未知规则 {rule!r}"
    for sub in applied:
        if not any(_sub_matches(r, sub) for r in rules):
            return (
                f"{sub['from']}({sub['pinyin_from']})→{sub['to']}({sub['pinyin_to']}) "
                f"不满足规则「{rule}」的混淆条件"
            )
    return None


def verify_lines(lines: list[dict]) -> tuple[Counter, list[str]]:
    """批量校验，返回（各规则条数, 错误列表）。"""
    counts: Counter = Counter()
    errors: list[str] = []
    for n, line in enumerate(lines, 1):
        counts[line.get("rule", "?")] += 1
        why = check_line(line)
        if why:
            errors.append(f"第 {n} 行 id={line.get('id')}: {why}")
    return counts, errors


# --------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------


def load_items(path: Path = SRC_EVAL) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


SCOPE_NOTE = (
    "适用边界：本集合由 pinyin_dialect.py 的 INITIAL_CONFUSIONS / FINAL_CONFUSIONS "
    "反向生成，扰动字是用本项目自己的代价矩阵挑出来的。因此它只能评测下游环节——"
    "候选排序、置信度闸门、门牌尾部解析、省市区层级补全；不能用来验证加权拼音距离/"
    "混淆表本身是否符合真实方言音变（用 A 生成的数据去证明 A 正确是循环论证）。"
    "音距离的正确性只能靠真实录音集（§3）评测。"
)


def write_config(
    path: Path,
    seed: int,
    per_item: int,
    total: int,
    counts: Counter,
    table: CharTable,
    zero_rules: list[str],
) -> None:
    cfg = {
        "generator": "scripts/gen_perturbed.py",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_file": str(SRC_EVAL.relative_to(ROOT)),
        "source_items": len(load_items()),
        "seed": seed,
        "per_item": per_item,
        "total": total,
        "counts_by_rule": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "rules_without_samples": zero_rules,
        "rules_without_samples_note": (
            "这些规则本次没产出样本，不是 bug：一是 24 句地名里没有落在该混淆组上的字；"
            "二是有的组成员（uei/iou/uen/ue/er）在 pypinyin strict=False 的韵母切分下"
            "根本不会出现，反向找不到替换字。换朗读稿或扩充评测集后会自然覆盖到。"
        ),
        "char_pool": {"size": table.size, "sources": table.desc},
        "protected": {
            "numbers": "".join(sorted(NUM_CHARS)),
            "tail_fields": "".join(sorted(TAIL_CHARS)),
            "type_suffixes": list(TYPE_SUFFIXES),
            "note": "数字、门牌尾部字、地名末尾类型后缀、多音字（句中读音≠单字读音）一律不扰动",
        },
        "shares": {HOMOPHONE: SHARE_HOMOPHONE, COMPOUND: SHARE_COMPOUND},
        "scope_note": SCOPE_NOTE,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="合成文本扰动集生成器（§2.4）")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--per-item", type=int, default=DEFAULT_PER_ITEM,
                    help=f"每句变体数，{PER_ITEM_MIN}–{PER_ITEM_MAX}")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--verify", action="store_true", help="回放校验已有 perturbed.jsonl")
    args = ap.parse_args(argv)

    out_jsonl = args.out_dir / "perturbed.jsonl"

    if args.verify:
        if not out_jsonl.exists():
            print(f"✗ 找不到 {out_jsonl}", file=sys.stderr)
            return 1
        lines = read_jsonl(out_jsonl)
        counts, errors = verify_lines(lines)
        print(f"校验 {out_jsonl.relative_to(ROOT)}：{len(lines)} 条")
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:5d}  {name}")
        if errors:
            print(f"\n✗ {len(errors)} 条不可回放：", file=sys.stderr)
            for e in errors[:20]:
                print(f"  {e}", file=sys.stderr)
            return 1
        print(f"\n✓ 全部 {len(lines)} 条可回放")
        return 0

    if not PER_ITEM_MIN <= args.per_item <= PER_ITEM_MAX:
        print(f"✗ --per-item 必须在 {PER_ITEM_MIN}–{PER_ITEM_MAX} 之间", file=sys.stderr)
        return 2

    items = load_items()
    table = build_char_table(items)
    lines = generate(items, table, args.per_item, args.seed)

    counts, errors = verify_lines(lines)
    assert not errors, f"自检失败：{errors[:5]}"
    assert len(lines) >= MIN_TOTAL, f"总量 {len(lines)} < {MIN_TOTAL}，加大 --per-item"

    write_jsonl(out_jsonl, lines)
    zero_rules = [n for n in RULE_NAMES if counts.get(n, 0) == 0]
    write_config(args.out_dir / "gen_config.json", args.seed, args.per_item,
                 len(lines), counts, table, zero_rules)

    print(f"字池 {table.size} 字；{len(items)} 句 × {args.per_item} → {len(lines)} 条")
    for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:5d}  {name}")
    if zero_rules:
        print(f"  （无样本的规则：{'、'.join(zero_rules)}）")
    print(f"\n→ {out_jsonl.relative_to(ROOT)}")
    print(f"→ {(args.out_dir / 'gen_config.json').relative_to(ROOT)}")
    print(f"\n{SCOPE_NOTE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
