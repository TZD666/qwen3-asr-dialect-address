"""数字归一化的回归用例。纯 assert，`python tests/test_normalize.py` 直接跑，装了 pytest 也能收。

背景：2026-09-09 上海话录音，ASR 吐出「八百一百八号」（听错，不是合法数字），
旧解析器按 800+100+8 算成 908号，自信地把门牌编错了。门牌绝不猜，非法串必须原样保留。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialect_addr.normalize import cn_to_int, extract_tail, normalize_numbers  # noqa: E402

# 非法数串 → None（位值不递减 / 万重复 / 非零数字相邻 / 百千前没数字）
INVALID = ["八百一百八", "一百八百", "八八百", "三十十", "一万万", "二十百", "百八", "三十二五", "八零百"]

# 既有行为必须原样保住
KEEP = {
    "十八": 18, "三十": 30, "二十五": 25, "一百一十六": 116, "七百七十七": 777,
    "三百二十六": 326, "两千六百": 2600, "一百零八": 108, "八百零八": 808,
    "一千零五十": 1050, "十万": 100000, "三万五千": 35000, "两百": 200,
    # 逐位读法
    "二零一": 201, "幺零幺": 101, "二零零八": 2008, "五零二": 502,
}

# 口语补位：单位后面直接跟一个数字且没有「零」，按下一位算
FILL = {"八百八": 880, "一百五": 150, "两千六": 2600, "一万五": 15000, "三千五": 3500}


def test_invalid_returns_none():
    for s in INVALID:
        assert cn_to_int(s) is None, f"{s!r} 应判非法，实得 {cn_to_int(s)}"


def test_keep_existing():
    for s, n in KEEP.items():
        assert cn_to_int(s) == n, f"{s!r} 期望 {n}，实得 {cn_to_int(s)}"


def test_colloquial_fill():
    for s, n in FILL.items():
        assert cn_to_int(s) == n, f"{s!r} 期望 {n}，实得 {cn_to_int(s)}"


def test_invalid_house_no_is_preserved_not_guessed():
    raw = "喏，侬听好，上海市静安区南京西路八百一百八号"
    norm = normalize_numbers(raw)
    assert norm == raw, f"非法门牌应原样保留，实得 {norm!r}"
    assert extract_tail(norm).house_no == "", "非法门牌不能被抽成 house_no"


def test_fill_in_address():
    assert normalize_numbers("上海市静安区南京西路八百八号") == "上海市静安区南京西路880号"
    assert normalize_numbers("民权路十八号三单元二零一") == "民权路18号3单元201室"
    assert normalize_numbers("新南路二十六号附一号") == "新南路26号附1号"


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
