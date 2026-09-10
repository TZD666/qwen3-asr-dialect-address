"""合成扰动集的回归用例。纯 assert，`python tests/test_perturbed.py` 直接跑，装了 pytest 也能收。

在进程内用真实评测集跑一遍 scripts/gen_perturbed.py 的核心函数，然后校验三件事：

1. 每条 applied 都能回放——替换字的拼音确实落在所标注规则的混淆对上（设计文档 §2.4 的验收条件）
2. 扰动只落在地名片段上——数字、门牌尾部字、地名末尾类型后缀一个都没动
3. ground_truth 原样不变——扰动的是"听错的文本"，不是真值
"""

from __future__ import annotations

import functools
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_perturbed import (  # noqa: E402
    DEFAULT_PER_ITEM,
    DEFAULT_SEED,
    MIN_TOTAL,
    build_char_table,
    generate,
    load_items,
    perturbable_positions,
    read_jsonl,
    verify_lines,
    write_jsonl,
)

PER_ITEM = DEFAULT_PER_ITEM  # 跟实际产出的那份保持一致（24×30=720 ≥ 500）


@functools.lru_cache(maxsize=1)
def _generated() -> tuple[tuple[dict, ...], dict]:
    """生成一份写到临时目录再读回来，顺带覆盖 write_jsonl / read_jsonl。"""
    items = load_items()
    table = build_char_table(items)
    lines = generate(items, table, per_item=PER_ITEM, seed=DEFAULT_SEED)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "perturbed.jsonl"
        write_jsonl(path, lines)
        lines = read_jsonl(path)
    return tuple(lines), {it["id"]: it for it in items}


def test_total_is_enough():
    lines, items = _generated()
    assert len(lines) >= MIN_TOTAL, f"总量 {len(lines)} < {MIN_TOTAL}"
    assert len(lines) == len(items) * PER_ITEM, "每句都该产出满额变体"


def test_every_line_replays():
    lines, _ = _generated()
    counts, errors = verify_lines(list(lines))
    assert not errors, "不可回放：\n" + "\n".join(errors[:10])
    assert len(counts) >= 10, f"规则覆盖太少：{dict(counts)}"


def test_only_geo_name_chars_changed():
    """逐位比对：与源句不同的位置必须都在可扰动的地名片段内。

    这一条同时守住了"数字不动"和"门牌尾部字不动"——它们本来就被
    perturbable_positions 排除在外。
    """
    lines, items = _generated()
    for line in lines:
        src = items[line["source_id"]]
        a, b = src["spoken"], line["spoken"]
        assert len(a) == len(b), f"{line['id']}: 句长变了（替换必须一对一）"
        diff = {i for i, (x, y) in enumerate(zip(a, b)) if x != y}
        assert diff, f"{line['id']}: 与源句完全相同，不是扰动"
        allowed = set(perturbable_positions(src))
        assert diff <= allowed, (
            f"{line['id']}: 位置 {sorted(diff - allowed)} 不在地名片段内 "
            f"（{''.join(a[i] for i in sorted(diff - allowed))}）"
        )
        assert len(diff) == line["n_subs"], f"{line['id']}: 改动位数与 n_subs 不符"


def test_ground_truth_untouched():
    lines, items = _generated()
    for line in lines:
        src = items[line["source_id"]]
        assert line["ground_truth"] == src["ground_truth"], f"{line['id']}: 真值被改了"
        assert line["fields"] == src.get("fields"), f"{line['id']}: fields 被改了"
        assert line["city"] == src.get("city"), f"{line['id']}: city 被改了"


def test_schema():
    lines, _ = _generated()
    required = ("id", "source_id", "rule", "spoken", "ground_truth", "fields",
                "difficulty", "city", "applied", "n_subs", "split", "negative_type")
    seen_ids = set()
    for line in lines:
        for key in required:
            assert key in line, f"{line.get('id')}: 缺字段 {key}"
        assert line["split"] == "eval"
        assert line["id"].startswith(line["source_id"] + "-")
        assert line["id"] not in seen_ids, f"id 重复：{line['id']}"
        seen_ids.add(line["id"])
    spokens = [ln["spoken"] for ln in lines]
    assert len(set(spokens)) == len(spokens), "存在重复的 spoken，去重失效"


def test_deterministic():
    items = load_items()
    table = build_char_table(items)
    a = generate(items, table, per_item=PER_ITEM, seed=DEFAULT_SEED)
    b = generate(items, table, per_item=PER_ITEM, seed=DEFAULT_SEED)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False), (
        "同一 seed 必须给出同样的结果"
    )


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
