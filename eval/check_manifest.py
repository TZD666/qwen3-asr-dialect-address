#!/usr/bin/env python3
"""校验 data/eval/manifest.jsonl 与 data/eval/xinan_guanhua.json（评测体系设计 §10 步骤 1）。

    python eval/check_manifest.py            # 通过返回 0；默认容忍 draft/pending 行，只统计
    python eval/check_manifest.py --strict   # 要求 quality=ok 的行全部 confirmed（对外报数前用）

检查项
  1. recordings/ 下每个文件都有清单行，清单里的每行文件都存在
  2. quality=ok 且 label_status ∈ {confirmed, draft} 的行：transcript_gold / address_gold 非空（has_address=false 的除外）
  3. gold_in_db 与脚本重算一致（清单与评测集都查）
  4. address_depth / noise / dialect_group 取值合法
  5. split 合法；calib/train 在样本量达门槛前应为空
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr.address_db import AddressDB  # noqa: E402
from stages import gold_chain  # noqa: E402

REC = ROOT / "data" / "eval" / "audio" / "recordings"
MANIFEST = ROOT / "data" / "eval" / "manifest.jsonl"
EVAL = ROOT / "data" / "eval" / "xinan_guanhua.json"

DIALECT_GROUPS = {"官话", "粤", "闽", "吴", "湘", "赣", "客", ""}
DEPTHS = {"full", "district", "street_only", "none", ""}
NOISES = {"clean", "filler", "complaint", ""}
QUALITIES = {"ok", "silent", "clipped", "unknown_format", "duplicate"}
STATUS = {"confirmed", "draft", "pending", "n/a"}


def load_manifest(path: Path = MANIFEST) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def check(strict: bool = False) -> list[str]:
    errs: list[str] = []
    db = AddressDB.default()
    rows = load_manifest()
    files_disk = {p.name for p in REC.iterdir() if p.is_file() and not p.name.startswith(".")}
    files_man = {Path(r["file"]).name for r in rows}
    for f in sorted(files_disk - files_man):
        errs.append(f"录音无清单行: {f}")
    for f in sorted(files_man - files_disk):
        errs.append(f"清单行无文件: {f}")

    for r in rows:
        f = r["file"]
        if r.get("quality") not in QUALITIES:
            errs.append(f"{f}: quality 非法 {r.get('quality')}")
        if r.get("label_status") not in STATUS:
            errs.append(f"{f}: label_status 非法 {r.get('label_status')}")
        if r.get("dialect_group", "") not in DIALECT_GROUPS:
            errs.append(f"{f}: dialect_group 非法 {r.get('dialect_group')}")
        if r.get("address_depth", "") not in DEPTHS:
            errs.append(f"{f}: address_depth 非法 {r.get('address_depth')}")
        if r.get("noise", "") not in NOISES:
            errs.append(f"{f}: noise 非法 {r.get('noise')}")
        if r.get("split") not in ("eval", "calib", "train"):
            errs.append(f"{f}: split 非法 {r.get('split')}")
        if r.get("quality") != "ok":
            continue
        st = r.get("label_status")
        if strict and st != "confirmed":
            errs.append(f"{f}: 未确认（{st}）")
        if st in ("confirmed", "draft"):
            if not r.get("transcript_gold"):
                errs.append(f"{f}: transcript_gold 为空")
            if r.get("has_address", True):
                if not r.get("address_gold"):
                    errs.append(f"{f}: address_gold 为空")
                if not r.get("fields_gold"):
                    errs.append(f"{f}: fields_gold 为空")
                else:
                    g = gold_chain(db, r["fields_gold"])
                    if r.get("gold_in_db") != g.in_db:
                        errs.append(f"{f}: gold_in_db 文件 {r.get('gold_in_db')} 重算 {g.in_db}")
                    # 规范地址应包含真值字段里的每个非空值（字段与地址串要自洽）
                    for k, v in r["fields_gold"].items():
                        if v and v not in r["address_gold"] and k not in ("province", "city"):
                            errs.append(f"{f}: fields_gold.{k}=「{v}」不在 address_gold 里")

    data = json.loads(EVAL.read_text(encoding="utf-8"))
    for it in data["items"]:
        g = gold_chain(db, it["fields"])
        if it.get("gold_in_db") != g.in_db:
            errs.append(f"item {it['id']}: gold_in_db 文件 {it.get('gold_in_db')} 重算 {g.in_db}")
        for k in ("dialect_group", "address_depth", "noise", "orthography_expected", "split"):
            if k not in it:
                errs.append(f"item {it['id']}: 缺字段 {k}")

    # 划分完整性：合成集按 source_id 不重不漏；自发录音在门槛前全部 eval
    from splits import check_synthetic_partition, load_splits, split_of
    errs.extend(check_synthetic_partition([it["id"] for it in data["items"]]))
    sp = load_splits()
    n_spont = sum(1 for r in rows if r.get("quality") == "ok" and r.get("set", "spontaneous") == "spontaneous")
    if n_spont <= sp.get("recordings", {}).get("spontaneous_threshold", 150):
        for r in rows:
            if r.get("quality") == "ok" and split_of("recording", r["file"]) != "eval":
                errs.append(f"{r['file']}: 自发录音未达 150 条前应全部为 eval")

    ok_rows = [r for r in rows if r.get("quality") == "ok"]
    print(f"清单 {len(rows)} 行，quality=ok {len(ok_rows)}，"
          f"label_status {dict(Counter(r.get('label_status') for r in ok_rows))}")
    print(f"评测集 {len(data['items'])} 条，gold_in_db=true {sum(1 for it in data['items'] if it.get('gold_in_db'))}")
    return errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    errs = check(a.strict)
    if errs:
        print(f"\n{len(errs)} 处问题:")
        for e in errs:
            print("  " + e)
        sys.exit(1)
    print("check_manifest: 通过")
