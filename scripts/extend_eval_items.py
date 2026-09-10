#!/usr/bin/env python3
"""给 data/eval/xinan_guanhua.json 的每个 item 补 §2.1 的分片字段，并自动算 gold_in_db。

    .venv/bin/python scripts/extend_eval_items.py            # 原地更新
    .venv/bin/python scripts/extend_eval_items.py --check    # 只校验 gold_in_db 是否与重算一致

已有字段全部保留；已手填的 sub_dialect / noise 等不覆盖，只补缺的。
gold_in_db 每次都重算（它是派生量，不许手填）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr.address_db import AddressDB  # noqa: E402
from stages import address_depth_of, gold_chain  # noqa: E402

SUB_DIALECT = {"成都": "成都话", "重庆": "重庆话", "贵阳": "贵阳话", "昆明": "昆明话"}


def extend(items: list[dict], db: AddressDB) -> list[str]:
    changes = []
    for it in items:
        g = gold_chain(db, it["fields"])
        defaults = {
            "dialect_group": "官话",
            "sub_dialect": SUB_DIALECT.get(it.get("city", ""), it.get("city", "")),
            "address_depth": address_depth_of(it["spoken"], it["fields"]),
            "noise": "clean",
            "orthography_expected": "simplified",
            "negative_type": None,
            "split": "eval",
        }
        for k, v in defaults.items():
            if k not in it:
                it[k] = v
                changes.append(f"{it['id']}: +{k}={v}")
        if it.get("gold_in_db") != g.in_db:
            changes.append(f"{it['id']}: gold_in_db {it.get('gold_in_db')} → {g.in_db}")
            it["gold_in_db"] = g.in_db
        if "transcript_gold" not in it:
            it["transcript_gold"] = it["spoken"]     # 朗读稿就是逐字稿
    return changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "data" / "eval" / "xinan_guanhua.json"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = Path(a.eval)
    data = json.loads(path.read_text(encoding="utf-8"))
    db = AddressDB.default()
    if a.check:
        bad = []
        for it in data["items"]:
            g = gold_chain(db, it["fields"])
            if it.get("gold_in_db") != g.in_db:
                bad.append(f"{it['id']}: 文件 {it.get('gold_in_db')} 重算 {g.in_db}")
            if it.get("address_depth") != address_depth_of(it["spoken"], it["fields"]):
                bad.append(f"{it['id']}: address_depth 文件 {it.get('address_depth')} 重算 {address_depth_of(it['spoken'], it['fields'])}")
        if bad:
            print("\n".join(bad))
            sys.exit(1)
        print(f"ok: {len(data['items'])} 条 gold_in_db / address_depth 与重算一致")
        return
    changes = extend(data["items"], db)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"更新 {len(changes)} 处 → {path}")
    for c in changes[:60]:
        print("  " + c)


if __name__ == "__main__":
    main()
