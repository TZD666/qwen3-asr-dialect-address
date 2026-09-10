#!/usr/bin/env python3
"""无模型回归套件（评测体系设计 §7）。

跑三样东西，逐分片与 eval/golden/ 里的快照比，任一条件触发即失败：
    1. 24 句 text 模式                       golden/text.json
    2. text 模式 + oov_drop 负样本             golden/text_oov.json
    3. 合成扰动集（perturbed.jsonl）          golden/text_synthetic.json
    4. （可选 --with-injection）注入用例       golden/injection.json   ← 需要模型或 ASR 缓存

阻断规则
    - 任一分片的 exact 或 deliverable 下降
    - oov_db 子集的过度纠正数上升
    - asr_correct 子集的改坏数上升（只在 audio golden 存在时比，见 --with-audio）
    - injection 用例中新增翻转（known_flip 的不算）
    - 归因分布里出现 assembly_error

快照更新是显式动作：`regression.py --update-golden`，并在 commit message 里写明哪个分片为什么变。

    python eval/regression.py                 # 比对
    python eval/regression.py --update-golden # 刷新快照
    python eval/regression.py --with-injection --asr-cache-readonly   # 本地有缓存时连注入用例一起跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import run_eval  # noqa: E402

GOLDEN = ROOT / "eval" / "golden"
SYNTH = ROOT / "data" / "eval" / "synthetic" / "perturbed.jsonl"


def make_args(**kw) -> Namespace:
    base = dict(mode="text", eval=str(ROOT / "data" / "eval" / "xinan_guanhua.json"),
                manifest=str(run_eval.MANIFEST), negatives="none", oracle="none", dialect=None,
                out=str(ROOT / "eval" / "cache" / "regression_reports"), tag="", compare=None,
                no_two_pass=False, no_grid=True,
                limit=0, model_dir=None, asr_cache_readonly=False, quiet=True, set=[])
    base.update(kw)
    return Namespace(**base)


SUITES = {
    "text": lambda a: make_args(),
    "text_oov": lambda a: make_args(negatives="oov"),
    "text_synthetic": lambda a: make_args(eval=str(SYNTH), tag="synthetic"),
}


def snapshot(run_obj: dict) -> dict:
    """golden 内容：分片汇总 + 注入结果，不含逐条。"""
    snap = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": run_obj["mode"], "tag": run_obj.get("tag"),
            "negatives": run_obj["negatives"], "thresholds": run_obj["thresholds"], "summary": run_obj["summary"]}
    if run_obj.get("injection"):
        inj = run_obj["injection"]
        snap["injection"] = {"new_flips": inj["new_flips"], "known_flips": inj["known_flips"],
                             "cases": [{"name": c.get("name"), "flipped": c["flipped"], "known_flip": bool(c.get("known_flip"))}
                                       for c in inj["cases"]]}
    return snap


def compare(name: str, now: dict, gold: dict) -> list[str]:
    fails: list[str] = []
    gs, ns = gold["summary"]["slices"], now["summary"]["slices"]
    for key, g in gs.items():
        n = ns.get(key)
        if n is None:
            fails.append(f"[{name}] 分片 {key} 消失了（上次 n={g['n']}）")
            continue
        for m in ("exact", "deliverable"):
            if n[m][0] < g[m][0]:
                fails.append(f"[{name}] {key} {m} 下降 {g[m][0]}/{g[m][1]} → {n[m][0]}/{n[m][1]}")
        if key.startswith("negative_type=oov_db") and n.get("over_correction", 0) > g.get("over_correction", 0):
            fails.append(f"[{name}] {key} 过度纠正上升 {g.get('over_correction')} → {n.get('over_correction')}")
        if "pair" in g and "pair" in n and n["pair"].get("改坏", 0) > g["pair"].get("改坏", 0):
            fails.append(f"[{name}] {key} 改坏上升 {g['pair'].get('改坏', 0)} → {n['pair'].get('改坏', 0)}")
    all_attr = now["summary"]["slices"].get("全部=全部", {}).get("attribution", {})
    if all_attr.get("assembly_error"):
        fails.append(f"[{name}] 归因出现 assembly_error × {all_attr['assembly_error']}（拼装逻辑改坏了）")
    if "injection" in gold and "injection" in now:
        if now["injection"]["new_flips"] > gold["injection"]["new_flips"]:
            fails.append(f"[{name}] 注入新增翻转 {gold['injection']['new_flips']} → {now['injection']['new_flips']}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-golden", action="store_true")
    ap.add_argument("--with-injection", action="store_true", help="连注入用例一起跑（需模型或 ASR 缓存）")
    ap.add_argument("--asr-cache-readonly", action="store_true")
    ap.add_argument("--only", default=None, help="只跑某个套件：text / text_oov / text_synthetic / injection")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="临时覆盖 rank 常量做故意破坏测试（如 --set W_CONFLICT=1.0，回归必须变红）；不得与 --update-golden 同用")
    a = ap.parse_args()
    if a.set and a.update_golden:
        sys.exit("--set 与 --update-golden 不能同用：破坏测试的结果不许写进快照")
    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(run_eval.rank_mod, k, float(v))
        print(f"[override] rank.{k} = {v}")
    GOLDEN.mkdir(parents=True, exist_ok=True)

    suites = dict(SUITES)
    if a.with_injection:
        suites["injection"] = lambda _: make_args(mode="text", negatives="injection", limit=1,
                                                  asr_cache_readonly=a.asr_cache_readonly)
    if a.only:
        suites = {a.only: suites[a.only]}
    if not SYNTH.exists() and "text_synthetic" in suites:
        print(f"跳过 text_synthetic：{SYNTH} 不存在（先跑 scripts/gen_perturbed.py）")
        suites.pop("text_synthetic")

    fails: list[str] = []
    t0 = time.time()
    for name, mk in suites.items():
        args = mk(a)
        t = time.time()
        run_obj = run_eval.run(args)
        snap = snapshot(run_obj)
        # 逐条报告落在 eval/cache/（不进版本库），出问题时翻这里
        rep_dir = Path(args.out)
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / f"{name}.md").write_text(run_eval.build_report(run_obj), encoding="utf-8")
        (rep_dir / f"{name}.json").write_text(json.dumps(run_obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        s = snap["summary"]["slices"].get("全部=全部", {})
        line = f"{name:<16} n={s.get('n')} exact={s.get('exact')} deliverable={s.get('deliverable')} attribution={s.get('attribution')}"
        if "injection" in snap:
            line += f" injection new_flips={snap['injection']['new_flips']} known={snap['injection']['known_flips']}"
        print(f"{line}  ({time.time() - t:.0f}s)")
        gpath = GOLDEN / f"{name}.json"
        if a.update_golden or not gpath.exists():
            gpath.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"  → 写入快照 {gpath.relative_to(ROOT)}" + ("" if a.update_golden else "（首次，无基线可比）"))
            continue
        gold = json.loads(gpath.read_text(encoding="utf-8"))
        f = compare(name, snap, gold)
        fails.extend(f)
        print("  " + ("通过" if not f else "\n  ".join(f)))

    print(f"\n总耗时 {time.time() - t0:.0f}s")
    if fails:
        print(f"回归失败：{len(fails)} 项")
        sys.exit(1)
    print("回归通过")


if __name__ == "__main__":
    main()
