#!/usr/bin/env python3
"""评测：字段级准确率 + 归一化完全匹配 + CER 对照，按难点标签分层。

为什么不只报 CER
----------------
CER 对错误的**严重性**不敏感：
    201室 → 202室   只错 1 字符，但快递彻底送错门
    新街口 → 新界口 只错 1 字符，但可能跨城市误判
所以主指标是字段级 exact-match 和归一化后完全匹配；CER 只作 ASR 底层质量的
辅助参照。终极指标其实是 geocode 一致率（能不能定位到同一扇门），
本仓库没接地图 API，留在文档里说明。

三种模式
--------
    --mode text      用朗读稿当"完美 ASR 输出"跑后处理，只测归一化+检索+排序
    --mode audio     真实音频跑完整流水线（需要 data/eval/audio/<id>.wav）
    --mode baseline  真实音频只做裸 ASR + 数字归一化，不做任何检索/重排——对照组

用法
----
    python eval/run_eval.py --mode text
    python eval/run_eval.py --mode audio --dialect Sichuan
    python eval/run_eval.py --mode baseline
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dialect_addr.address_db import AddressDB  # noqa: E402
from dialect_addr.normalize import normalize as num_normalize  # noqa: E402
from dialect_addr.pipeline import Pipeline  # noqa: E402

ADMIN = ("province", "city", "district")
GEO = ("street", "road", "community")
TAIL = ("house_no", "building", "unit", "room")
AUDIO_EXT = (".wav", ".m4a", ".mp3", ".flac", ".aac")


def cer(hyp: str, ref: str) -> float:
    """字符错误率 = 编辑距离 / 参考长度。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m] / n


def _norm_admin(v: str) -> str:
    return v.replace("市辖区", "市")


def score_item(pred_fields: dict[str, str], pred_addr: str, truth: dict) -> dict:
    tf = truth["fields"]
    r: dict = {}
    # 行政区三级：exact
    for k in ADMIN:
        r[k] = _norm_admin(pred_fields.get(k, "")) == _norm_admin(tf.get(k, ""))
    # 地名（街道/路/小区）：真值里非空的名字，是否出现在预测地址串里。
    # 用"出现"而非字段对位，因为口语里街道/路/小区的层级归属本就模糊。
    geo_names = [tf[k] for k in GEO if tf.get(k)]
    r["geo_recall"] = (sum(1 for g in geo_names if g in pred_addr) / len(geo_names)) if geo_names else 1.0
    # 门牌以下：exact（空对空算对）
    for k in TAIL:
        r[k] = pred_fields.get(k, "") == tf.get(k, "")
    r["tail_all"] = all(r[k] for k in TAIL)
    r["admin_all"] = all(r[k] for k in ADMIN)
    r["exact"] = pred_addr == truth["ground_truth"]
    r["cer"] = cer(pred_addr, truth["ground_truth"])
    # 严重性分级：区级错 = 严重；只错门牌以下 = 轻微；全对 = 0
    if not r["admin_all"]:
        r["severity"] = "严重(行政区错)"
    elif r["geo_recall"] < 1.0:
        r["severity"] = "中等(地名漏/错)"
    elif not r["tail_all"]:
        r["severity"] = "轻微(门牌以下错)"
    else:
        r["severity"] = "正确"
    return r


def find_audio(audio_dir: Path, item_id: str) -> Path | None:
    for ext in AUDIO_EXT:
        p = audio_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    return None


def run(mode: str, dialect: str | None, eval_path: Path, out_dir: Path, two_pass: bool) -> Path:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    items = data["items"]
    audio_dir = eval_path.parent / "audio"
    db = AddressDB.default()

    pipe: Pipeline | None = None
    if mode in ("audio", "baseline"):
        from dialect_addr.asr import Qwen3ASR

        pipe = Pipeline(db=db, asr=Qwen3ASR(), two_pass=(two_pass and mode == "audio"))
    else:
        pipe = Pipeline(db=db, asr=None, two_pass=False)  # type: ignore[arg-type]

    rows = []
    t0 = time.time()
    for it in items:
        rec: dict = {"id": it["id"], "city": it["city"], "difficulty": it["difficulty"],
                     "truth": it["ground_truth"]}
        if mode == "text":
            res = pipe.process_text(it["spoken"], dialect)
            rec.update(raw=it["spoken"], pred=res.address, fields=res.fields,
                       decision=res.final.decision, chosen="text")
        else:
            ap = find_audio(audio_dir, it["id"])
            if ap is None:
                rec.update(raw="", pred="", fields={}, decision="no_audio", chosen="-", skipped=True)
                rows.append(rec)
                continue
            if mode == "baseline":
                a = pipe.asr.transcribe(str(ap), language=None)
                norm, tail = num_normalize(a.text)
                rec.update(raw=a.text, pred=norm, fields=tail.as_dict(), decision="baseline",
                           chosen="baseline", lang=a.language)
            else:
                res = pipe.process(str(ap), dialect_hint=dialect)
                rec.update(raw=res.pass1.raw_text, pred=res.address, fields=res.fields,
                           decision=res.final.decision, chosen=res.chosen, lang=res.dialect,
                           pass2_raw=res.pass2.raw_text if res.pass2 else "",
                           ctx_used=bool(res.pass2 and res.pass2.asr and res.pass2.asr.context_used))
        rec["score"] = score_item(rec["fields"], rec["pred"], it)
        rows.append(rec)
        s = rec["score"]
        mark = "✓" if s["exact"] else ("~" if s["admin_all"] else "✗")
        print(f"{mark} {it['id']} [{rec['decision']:9}] {rec['pred']}")
        if not s["exact"]:
            print(f"      真值 {it['ground_truth']}")

    elapsed = time.time() - t0
    scored = [r for r in rows if not r.get("skipped")]
    report = build_report(mode, dialect, scored, len(rows), elapsed, data)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    md = out_dir / f"{stamp}_{mode}.md"
    md.write_text(report, encoding="utf-8")
    (out_dir / f"{stamp}_{mode}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\n" + report)
    print(f"报告: {md}")
    return md


def build_report(mode, dialect, rows, total, elapsed, data) -> str:
    n = len(rows)
    if n == 0:
        return f"# 评测报告（{mode}）\n\n没有可评测的样本（音频缺失？）。"

    def rate(key):
        return sum(1 for r in rows if r["score"][key]) / n

    lines = [
        "---",
        "type: report",
        f"title: 方言地址识别评测 · {mode}",
        f"description: {data['_meta']['name']}，{n}/{total} 条样本，模式 {mode}",
        "tags: [ASR, 方言, 评测]",
        f"timestamp: {time.strftime('%Y-%m-%d')}",
        "---",
        "",
        f"# 方言地址识别评测（{mode}）",
        "",
        f"- 评测集: {data['_meta']['name']}  样本 {n}/{total}",
        f"- 方言: {dialect or '自动识别'}  耗时 {elapsed:.0f}s",
        "",
        "## 总体",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| **归一化完全匹配（EM）** | **{rate('exact'):.1%}** |",
        f"| 行政区三级全对 | {rate('admin_all'):.1%} |",
        f"| 省 / 市 / 区 | {rate('province'):.0%} / {rate('city'):.0%} / {rate('district'):.0%} |",
        f"| 地名召回（街道/路/小区） | {sum(r['score']['geo_recall'] for r in rows)/n:.1%} |",
        f"| 门牌以下全对 | {rate('tail_all'):.1%} |",
        f"| 门牌 / 楼栋 / 单元 / 室 | {rate('house_no'):.0%} / {rate('building'):.0%} / {rate('unit'):.0%} / {rate('room'):.0%} |",
        f"| 平均 CER（参照） | {sum(r['score']['cer'] for r in rows)/n:.3f} |",
        "",
        "## 错误严重性分布",
        "",
        "| 等级 | 数量 |",
        "|---|---|",
    ]
    sev = defaultdict(int)
    for r in rows:
        sev[r["score"]["severity"]] += 1
    for k in ("正确", "轻微(门牌以下错)", "中等(地名漏/错)", "严重(行政区错)"):
        lines.append(f"| {k} | {sev.get(k, 0)} |")

    lines += ["", "## 按难点分层", "", "| 难点 | 样本 | EM | 行政区全对 | 门牌全对 |", "|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        for d in r["difficulty"]:
            by[d].append(r)
    legend = data["_meta"].get("difficulty_legend", {})
    for d, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        k = len(rs)
        lines.append(
            f"| {d}（{legend.get(d, '')[:12]}） | {k} | "
            f"{sum(1 for r in rs if r['score']['exact'])/k:.0%} | "
            f"{sum(1 for r in rs if r['score']['admin_all'])/k:.0%} | "
            f"{sum(1 for r in rs if r['score']['tail_all'])/k:.0%} |"
        )

    lines += ["", "## 决策分布", ""]
    dec = defaultdict(int)
    for r in rows:
        dec[r["decision"]] += 1
    lines.append(", ".join(f"{k}={v}" for k, v in dec.items()))

    lines += ["", "## 逐条", "", "| # | 结果 | 决策 | 预测 | 真值 |", "|---|---|---|---|---|"]
    for r in rows:
        s = r["score"]
        mark = "✓" if s["exact"] else ("~" if s["admin_all"] else "✗")
        lines.append(f"| {r['id']} | {mark} | {r['decision']} | {r['pred']} | {r['truth']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["text", "audio", "baseline"], default="text")
    ap.add_argument("--dialect", default=None, help="方言提示，如 Sichuan；不给则由 ASR 自动识别")
    ap.add_argument("--eval", default=str(ROOT / "data" / "eval" / "xinan_guanhua.json"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "reports"))
    ap.add_argument("--no-two-pass", action="store_true", help="关闭第二遍上下文注入解码")
    ap.add_argument("--model-dir", default=None,
                    help="权重目录；不给则用 $DIALECT_ADDR_MODEL_DIR 或仓库内 models/Qwen3-ASR-1.7B-hf")
    a = ap.parse_args()
    if a.model_dir:
        # 让下游 Qwen3ASR() 的默认值走同一个环境变量，避免再传一层参数
        os.environ["DIALECT_ADDR_MODEL_DIR"] = a.model_dir
    run(a.mode, a.dialect, Path(a.eval), Path(a.out), two_pass=not a.no_two_pass)
