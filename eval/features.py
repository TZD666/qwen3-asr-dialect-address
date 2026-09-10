#!/usr/bin/env python3
"""特征重放的第一步：检索只跑一次，把每条样本**全部候选链**的五项分量存下来。

score_chain 的五项分量（sim / coverage / prior / depth / conflict）不依赖五个权重，
决策只依赖总分、相似度、分差和阈值。所以检索跑一次，之后任意 (权重, 阈值) 组合
都能在 rescore.py 里毫秒级重放——这是调参闭环跑得动的前提（一次全量检索约 5 分钟，
一次重放不到 1 毫秒）。

来源
----
    text       24 句朗读稿（xinan_guanhua.json）
    synthetic  720 条合成扰动（perturbed.jsonl）
    oov        oov_drop.json 影响到的 10 条，用删过条目的库跑
    audio      清单里 quality=ok 且有真值草稿/确认、或无地址负样本的录音，文本取 ASR 缓存里的第 1 遍转写

近似（写进提案头）：端到端正确 ≈ "Top-1 链是真值链"（拼装错不在环内）；音频只用第 1 遍；
STRONG_HIT / LONGER_NAME_TOL / UNIQUE_SHORT_MAX / MAX_DIST 影响检索本身，不可重放，本轮保持手设。

    python eval/features.py                       # 全部来源 → eval/cache/features.jsonl
    python eval/features.py --sources text,oov    # 只跑一部分
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dialect_addr import rank as rank_mod  # noqa: E402
from dialect_addr.address_db import AddressDB  # noqa: E402
from dialect_addr.asr import normalize_dialect_label  # noqa: E402
from dialect_addr.dialect_lexicon import normalize as lex_normalize  # noqa: E402
from dialect_addr.normalize import normalize as num_normalize  # noqa: E402
from dialect_addr.pipeline import to_simplified  # noqa: E402
from dialect_addr.rank import Chain, admin_key, has_deep_hit, rank  # noqa: E402

from run_eval import AUDIO_ROOT, MANIFEST, NEG_DIR, db_without, load_items, load_manifest, manifest_to_items  # noqa: E402
from stages import ADMIN, GEO, GoldChain, chain_matches, gold_chain  # noqa: E402

OUT_DEFAULT = ROOT / "eval" / "cache" / "features.jsonl"
SOURCES = ("text", "synthetic", "oov", "audio")
FEATURE_NAMES = ("sim", "coverage", "prior", "depth", "conflict")


def sha1_file(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def chain_ok(chain: Chain, gold: GoldChain, has_address: bool, gold_names: tuple[str, ...] = ()) -> bool:
    """这条候选链算不算"对"：真值可评时就是 chain_matches；真值路级不在库时，
    行政区一致且链上没有真值以外的路级条目才算对——挂上一条邻居路就是过度纠正。

    "真值以外"的判定放宽一档：库条目是真值地名的子串（库里叫「中街」，人说「中街路」；
    「喷水池」是「喷水池国贸广场」的一部分）不算邻居，拼装会把说的名字原样保住。
    只有音近但字不同的条目（中华北路 vs 中华中路）才是过度纠正。
    """
    if not has_address:
        return False
    if gold.deepest is not None:
        return chain_matches(chain, gold)
    for lv in ADMIN:
        if lv in gold.admin and lv in chain.entries and chain.entries[lv].adcode != gold.admin[lv].adcode:
            return False
    if not any(lv in chain.entries for lv in ADMIN if lv in gold.admin):
        return False
    road_entries = [e for lv, e in chain.entries.items() if lv not in ADMIN] + [x.entry for x in chain.extra]
    for e in road_entries:
        if e.adcode in gold.road_codes:
            continue
        if any(e.name in g or g in e.name for g in gold_names if g):
            continue
        return False
    return True


def featurize(text: str, db: AddressDB, dialect: str | None, fields: dict, has_address: bool) -> dict:
    """与 pipeline._postprocess 同一条路（少了拼装），候选取 topk=200 拿到全部链。"""
    lex_text, _ = lex_normalize(to_simplified(text))
    _, tail = num_normalize(lex_text)
    rr = rank(tail.geo_text, db, dialect, topk=200)
    gold = gold_chain(db, fields) if fields else GoldChain()
    gold_names = tuple(fields.get(k, "") for k in GEO if fields.get(k))
    chains = []
    for c in rr.nbest:
        ev = list(c.hits.values()) + c.extra
        chains.append({
            "name": c.full_name(),
            "f": [round(c.sim, 6), round(c.coverage, 6), round(c.prior, 6), round(c.depth, 6), round(c.conflict, 6)],
            "ok": chain_ok(c, gold, has_address, gold_names),
            "admin": "|".join(admin_key(c)),
            "deep": has_deep_hit(c),
            "n_ev": len(ev),
            "shd": round(ev[0].dist, 6) if len(ev) == 1 else None,
            "total_live": round(c.total, 6),
        })
    return {
        "geo_text": rr.geo_text, "dialect": dialect, "decision_live": rr.decision,
        "top_live": 0 if chains else None, "chains": chains,
        "gold": {"in_db": gold.in_db, "evaluable": gold.evaluable, "short_name": gold.short_name,
                 "deepest": gold.deepest.name if gold.deepest else "", "missing": gold.missing},
    }


def _meta(it: dict, source: str, group: str) -> dict:
    return {
        "dialect_group": it.get("dialect_group"), "address_depth": it.get("address_depth"), "noise": it.get("noise"),
        "difficulty": it.get("difficulty") or [], "negative_type": it.get("negative_type"), "rule": it.get("rule"),
        "label_status": it.get("label_status"), "has_address": it.get("has_address", True),
        "set": it.get("set"), "source": source, "group": group, "split": it.get("split", "eval"),
    }


def build(sources: list[str], limit: int = 0, verbose: bool = True) -> tuple[list[dict], dict]:
    db = AddressDB.default()
    rows: list[dict] = []
    info: dict = {"sources": {}, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "params": rank_mod.params_info(), "params_values": rank_mod.current_params(),
                  "inputs": {"xinan_guanhua": sha1_file(ROOT / "data/eval/xinan_guanhua.json"),
                             "perturbed": sha1_file(ROOT / "data/eval/synthetic/perturbed.jsonl"),
                             "manifest": sha1_file(MANIFEST), "cn_subset": sha1_file(ROOT / "data/addresses/cn_subset.json"),
                             "oov_drop": sha1_file(NEG_DIR / "oov_drop.json")}}

    def add(source: str, it: dict, text: str, group: str, d: AddressDB, dialect: str | None) -> None:
        fx = featurize(text, d, dialect, it.get("fields") or {}, it.get("has_address", True))
        rows.append({"id": f"{source}:{it['id']}", "item_id": it["id"], "source": source,
                     "split": it.get("split", "eval"), "group": group, "meta": _meta(it, source, group),
                     "gold_in_db": fx["gold"]["in_db"], "gold_evaluable": fx["gold"]["evaluable"],
                     "short_name": fx["gold"]["short_name"], **{k: v for k, v in fx.items() if k != "gold"}})

    t0 = time.time()
    if "text" in sources:
        _, items = load_items(ROOT / "data/eval/xinan_guanhua.json")
        for it in items[: limit or None]:
            add("text", it, it["spoken"], it["id"], db, None)
        info["sources"]["text"] = len(items[: limit or None])
    if "synthetic" in sources:
        _, items = load_items(ROOT / "data/eval/synthetic/perturbed.jsonl")
        for i, it in enumerate(items[: limit or None]):
            add("synthetic", it, it["spoken"], str(it.get("source_id", it["id"])), db, None)
            if verbose and (i + 1) % 100 == 0:
                print(f"  synthetic {i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
        info["sources"]["synthetic"] = len(items[: limit or None])
    if "oov" in sources:
        spec = json.loads((NEG_DIR / "oov_drop.json").read_text(encoding="utf-8"))
        d, removed = db_without(db, spec["drop"])
        affected = {i for x in spec["drop"] for i in x.get("affects", [])}
        _, items = load_items(ROOT / "data/eval/xinan_guanhua.json")
        n = 0
        for it in items:
            if it["id"] in affected:
                it = {**it, "negative_type": "oov_db"}
                add("oov", it, it["spoken"], it["id"], d, None)
                n += 1
        info["sources"]["oov"] = n
        info["oov_removed"] = removed
    if "audio" in sources:
        from asr_cache import CachedASR

        asr = CachedASR(None, readonly=True)
        items = manifest_to_items(load_manifest(MANIFEST))
        n = skipped = 0
        for it in items[: limit or None]:
            if not it.get("scored"):
                continue
            ap = AUDIO_ROOT / it["file"]
            try:
                a = asr.transcribe(str(ap), language=None)
            except RuntimeError:
                skipped += 1
                continue
            dialect = normalize_dialect_label(a.language)
            it = {**it, "file": it["file"]}
            add("audio", it, a.text, it["file"], db, dialect)
            rows[-1]["asr_text"] = a.text
            n += 1
        info["sources"]["audio"] = n
        info["audio_skipped_no_cache"] = skipped
    info["elapsed"] = round(time.time() - t0, 1)
    info["n"] = len(rows)
    return rows, info


def write(rows: list[dict], info: dict, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    digest = sha1_file(out)
    info["features_sha1"] = digest
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sources = [s for s in a.sources.split(",") if s]
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        sys.exit(f"未知来源 {bad}，可选 {SOURCES}")
    rows, info = build(sources, a.limit)
    digest = write(rows, info, Path(a.out))
    print(f"{info['n']} 条 → {a.out}  sha1={digest}  耗时 {info['elapsed']}s  来源 {info['sources']}")


if __name__ == "__main__":
    main()
