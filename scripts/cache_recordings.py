#!/usr/bin/env python3
"""把 recordings/ 下所有可用录音的第 1 遍（无注入）转写结果写进 ASR 缓存。

顺带输出每个文件的质量判定（静音 / 未知格式 / 重复源文件），供 build_manifest 用。

    .venv/bin/python scripts/cache_recordings.py            # 全部
    .venv/bin/python scripts/cache_recordings.py --limit 5  # 试跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from asr_cache import CachedASR  # noqa: E402

REC = ROOT / "data" / "eval" / "audio" / "recordings"
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".flac", ".aac", ".webm", ".ogg"}


def audio_quality(p: Path) -> tuple[str, dict]:
    """ok / silent / unknown_format / duplicate（webm/m4a 源文件，已有同名 wav）。"""
    if p.suffix.lower() not in AUDIO_EXT:
        return "unknown_format", {}
    if p.suffix.lower() != ".wav" and p.with_suffix(".wav").exists():
        return "duplicate", {"of": p.with_suffix(".wav").name}
    try:
        import numpy as np
        import soundfile as sf

        raw, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if raw.ndim == 2:
            raw = raw.mean(axis=1)
        rms = float(np.sqrt(np.mean(raw ** 2))) if raw.size else 0.0
        peak = float(np.abs(raw).max()) if raw.size else 0.0
        info = {"rms": round(rms, 4), "peak": round(peak, 3), "dur": round(len(raw) / sr, 1)}
        if rms < 0.002:
            return "silent", info
        if peak >= 0.999:
            info["note"] = "peak clipped"
        return "ok", info
    except Exception as e:  # webm 等 soundfile 不认的格式
        return "unknown_format", {"error": f"{type(e).__name__}: {e}"[:80]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "eval" / "cache" / "recordings_quality.json"))
    a = ap.parse_args()

    files = sorted(p for p in REC.iterdir() if p.is_file() and not p.name.startswith("."))
    qual: dict[str, dict] = {}
    todo: list[Path] = []
    for p in files:
        q, info = audio_quality(p)
        qual[p.name] = {"quality": q, **info}
        if q == "ok":
            todo.append(p)
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(files)} 个文件，{len(todo)} 条可转写", flush=True)

    from dialect_addr.asr import Qwen3ASR

    asr = CachedASR(Qwen3ASR())
    t0 = time.time()
    for i, p in enumerate(todo, 1):
        t = time.time()
        out = asr.transcribe(str(p), language=None)
        qual[p.name]["asr_text"] = out.text
        qual[p.name]["asr_language"] = out.language
        print(f"[{i}/{len(todo)}] {p.name}  {time.time()-t:4.1f}s  [{out.language}] {out.text}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(qual, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：命中缓存 {asr.hits}，新转写 {asr.misses}，耗时 {time.time()-t0:.0f}s → {a.out}")


if __name__ == "__main__":
    main()
