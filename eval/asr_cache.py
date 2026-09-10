"""ASR 结果磁盘缓存：同一段音频 + 同一份注入上下文只让模型跑一次。

评测要反复跑（改一次权重跑一次），而 43 条录音在 MPS 上每条要几秒。
缓存键 = 音频文件内容的 sha1 + 强制语种 + 注入上下文行；键相同则结果一定相同
（解码是贪心的、确定性的），所以可以放心复用。

    asr = CachedASR(Qwen3ASR())          # 和 Qwen3ASR 接口一致，可直接塞进 Pipeline
    asr.transcribe(path, context=[...])  # 命中缓存不加载模型

缓存目录默认 eval/cache/asr/，在 .gitignore 里。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "eval" / "cache" / "asr"


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CachedASR:
    """包一层 Qwen3ASR：先查缓存，未命中才加载模型转写并写回。"""

    def __init__(self, inner: Any | None = None, cache_dir: Path | str | None = None, readonly: bool = False):
        self.inner = inner
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.readonly = readonly          # True：只读缓存，未命中直接报错（CI / 无模型环境）
        self.hits = 0
        self.misses = 0
        self._sha: dict[str, str] = {}

    # 与 Qwen3ASR 保持同名属性，Pipeline/demo 里有用到
    @property
    def ready(self) -> bool:
        return bool(self.inner and getattr(self.inner, "ready", False))

    def load(self) -> None:
        if self.inner is not None:
            self.inner.load()

    def _key(self, audio: str | Path, language: str | None, context: list[str] | str | None) -> str:
        p = Path(audio)
        sha = self._sha.get(str(p))
        if sha is None:
            sha = _file_sha1(p)
            self._sha[str(p)] = sha
        if isinstance(context, str):
            ctx = context
        elif context:
            ctx = "\n".join(context)
        else:
            ctx = ""
        h = hashlib.sha1()
        h.update(sha.encode())
        h.update(b"\x00")
        h.update((language or "").encode())
        h.update(b"\x00")
        h.update(ctx.encode())
        return h.hexdigest()

    def transcribe(self, audio: str | Path, language: str | None = None, context: list[str] | str | None = None):
        from dialect_addr.asr import ASROutput

        if not isinstance(audio, (str, Path)):
            # 内存数组没有稳定的键，直接透传
            return self.inner.transcribe(audio, language=language, context=context)
        key = self._key(audio, language, context)
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            self.hits += 1
            return ASROutput(text=d["text"], language=d.get("language"), context_used=bool(context),
                             elapsed=0.0, raw_output=d.get("raw_output", ""))
        if self.readonly or self.inner is None:
            raise RuntimeError(f"ASR 缓存未命中且无模型可用: {audio} (key={key[:10]})")
        self.misses += 1
        out = self.inner.transcribe(audio, language=language, context=context)
        f.write_text(json.dumps({
            "audio": str(Path(audio).name), "language_forced": language,
            "context": context if not isinstance(context, str) else [context],
            "text": out.text, "language": out.language, "raw_output": out.raw_output,
            "elapsed": round(out.elapsed, 2), "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return out
