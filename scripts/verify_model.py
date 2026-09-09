#!/usr/bin/env python3
"""校验 Qwen3-ASR-1.7B 权重目录是否完整可用。

三道检查，由快到慢：
  1. 文件齐全 + 大小精确匹配（秒级）
  2. safetensors 头部可解析（秒级）——排除"大小对但内容是空洞"的稀疏文件
  3. sha256 与官方一致（4.7GB 约 1~2 分钟）——最终裁决

官方值取自 huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B/tree/main，
2026-09-09 抓取。若上游更新了权重，这里的 sha256 也要跟着改。

用法:
    python3 scripts/verify_model.py            # 全部三道
    python3 scripts/verify_model.py --fast     # 只跑前两道
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _model_dir() -> Path:
    """校验目标目录。优先级：--model-dir > $DIALECT_ADDR_SRC_MODEL_DIR > 仓库内默认。"""
    argv = sys.argv
    if "--model-dir" in argv:
        i = argv.index("--model-dir")
        if i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
    env = os.environ.get("DIALECT_ADDR_SRC_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return HERE.parent / "models" / "Qwen3-ASR-1.7B"


MODEL_DIR = _model_dir()

# 文件名 → (字节数, sha256 或 None)
EXPECTED: dict[str, tuple[int, str | None]] = {
    "chat_template.json": (1161, None),
    "config.json": (6194, None),
    "generation_config.json": (142, None),
    "merges.txt": (1671853, None),
    "model.safetensors.index.json": (64821, None),
    "preprocessor_config.json": (330, None),
    "tokenizer_config.json": (12487, None),
    "vocab.json": (2776833, None),
    "model-00001-of-00002.safetensors": (
        4220320824,
        "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
    ),
    "model-00002-of-00002.safetensors": (
        478200688,
        "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
    ),
}


def check_header(path: Path) -> str | None:
    """safetensors 头部：8 字节小端长度 + JSON。返回错误描述或 None。"""
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return "文件过短"
        hlen = int.from_bytes(raw, "little")
        if not (0 < hlen < 200_000_000):
            return f"头部长度非法 {hlen}（多半是空洞文件）"
        try:
            meta = json.loads(f.read(hlen))
        except Exception as exc:
            return f"头部 JSON 解析失败: {exc}"
        if not isinstance(meta, dict) or len(meta) < 2:
            return "头部内容为空"
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
            done += len(chunk)
            print(f"\r    哈希中 {done/total*100:5.1f}%", end="", flush=True)
    print("\r" + " " * 30 + "\r", end="")
    return h.hexdigest()


def main() -> int:
    fast = "--fast" in sys.argv
    print(f"目录: {MODEL_DIR}\n")
    ok = True

    for name, (size, sha) in EXPECTED.items():
        p = MODEL_DIR / name
        if not p.exists():
            print(f"  缺失  {name}")
            ok = False
            continue
        got = p.stat().st_size
        if got != size:
            print(f"  大小错  {name}  得到 {got:,}  期望 {size:,}")
            ok = False
            continue
        if name.endswith(".safetensors"):
            err = check_header(p)
            if err:
                print(f"  头部错  {name}  {err}")
                ok = False
                continue
            if not fast and sha:
                digest = sha256_of(p)
                if digest != sha:
                    print(f"  哈希错  {name}\n        得到 {digest}\n        期望 {sha}")
                    ok = False
                    continue
                print(f"  OK    {name}  ({size/1e9:.2f}GB, sha256 一致)")
                continue
            print(f"  OK    {name}  ({size/1e9:.2f}GB, 头部通过{'，未校哈希' if fast else ''})")
            continue
        print(f"  OK    {name}")

    print()
    if ok:
        print("权重完整" + ("（快速模式，未校哈希）" if fast else "，sha256 全部一致") + "。")
        return 0
    print("权重不完整，见上方标记。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
