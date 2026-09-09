#!/usr/bin/env python3
"""把 Qwen/Qwen3-ASR-1.7B（qwen-asr 包格式）转成 transformers 原生可加载的 -hf 布局。

为什么要转
----------
官方发了两套权重：`Qwen3-ASR-1.7B`（给 qwen-asr 包 / vLLM 用，键名是
Qwen3-Omni 的 `thinker.*` 布局）和 `Qwen3-ASR-1.7B-hf`（transformers 原生）。
两套张量内容一样，只是键名前缀不同。本机网络下再拉一份 4.7GB 不现实，
而 qwen-asr 包又拽 gradio 一堆依赖装不上——所以就地重命名。

映射（来自 transformers 5.16 的 LOAD REPORT，逐条核对过）
    thinker.audio_tower.*   →  model.audio_tower.*
    thinker.model.*         →  model.language_model.*
    thinker.lm_head.weight  →  lm_head.weight

转换后用 meta device 实例化原生模型，对比键集合，必须 missing=0 unexpected=0
才算成功——不靠"没报错"下结论。

用法
----
    python scripts/convert_to_hf.py            # 输出到 models/Qwen3-ASR-1.7B-hf/
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "Qwen3-ASR-1.7B"
REF = ROOT / "models" / "_hf_ref"
DST = ROOT / "models" / "Qwen3-ASR-1.7B-hf"

# 顺序有意：投影层的两条必须在通用 audio_tower 规则之前，否则会被先吃掉。
# proj1/proj2 是音频编码→文本维度的两层投影，原生把它单独拎成 multi_modal_projector。
RENAMES = [
    ("thinker.audio_tower.proj1.", "model.multi_modal_projector.linear_1."),
    ("thinker.audio_tower.proj2.", "model.multi_modal_projector.linear_2."),
    ("thinker.audio_tower.", "model.audio_tower."),
    ("thinker.model.", "model.language_model."),
    ("thinker.lm_head.", "lm_head."),
]


def rename(key: str) -> str:
    for old, new in RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    raise KeyError(f"未知前缀，映射表需要补: {key}")


def main() -> int:
    from safetensors import safe_open
    from safetensors.torch import save_file

    if not (SRC / "model.safetensors.index.json").exists():
        print(f"源目录不完整: {SRC}")
        return 1
    if not (REF / "config.json").exists():
        print(f"缺 -hf 参考配置: {REF}/config.json")
        return 1

    DST.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 张量：逐分片读、改名、写 ----
    idx = json.loads((SRC / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(idx["weight_map"].values()))
    new_map: dict[str, str] = {}
    shapes: dict[str, tuple] = {}
    total = 0
    for shard in shards:
        print(f"  转换 {shard} ...", flush=True)
        tensors = {}
        with safe_open(SRC / shard, framework="pt") as f:
            for k in f.keys():
                nk = rename(k)
                tensors[nk] = f.get_tensor(k)
                new_map[nk] = shard
                shapes[nk] = tuple(tensors[nk].shape)
                total += 1
        save_file(tensors, DST / shard, metadata={"format": "pt"})
        del tensors
    (DST / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": idx.get("metadata", {}), "weight_map": new_map}, indent=2),
        encoding="utf-8",
    )
    print(f"  {total} 个张量，{len(shards)} 个分片，耗时 {time.time()-t0:.0f}s")

    # ---- 配置：模型侧用 -hf 的，分词/特征提取用源目录的 ----
    for f in ("config.json", "generation_config.json", "chat_template.jinja", "processor_config.json"):
        shutil.copy(REF / f, DST / f)
    for f in ("tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json"):
        if (SRC / f).exists():
            shutil.copy(SRC / f, DST / f)

    # ---- 校验：键集合必须与原生模型完全一致 ----
    import torch
    from transformers import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration

    cfg = Qwen3ASRConfig.from_pretrained(DST)
    with torch.device("meta"):
        m = Qwen3ASRForConditionalGeneration(cfg)
    expected = set(m.state_dict().keys())
    got = set(new_map)
    # tie_word_embeddings=True 时 lm_head 与 embed_tokens 共享，state_dict 里可能只列一个
    tied_ok = {"lm_head.weight", "model.language_model.embed_tokens.weight"}
    missing = expected - got - tied_ok
    unexpected = got - expected - tied_ok
    print(f"  校验: 期望 {len(expected)}  实有 {len(got)}  缺失 {len(missing)}  多余 {len(unexpected)}")
    for k in sorted(missing)[:8]:
        print("    缺失", k)
    for k in sorted(unexpected)[:8]:
        print("    多余", k)
    if missing or unexpected:
        print("转换失败：键集合不一致")
        return 2

    # 键名对上了还不够——名字一样形状不一样照样是错的映射。逐个比形状。
    exp_shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    bad = [(k, shapes[k], exp_shapes[k]) for k in got & expected if shapes[k] != exp_shapes[k]]
    print(f"  形状校验: 比对 {len(got & expected)} 个，不一致 {len(bad)}")
    for k, a, b in bad[:8]:
        print(f"    {k}: 权重 {a} vs 模型 {b}")
    if bad:
        print("转换失败：张量形状不一致")
        return 2

    # ---- 分词器 / 处理器能否加载 ----
    try:
        from transformers import AutoProcessor

        p = AutoProcessor.from_pretrained(DST)
        print(f"  AutoProcessor OK: {type(p).__name__}")
    except Exception as e:
        print(f"  AutoProcessor 失败: {type(e).__name__}: {str(e)[:200]}")
        return 3

    print(f"完成 → {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
