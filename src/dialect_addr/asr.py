"""Qwen3-ASR-1.7B 封装（transformers 原生后端）：裸转写 + 上下文注入的二次解码。

两遍解码（retrieve-then-rescore）
--------------------------------
    第 1 遍  裸转写                →  粗文本（方言误识大概率在这里）
             ↓ 转音、检索地址库 Top-K
    第 2 遍  把候选写进 system prompt，重新解码  →  模型把声学上模糊的部分往候选靠

为什么 system prompt 能起作用：官方 chat_template.jinja 的结构是
    <|im_start|>system\\n{system 文本}<|im_end|>
    <|im_start|>user\\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>
system 位是自由文本，直接进 Qwen3 解码器；文本侧 max_position_embeddings=65536，
塞几千条候选没问题。

诚实标注：开源版 README 里没有 "context" 字样，这个能力是从模板结构推断的，
云端 Qwen3-ASR-Flash API 主打它，开源版是否等效**必须实测**。
两遍解码做成可开关的，实测不通就退回纯外挂重排，方案不依赖它成立。

后端选择
--------
直接用 transformers 5.16 原生的 Qwen3ASRForConditionalGeneration，不装 qwen-asr 包：
  * 官方包拽 gradio/vLLM 一串依赖，本机网络装不上；
  * 原生后端让我**自己拼 messages**，system prompt 完全可控——这正是两遍解码需要的。
权重用 scripts/convert_to_hf.py 从 qwen-asr 格式转过来（纯键名重命名）。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import load_audio_16k

def _default_model_dir() -> Path:
    """权重目录。环境变量优先，便于把 4.4GB 权重放到仓库外或另一块盘。

    仓库内布局 <root>/src/dialect_addr/asr.py → parents[2] 即仓库根。
    若本包被 pip 装进 site-packages，parents[2] 指向 site-packages，
    那个默认值没有意义，此时必须用环境变量或构造参数显式指定。
    """
    env = os.environ.get("DIALECT_ADDR_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "models" / "Qwen3-ASR-1.7B-hf"


DEFAULT_MODEL_DIR = _default_model_dir()

# 官方模型卡列出的方言标签；语种识别输出的就是这些名字（或 Chinese/English 等）
KNOWN_DIALECTS = {
    "Anhui", "Dongbei", "Fujian", "Gansu", "Guizhou", "Hebei", "Henan", "Hubei",
    "Hunan", "Jiangxi", "Ningxia", "Shandong", "Shaanxi", "Shanxi", "Sichuan",
    "Tianjin", "Yunnan", "Zhejiang", "Cantonese (Hong Kong accent)",
    "Cantonese (Guangdong accent)", "Wu language", "Minnan language",
}

# 模型原始输出实测格式（2026-09-09，transformers 5.16 原生后端，skip_special_tokens=True）：
#     "language Chinese<asr_text>我家在成都市武侯区……"
#     "language Cantonese<asr_text>我住喺广州市天河区……"      ← 粤语输出的是粤文
# 即 "language" + 空格 + 语种/方言名 + 字面量 "<asr_text>" + 正文，没有冒号也没有换行。
# 下面几条按优先级兼容实测格式和官方文档里提过的变体。
_FMT_MAIN = re.compile(r"^\s*language\s+([^<\n]+?)\s*<\|?asr_text\|?>(.*)$", re.S)
_LANG_LINE = re.compile(r"^\s*(?:language\s*[:：]\s*)?([A-Za-z][A-Za-z ()\-]+?)\s*[\n：:]\s*(.*)$", re.S)
_ASR_TAG = re.compile(r"<\|?asr_text\|?>(.*)$", re.S)


@dataclass
class ASROutput:
    text: str
    language: str | None
    context_used: bool
    elapsed: float
    raw_output: str = ""
    raw: Any = field(default=None, repr=False)


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def parse_output(decoded: str) -> tuple[str | None, str]:
    """模型解码文本 → (语种/方言标签, 转写文本)。"""
    s = decoded.strip()
    m = _FMT_MAIN.match(s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _ASR_TAG.search(s)
    if m:
        return None, m.group(1).strip()
    m = _LANG_LINE.match(s)
    if m and len(m.group(1)) <= 40:
        return m.group(1).strip(), m.group(2).strip()
    return None, s


class Qwen3ASR:
    """延迟加载，第一次 transcribe 才真正把权重读进内存。"""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
    ):
        # 默认值在这里求值而不是写进签名：签名默认值在 import 时就固化了，
        # 调用方（如 run_eval 的 --model-dir）在 import 之后设 DIALECT_ADDR_MODEL_DIR 会失效。
        self.model_dir = str(model_dir) if model_dir is not None else str(_default_model_dir())
        self.device = device or _pick_device()
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self.load_time = 0.0

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        t = time.time()
        # mps 上 bf16 部分算子仍不稳，float16 更稳；cuda 用 bf16；cpu 只能 fp32
        if self.device.startswith("cuda"):
            dt = torch.bfloat16
        elif self.device == "mps":
            dt = torch.float16
        else:
            dt = torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_dir)
        self._model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.model_dir, dtype=dt, low_cpu_mem_usage=True
        ).to(self.device).eval()
        self.load_time = time.time() - t

    @property
    def ready(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: str | tuple,
        language: str | None = None,
        context: list[str] | str | None = None,
    ) -> ASROutput:
        """转写一段音频。

        audio     本地路径 或 (np.ndarray, sr)
        language  强制语种（如 "Chinese"）；None 则自动识别（会给出方言标签）
        context   候选地址列表 → 注入 system prompt 做二次解码
        """
        import torch

        self.load()
        t = time.time()
        wav, sr = load_audio_16k(audio) if isinstance(audio, (str, Path)) else audio

        system_text = self._build_context(context)
        if language:
            # 强制语种：官方做法是在 assistant 位预填 "language: X"，这里放进 system 更简单
            system_text = (system_text + "\n" if system_text else "") + f"language: {language}"

        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": [{"type": "audio", "audio": wav}]})

        proc = self._processor
        text_in = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text_in, audio=[wav], sampling_rate=sr, return_tensors="pt")
        # processor 给的梅尔特征是 float32，模型在 mps/cuda 上是半精度：
        # 浮点张量要跟着模型 dtype 走，整型（input_ids）和 bool 掩码不能动。
        model_dt = next(self._model.parameters()).dtype
        moved = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                v = v.to(self.device)
                if v.is_floating_point():
                    v = v.to(model_dt)
            moved[k] = v
        inputs = moved

        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = proc.batch_decode(gen, skip_special_tokens=True)[0]
        lang, txt = parse_output(decoded)

        return ASROutput(
            text=txt, language=lang, context_used=bool(context),
            elapsed=time.time() - t, raw_output=decoded,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _build_context(context: list[str] | str | None) -> str:
        if not context:
            return ""
        if isinstance(context, str):
            return context
        # 去重保序；行数很多时截断——65K 上下文够，但没必要把整个库塞进去
        seen: set[str] = set()
        lines = []
        for c in context:
            if c and c not in seen:
                seen.add(c)
                lines.append(c)
        lines = lines[:200]
        return "以下是可能出现的地址，请优先按这些地名转写：\n" + "\n".join(lines)


def normalize_dialect_label(lang: str | None) -> str | None:
    """把 ASR 的语种输出规整成 romanize.DIALECT_ROUTING 的键。"""
    if not lang:
        return None
    lang = lang.strip()
    if lang in KNOWN_DIALECTS:
        return lang
    low = lang.lower()
    if "cantonese" in low or low in ("yue", "粤语"):
        return "Cantonese (Guangdong accent)"
    if "minnan" in low or "hokkien" in low:
        return "Minnan language"
    if low.startswith("wu"):
        return "Wu language"
    # "Chinese" / "Mandarin" → 普通话，不指定方言
    return None
