"""音频读取与统一为 16kHz 单声道 float32——模型只认这一种输入。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_SR = 16000


def load_audio_16k(path: str | Path) -> tuple[np.ndarray, int]:
    """任意格式（wav/m4a/mp3/flac）→ (float32 单声道 16kHz, 16000)。

    先试 soundfile（快、无损格式全支持），m4a/mp3 这类它不认的再走 librosa
    （内部走 audioread/ffmpeg）。手机录音多为 m4a，所以第二条路不是摆设。
    """
    p = str(path)
    wav: np.ndarray
    sr: int
    try:
        import soundfile as sf

        wav, sr = sf.read(p, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        wav, sr = librosa.load(p, sr=None, mono=False)
        wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2:
        # soundfile 给 (帧, 声道)，librosa 给 (声道, 帧)；按短边判定
        axis = 1 if wav.shape[1] <= 2 else 0
        wav = wav.mean(axis=axis)

    if sr != TARGET_SR:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # 峰值归一化到 -1dB 附近，避免手机录音电平过低影响识别
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav * min(1.0, 0.9 / peak)
    return wav.astype(np.float32), sr


def duration_s(wav: np.ndarray, sr: int = TARGET_SR) -> float:
    return len(wav) / sr
