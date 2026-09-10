#!/usr/bin/env python3
"""方言地址识别 · 测试界面服务端（只用标准库，零额外依赖）。

    python demo/server.py                 # http://127.0.0.1:8850
    python demo/server.py --port 8851 --no-two-pass
    python demo/server.py --no-model      # 不加载模型，只测文本链路（开发用）

接口
----
    GET  /                      测试页面
    GET  /api/status            模型是否就绪、设备、音系空间、方言列表
    POST /api/transcribe?dialect=&two_pass=1&language=
                                body 为 WAV 字节；返回完整流水线结果 JSON
    POST /api/text              {"text": "...", "dialect": "..."}  纯文本链路

设计取舍
--------
* 模型在后台线程加载，页面秒开，状态栏显示"加载中"。
* 推理用一把锁串行化：MPS 上并发 generate 不安全，而且这是单人测试台。
* 每次录音落盘到 data/eval/audio/recordings/，既是留档也是在**积累评测集**。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import traceback
import urllib.parse
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dialect_addr.address_db import AddressDB  # noqa: E402
from dialect_addr.asr import Qwen3ASR, DEFAULT_MODEL_DIR  # noqa: E402
from dialect_addr.pipeline import Pipeline, Result  # noqa: E402
from dialect_addr.rank import current_params as rank_current_params, params_info as rank_params_info  # noqa: E402
from dialect_addr.romanize import DIALECT_ROUTING, SPACES, family_of  # noqa: E402

HTML = ROOT / "demo" / "index.html"
REC_DIR = ROOT / "data" / "eval" / "audio" / "recordings"


def _git_head() -> str:
    """当前提交短号。只读 .git 文件，不依赖 git 命令；读不到返回 unknown。"""
    try:
        head = (ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = ROOT / ".git" / head[5:]
            if ref.exists():
                return ref.read_text(encoding="utf-8").strip()[:7]
            packed = ROOT / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(" " + head[5:]):
                        return line.split()[0][:7]
            return "unknown"
        return head[:7]
    except Exception:
        return "unknown"


def _db_sha1() -> str:
    import hashlib
    import os

    p = Path(os.environ.get("DIALECT_ADDR_DB") or (ROOT / "data" / "addresses" / "cn_subset.json")).expanduser()
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:12]
    except Exception:
        return "unknown"

STATE: dict = {
    "model_ready": False,
    "model_loading": False,
    "model_error": "",
    "load_time": 0.0,
    "device": "",
    "no_model": False,
}
LOCK = threading.Lock()
PIPE: Pipeline | None = None


# --------------------------------------------------------------------------
# 序列化：把流水线的 dataclass 结果变成前端好用的 JSON
# --------------------------------------------------------------------------


def _chain_json(ch, db: AddressDB) -> dict:
    hits = []
    for lv, h in ch.hits.items():
        hits.append({"level": lv, "name": h.entry.name, "matched": h.matched_name,
                     "dist": round(h.dist, 3), "span": list(h.span)})
    for h in ch.extra:
        hits.append({"level": h.entry.level + "+", "name": h.entry.name, "matched": h.matched_name,
                     "dist": round(h.dist, 3), "span": list(h.span)})
    return {
        "name": ch.full_name(),
        "total": round(ch.total, 3), "sim": round(ch.sim, 3), "coverage": round(ch.coverage, 3),
        "prior": round(ch.prior, 3), "depth": round(ch.depth, 3), "conflict": round(ch.conflict, 3),
        "hits": hits,
    }


def _pass_json(p, db: AddressDB) -> dict:
    if p is None:
        return None
    return {
        "raw_text": p.raw_text,
        "asr_language": p.asr.language if p.asr else None,
        "asr_raw_output": p.asr.raw_output if p.asr else "",
        "asr_elapsed": round(p.asr.elapsed, 2) if p.asr else None,
        "context_used": p.asr.context_used if p.asr else False,
        "lex_text": p.lex_text,
        "lex_subs": [{"src": s.src, "dst": s.dst, "category": s.category} for s in p.lex_subs],
        "norm_text": p.norm_text,
        "geo_text": p.tail.geo_text,
        "tail": p.tail.as_dict(),
        "decision": p.ranking.decision,
        "reason": p.ranking.reason,
        "space": p.ranking.space_name,
        "nbest": [_chain_json(c, db) for c in p.ranking.nbest[:5]],
    }


def result_json(r: Result, db: AddressDB) -> dict:
    return {
        "audio": r.audio,
        "dialect": r.dialect,
        "family": family_of(r.dialect),
        "address": r.address,
        "fields": r.fields,
        "segments": r.segments,
        "decision": r.final.decision,
        "reason": r.final.ranking.reason,
        "chosen": r.chosen,
        "nbest": r.nbest[:5],
        "elapsed": round(r.elapsed, 2),
        "pass1": _pass_json(r.pass1, db),
        "pass2": _pass_json(r.pass2, db),
    }


# --------------------------------------------------------------------------


def load_model_bg(model_dir: str, two_pass: bool) -> None:
    global PIPE
    STATE["model_loading"] = True
    try:
        asr = Qwen3ASR(model_dir=model_dir)
        STATE["device"] = asr.device
        asr.load()
        PIPE = Pipeline(db=AddressDB.default(), asr=asr, two_pass=two_pass)
        STATE.update(model_ready=True, load_time=round(asr.load_time, 1))
        print(f"[model] 就绪 device={asr.device} load={asr.load_time:.1f}s", flush=True)
    except Exception as e:  # 加载失败不能让服务崩：页面要能显示错误原因
        STATE["model_error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        STATE["model_loading"] = False


class Handler(BaseHTTPRequestHandler):
    server_version = "DialectAddr/0.1"

    def log_message(self, fmt, *args):  # 精简日志
        sys.stderr.write("[http] %s\n" % (fmt % args))

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            db = PIPE.db if PIPE else AddressDB.default()
            fam: dict[str, list[str]] = {}
            for d, (_, _, f) in DIALECT_ROUTING.items():
                fam.setdefault(f, []).append(d)
            self._json({
                **STATE,
                "two_pass": PIPE.two_pass if PIPE else None,
                # 跑的是哪一版：代码提交号 + 参数文件版本 + 地址库指纹。没有这三样，两个端口上的服务分不出新旧。
                "git_commit": _git_head(),
                "params": {**rank_params_info(), "values": rank_current_params()},
                "db_sha1": _db_sha1(),
                "db_stats": db.stats(),
                "spaces": {k: {"name": v.name, "available": v.available, "reason": v.reason}
                           for k, v in SPACES.items()},
                "dialect_families": fam,
            })
            return
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n) if n else b""

        if parsed.path == "/api/text":
            try:
                req = json.loads(body.decode("utf-8"))
                pipe = PIPE or Pipeline(db=AddressDB.default(), asr=Qwen3ASR(), two_pass=False)
                r = pipe.process_text(req.get("text", ""), req.get("dialect") or None)
                self._json(result_json(r, pipe.db))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}, 500)
            return

        if parsed.path == "/api/transcribe":
            if STATE["no_model"]:
                self._json({"error": "服务以 --no-model 启动，只能用文本接口"}, 400)
                return
            if not STATE["model_ready"]:
                self._json({"error": "模型尚未就绪", "loading": STATE["model_loading"],
                            "model_error": STATE["model_error"]}, 503)
                return
            dialect = (qs.get("dialect") or [""])[0] or None
            language = (qs.get("language") or [""])[0] or None
            two_pass = (qs.get("two_pass") or ["1"])[0] not in ("0", "false", "")
            try:
                REC_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                # dialect 来自查询串，直接拼进文件名会被 ?dialect=../x 逃出 REC_DIR。
                # 只保留字母数字下划线连字符，其余（含 . / \ 空格括号）一律换成 -
                safe_dialect = re.sub(r"[^A-Za-z0-9_-]+", "-", dialect).strip("-") if dialect else ""
                wav_path = REC_DIR / f"{stamp}_{safe_dialect or 'auto'}.wav"
                ctype = (self.headers.get("Content-Type") or "audio/wav").split(";")[0].strip().lower()
                if ctype in ("audio/wav", "audio/x-wav", "audio/wave") or body[:4] == b"RIFF":
                    wav_path.write_bytes(body)
                else:
                    # MediaRecorder 回退路：webm/opus、mp4/aac、ogg → ffmpeg 统一转 16k 单声道 wav
                    import subprocess

                    ext = {"audio/webm": ".webm", "audio/mp4": ".m4a", "audio/ogg": ".ogg",
                           "audio/mpeg": ".mp3", "audio/aac": ".aac"}.get(ctype, ".bin")
                    src_path = wav_path.with_suffix(ext)
                    src_path.write_bytes(body)
                    try:
                        cp = subprocess.run(
                            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_path),
                             "-ar", "16000", "-ac", "1", str(wav_path)],
                            capture_output=True, text=True, timeout=60,
                        )
                    except FileNotFoundError:
                        self._json({
                            "error": f"浏览器提交的是 {ctype}，转码需要 ffmpeg，但系统 PATH 里找不到它。"
                                     "macOS: brew install ffmpeg；"
                                     "Windows: winget install Gyan.FFmpeg 或从 ffmpeg.org 下载后把 bin 目录加进 PATH。"
                                     "装好后重启本服务。",
                            "need_ffmpeg": True,
                        }, 400)
                        return
                    if cp.returncode != 0 or not wav_path.exists():
                        self._json({"error": f"ffmpeg 转码失败（{ctype}）: {cp.stderr[:300]}"}, 400)
                        return
                # 静音拦截：全零/近零的音频跑模型只会得到空串，还会把问题伪装成"识别不准"。
                # 用原始电平判断（load_audio_16k 会做峰值归一化，不能用它之后的值）。
                import numpy as np
                import soundfile as sf

                raw, _sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
                if raw.ndim == 2:
                    raw = raw.mean(axis=1)
                peak = float(np.abs(raw).max()) if raw.size else 0.0
                rms = float(np.sqrt(np.mean(raw ** 2))) if raw.size else 0.0
                if rms < 0.002:
                    self._json({
                        "error": f"录音为静音（RMS={rms:.4f} 峰值={peak:.3f} 时长={len(raw)/_sr:.1f}s）。"
                                 "浏览器拿到的是全零音轨，按顺序检查："
                                 "① 地址栏的麦克风权限是否放行；"
                                 "② 系统麦克风权限——macOS 在 系统设置→隐私与安全性→麦克风，"
                                 "Windows 在 设置→隐私和安全性→麦克风，勾选当前浏览器后需重启浏览器；"
                                 "③ 输入设备是否选错（装了 BlackHole/VB-Cable 这类虚拟声卡时，"
                                 "系统默认输入常被它占用，录出来永远是全零）。",
                        "silent": True, "rms": rms, "peak": peak,
                    }, 400)
                    return
                with LOCK:
                    old = PIPE.two_pass
                    PIPE.two_pass = two_pass
                    try:
                        r = PIPE.process(str(wav_path), dialect_hint=dialect, language=language)
                    finally:
                        PIPE.two_pass = old
                out = result_json(r, PIPE.db)
                out["saved_as"] = str(wav_path.relative_to(ROOT))
                self._json(out)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}, 500)
            return

        self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8850)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--no-two-pass", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="不加载模型，仅文本链路")
    a = ap.parse_args()

    global PIPE
    if a.no_model:
        STATE["no_model"] = True
        PIPE = Pipeline(db=AddressDB.default(), asr=Qwen3ASR(model_dir=a.model_dir), two_pass=False)
    else:
        threading.Thread(target=load_model_bg, args=(a.model_dir, not a.no_two_pass), daemon=True).start()

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"测试界面: http://{a.host}:{a.port}   (模型{'不加载' if a.no_model else '后台加载中'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
