#!/usr/bin/env python3
"""新录音接入：从 recordings/ 里多出来的文件，到一份调参提案，一条命令串起来。

    python scripts/intake.py                 # ① 转写进缓存（需模型） ② 补清单行 ③ 校验 → 列出待标注的行
    python scripts/intake.py --no-model      # 缓存已覆盖全部文件时跳过 ①
    python scripts/intake.py --tune          # 没有 pending 行时接着 ④ 特征 dump ⑤ 出提案
    python scripts/intake.py --tune --strict # 只用 confirmed 的录音

每一步非零退出就停在那一步。真值必须由**说话人本人**填（label_status: pending → 改好后 confirmed），
脚本不会替人标注——这是评测体系设计 §2.2 的规矩，标注人听录音会被 ASR 输出带偏。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MANIFEST = ROOT / "data" / "eval" / "manifest.jsonl"


def run(cmd: list[str], label: str) -> None:
    print(f"\n== {label}: {' '.join(Path(c).name if c.startswith(str(ROOT)) else c for c in cmd)}")
    cp = subprocess.run(cmd, cwd=ROOT)
    if cp.returncode != 0:
        sys.exit(f"{label} 失败（exit {cp.returncode}），停在这一步")


def pending_rows() -> list[dict]:
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("quality") == "ok" and r.get("label_status") == "pending"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true", help="不加载模型：假定 ASR 缓存已覆盖全部录音")
    ap.add_argument("--tune", action="store_true", help="清单没有 pending 行时，接着跑特征 dump 与调参提案")
    ap.add_argument("--strict", action="store_true", help="调参只用 confirmed 的录音")
    ap.add_argument("--confirm", action="store_true", help="提案后用候选参数真跑 run_eval + 回归")
    a = ap.parse_args()

    if not a.no_model:
        run([PY, str(ROOT / "scripts" / "cache_recordings.py")], "① 转写新录音进缓存")
    run([PY, str(ROOT / "scripts" / "build_manifest.py")], "② 补清单行")
    run([PY, str(ROOT / "eval" / "check_manifest.py")], "③ 清单校验")

    pend = pending_rows()
    if pend:
        print(f"\n待标注 {len(pend)} 条（说话人本人填 transcript_gold / address_gold / fields_gold，改 label_status 为 confirmed）：")
        for r in pend:
            print(f"  {r['file']}  ASR: {r.get('transcript_asr', '')}")
    else:
        print("\n没有待标注的行。")
    if not a.tune:
        return
    if pend:
        sys.exit("有 pending 行未标注，不进调参（标完再跑 --tune）")
    run([PY, str(ROOT / "eval" / "features.py")], "④ 特征 dump")
    cmd = [PY, str(ROOT / "eval" / "tune.py")] + (["--strict"] if a.strict else []) + (["--confirm"] if a.confirm else [])
    run(cmd, "⑤ 调参提案")
    print("\n提案在 eval/tuning/；看过之后 python eval/tune.py --apply <proposal.json> 才会写参数文件。")


if __name__ == "__main__":
    main()
