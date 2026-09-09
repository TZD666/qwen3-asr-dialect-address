#!/usr/bin/env bash
# 拉取 Qwen3-ASR-1.7B 权重。
#
# 为什么不用 huggingface-cli / modelscope：
#   1) 它们本身要先 pip 装，而装包同样卡在网络上（先有鸡后有蛋）
#   2) 本机单连接被限速 ~80KB/s，官方 CLI 不做多连接聚合
# pget.py 只用标准库 + 16 路并行 + 断点续传，实测提速约 4 倍。
#
# 重跑本脚本即续传，已完整的文件会自动跳过。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${MODEL_DIR:-$PROJ/models/Qwen3-ASR-1.7B}"

# hf-mirror 与 modelscope 实测速度相当，前者路径规则更简单
BASE="${HF_BASE:-https://hf-mirror.com/Qwen/Qwen3-ASR-1.7B/resolve/main}"

# 走直连：本机系统代理(127.0.0.1:1082)在 HF 的 CDN 重定向上会断流
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

FILES=(
  config.json
  generation_config.json
  chat_template.json
  preprocessor_config.json
  tokenizer_config.json
  vocab.json
  merges.txt
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
)

mkdir -p "$MODEL_DIR"
echo "目标目录: $MODEL_DIR"
echo "下载源:   $BASE"
echo

fail=0
for f in "${FILES[@]}"; do
  echo ">>> $f"
  if ! python3 "$SCRIPT_DIR/pget.py" "$BASE/$f" "$MODEL_DIR/$f" --conns=16; then
    echo "!!! $f 下载失败（重跑本脚本可续传）"
    fail=1
  fi
done

echo
if [[ $fail -eq 0 ]]; then
  echo "=== 全部完成 ==="
  du -sh "$MODEL_DIR"
else
  echo "=== 有文件未完成，重跑本脚本续传 ==="
  exit 1
fi
