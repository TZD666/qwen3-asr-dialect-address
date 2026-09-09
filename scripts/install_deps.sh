#!/usr/bin/env bash
# 逐包安装依赖，多镜像轮换 + 重试。
#
# 为什么不能一条 `uv pip install a b c` 了事：
# uv 的解析-下载是原子的，本机网络下任何一个大包（gradio/torch/numpy）超时
# 都会让整批回滚，前面下好的全白费。逐包装 = 已成功的包留下来，
# 重跑只补没装上的，天然断点续传。
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
cd "$PROJ"
source .venv/bin/activate

# 本机系统代理在部分 CDN 上会断流，一律直连
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

MIRRORS=(
  "https://mirrors.aliyun.com/pypi/simple/"
  "https://pypi.tuna.tsinghua.edu.cn/simple/"
  "https://mirrors.cloud.tencent.com/pypi/simple/"
  "https://pypi.org/simple/"
)

# 顺序有意为之：先小后大，先无依赖后有依赖。
# 前面的包装上了，即使后面的失败，核心算法链路（拼音/方言/地址库）依然可跑。
PKGS=(
  pypinyin                        # 普通话拼音 —— 官话系路径
  pycantonese                     # 粤拼 —— 粤语路径
  taibun                          # 台罗 —— 闽南语路径
  opencc-python-reimplemented     # 粤文/繁简转换 —— 语义归一化层
  numpy
  soundfile
  librosa
  torch
  transformers
  accelerate
)

install_one() {
  local pkg="$1"
  # 已装则跳过（幂等）
  if python -c "import importlib.metadata as m; m.version('${pkg}')" >/dev/null 2>&1; then
    echo "  [已装] $pkg"
    return 0
  fi
  for mirror in "${MIRRORS[@]}"; do
    for attempt in 1 2; do
      echo "  [装] $pkg  <- ${mirror}  (第 ${attempt} 次)"
      if uv pip install --index-url "$mirror" --index-strategy unsafe-best-match "$pkg" >/dev/null 2>&1; then
        echo "  [成功] $pkg"
        return 0
      fi
      sleep 3
    done
  done
  echo "  [失败] $pkg —— 四个镜像都没成功"
  return 1
}

failed=()
for p in "${PKGS[@]}"; do
  install_one "$p" || failed+=("$p")
done

echo
echo "==================== 结果 ===================="
python - <<'PY'
import importlib
mods = {
    "pypinyin": "普通话拼音(官话系)",
    "pycantonese": "粤拼(粤语)",
    "taibun": "台罗(闽南语)",
    "opencc": "粤文/繁简转换",
    "numpy": "数值",
    "soundfile": "音频读写",
    "librosa": "重采样",
    "torch": "推理后端",
    "transformers": "模型加载",
}
for m, desc in mods.items():
    try:
        mod = importlib.import_module(m)
        print(f"  OK    {m:<14} {getattr(mod, '__version__', '?'):<12} {desc}")
    except Exception as e:
        print(f"  缺    {m:<14} {'':<12} {desc}  ({type(e).__name__})")
PY

if ((${#failed[@]})); then
  echo
  echo "未装上: ${failed[*]}（重跑本脚本只补这几个）"
  exit 1
fi
