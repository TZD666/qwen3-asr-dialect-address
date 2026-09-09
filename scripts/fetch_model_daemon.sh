#!/usr/bin/env bash
# 守护循环：反复重跑 fetch_model.sh 直到全部下完。
# 本机网络会随机断流，单次运行必然中途失败；靠 pget 的分片续传 + 外层重试兜住。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for i in $(seq 1 200); do
  echo "===== 第 $i 轮 $(date '+%H:%M:%S') ====="
  if bash "$SCRIPT_DIR/fetch_model.sh"; then
    echo "===== 全部下载完成 $(date '+%H:%M:%S') ====="
    exit 0
  fi
  sleep 10
done
echo "===== 200 轮仍未完成，放弃 ====="
exit 1
