#!/usr/bin/env python3
"""并行分片下载器（工作池 + 断点续传）。

本机单连接被限速在 ~80KB/s，多连接可聚合到 300KB/s+，故自建。
只依赖标准库，不需要先装任何三方包 —— 这正是它存在的理由：
装包本身也卡在网络上，先有它才能有别的。

设计要点（第一版踩过的坑）：
  * 用**持久 worker + 任务队列**，不是"一批线程跑完再发下一批"。
    批同步会让 15 个已完成的线程在栅栏前干等 1 个卡住的，实测直接停摆。
  * 每次请求超时短（45s），失败的分片扔回队列尾部而不是原地重试，
    避免一个坏分片长时间占住一个 worker。
  * 分片完成标记落在磁盘上（.parts/<i>.ok），中断后重跑自动续传。

用法:
    python3 pget.py <url> <输出路径> [--conns=16]
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}
CHUNK = 4 * 1024 * 1024      # 4MB 一片
REQ_TIMEOUT = 45             # 单次请求超时，宁短勿长
MAX_ATTEMPTS = 8             # 每个分片累计尝试次数（跨 worker 累加）
STALL_LIMIT = 300            # 全局无进展多少秒后判定卡死并退出

# 关键：macOS 上 urllib 会通过 _scproxy 自动读**系统代理**设置，
# shell 里 unset http_proxy/https_proxy 对它无效（curl 则不受影响）。
# 本机系统代理在 HF 的 CDN 上会返回 "Tunnel connection failed: 503"，
# 所以必须显式装一个空 ProxyHandler 强制直连。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _urlopen(req, timeout):
    return _OPENER.open(req, timeout=timeout)


def probe(url: str) -> tuple[int, str]:
    """取文件总长度 + 重定向后的最终 URL。

    用 Range 探测而非 HEAD：部分 CDN 对 HEAD 返回错误的 Content-Length。
    """
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with _urlopen(req, timeout=30) as r:
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1]), r.geturl()
    raise RuntimeError("服务器不支持 Range 请求，无法并行下载")


def download(url: str, out: str, conns: int = 16) -> None:
    total, final_url = probe(url)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    name = os.path.basename(out)

    # 只认完成标记，不认文件大小：稀疏预分配让大小从一开始就是满的
    if os.path.exists(out + ".complete") and os.path.getsize(out) == total:
        print(f"  [跳过] 已完整 {name} ({total/1e6:.1f}MB)")
        return

    with open(out, "ab"):      # 确保存在
        pass
    if os.path.getsize(out) != total:
        with open(out, "r+b") as f:
            f.truncate(total)  # 稀疏预分配，各 worker 直接 seek 写自己的区段

    part_dir = out + ".parts"
    os.makedirs(part_dir, exist_ok=True)

    ranges = [(s, min(s + CHUNK - 1, total - 1)) for s in range(0, total, CHUNK)]
    flags = [os.path.join(part_dir, f"{i}.ok") for i in range(len(ranges))]

    q: queue.Queue[int] = queue.Queue()
    todo = 0
    for i in range(len(ranges)):
        if not os.path.exists(flags[i]):
            q.put(i)
            todo += 1

    already = len(ranges) - todo
    if todo == 0:
        _finalize(out, total, flags, part_dir, name)
        return

    print(f"  {name}: {len(ranges)} 片, 已完成 {already}, 待下 {todo}", flush=True)

    lock = threading.Lock()
    state = {
        "done": already,
        "bytes": 0,
        "t0": time.time(),
        "last": time.time(),   # 最后一次有进展的时刻，用于卡死检测
    }
    attempts: dict[int, int] = {}
    dead: list[int] = []
    fh = open(out, "r+b")
    stop = threading.Event()

    remaining = threading.Semaphore(0)  # 仅用于可读性，实际用 state["left"]
    state["left"] = todo

    def worker() -> None:
        # 退出条件必须是"确实没有待办分片了"，不能是"队列此刻恰好空"。
        # 队列空只说明其他 worker 正拿着分片在跑或在 sleep 重试，
        # 这时退出会让主循环误判全部完成 —— 这个 bug 曾把一个
        # 99% 是空洞的稀疏文件判成"下载完成"。
        while not stop.is_set():
            with lock:
                if state["left"] <= 0:
                    return
            try:
                idx = q.get(timeout=2)
            except queue.Empty:
                continue      # 继续等，由 state["left"] 决定何时退出
            s, e = ranges[idx]
            try:
                req = urllib.request.Request(
                    final_url, headers={**UA, "Range": f"bytes={s}-{e}"}
                )
                with _urlopen(req, timeout=REQ_TIMEOUT) as r:
                    data = r.read()
                if len(data) != e - s + 1:
                    raise OSError(f"分片长度不符 {len(data)} != {e-s+1}")
                with lock:
                    fh.seek(s)
                    fh.write(data)
                    open(flags[idx], "wb").close()
                    state["done"] += 1
                    state["left"] -= 1
                    state["bytes"] += len(data)
                    state["last"] = time.time()
            except (urllib.error.URLError, OSError, TimeoutError, ValueError):
                with lock:
                    attempts[idx] = attempts.get(idx, 0) + 1
                    n = attempts[idx]
                if n >= MAX_ATTEMPTS:
                    with lock:
                        dead.append(idx)
                        state["left"] -= 1   # 放弃也要减，否则 worker 永不退出
                else:
                    q.put(idx)      # 扔回队尾，不原地重试——别占住 worker
                    time.sleep(min(2 * n, 10))
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(conns)]
    for t in threads:
        t.start()

    # 主线程只负责报进度和卡死检测
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with lock:
            done, nbytes, t0, last = (
                state["done"], state["bytes"], state["t0"], state["last"]
            )
        el = max(time.time() - t0, 0.1)
        spd = nbytes / el / 1024
        pct = done / len(ranges) * 100
        remain = (len(ranges) - done) * CHUNK
        eta = remain / max(nbytes / el, 1) / 60
        print(
            f"  {name[:32]:<32} {pct:5.1f}%  {spd:6.0f} KB/s  ETA {eta:5.1f}min",
            flush=True,
        )
        if time.time() - last > STALL_LIMIT:
            stop.set()
            fh.close()
            raise RuntimeError(f"{STALL_LIMIT}s 无进展，判定卡死（重跑可续传）")

    fh.close()
    if dead:
        raise RuntimeError(f"{len(dead)} 个分片重试耗尽（重跑可续传）")
    _finalize(out, total, flags, part_dir, name)


def _verify_safetensors(path: str) -> None:
    """校验 safetensors 头部可解析。

    必要性：文件是稀疏预分配的，"大小对"完全不能证明"内容对"——
    一个 99% 是空洞的文件大小也是满的。头部是 8 字节小端长度 + JSON，
    若前若干分片没真正下下来，这里会立刻炸，而不是等到加载模型时才发现。
    """
    if not path.endswith(".safetensors"):
        return
    import json as _json

    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise RuntimeError("文件过短，头部缺失")
        hlen = int.from_bytes(raw, "little")
        if not (0 < hlen < 200_000_000):
            raise RuntimeError(f"头部长度非法({hlen})，内容多半是空洞")
        head = f.read(hlen)
        if len(head) < hlen:
            raise RuntimeError("头部被截断")
        try:
            meta = _json.loads(head)
        except Exception as exc:
            raise RuntimeError(f"头部 JSON 解析失败: {exc}") from exc
        if not isinstance(meta, dict) or not meta:
            raise RuntimeError("头部内容为空")


def _finalize(out: str, total: int, flags: list[str], part_dir: str, name: str) -> None:
    got = os.path.getsize(out)
    if got != total:
        raise RuntimeError(f"大小不符: {got} != {total}")

    # 关键：必须每一片都有完成标记才算数。
    # 只比对文件大小是不够的——稀疏预分配的文件从第一秒起大小就是满的。
    missing = [i for i, p in enumerate(flags) if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"还差 {len(missing)}/{len(flags)} 片未完成（重跑可续传）"
        )

    _verify_safetensors(out)

    for p in flags:
        if os.path.exists(p):
            os.remove(p)
    if os.path.isdir(part_dir) and not os.listdir(part_dir):
        os.rmdir(part_dir)
    # 落一个完成标记：稀疏文件的大小不可信，只有这个标记可信
    with open(out + ".complete", "w") as f:
        f.write(str(total))
    print(f"  [完成] {name} ({total/1e6:.1f}MB) 校验通过", flush=True)


if __name__ == "__main__":
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    conns = 16
    for a in sys.argv[1:]:
        if a.startswith("--conns"):
            conns = int(a.split("=", 1)[1])
    download(pos[0], pos[1], conns)
