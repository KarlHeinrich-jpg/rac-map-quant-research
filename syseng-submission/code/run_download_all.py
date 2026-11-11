"""全 A 股行情下载的常驻运行器。

以主进程 + 多个分片子进程的方式持续下载，直到全部代码成功或达到最大轮次：

- 每个分片一个独立的 ``dl_ashare_v2.py --shard i --shards N`` 子进程；
- 子进程异常退出（限流被打断、连接重置等）后自动重启继续（已下载文件自动跳过）；
- 主进程每 30 秒写一次心跳文件，便于外部查看进度。

用法::

    python src/run_download_all.py --shards 4 --workers 4 --shard-rounds 6
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STORE = DATA / "ashare_daily"
LOGS = ROOT / "logs"
sys.path.insert(0, str(Path(__file__).resolve().parent))

PYTHON = sys.executable


def count_store() -> int:
    return len(list(STORE.glob("*.parquet")))


def heartbeats() -> dict:
    beats = {}
    for i in range(8):
        path = LOGS / f"hb_shard{i}.json"
        if path.exists():
            try:
                beats[i] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return beats


def total_target(shards: int) -> int:
    sec_path = DATA / "a_share_securities.parquet"
    if not sec_path.exists():
        return 0
    import pandas as pd

    sec = pd.read_parquet(sec_path)
    sec = sec[sec["exchange"].isin(["SH", "SZ"])]
    return int(len(sec) // shards) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-restarts", type=int, default=40)
    parser.add_argument("--poll", type=int, default=30)
    parser.add_argument("--pace", type=float, default=0.2, help="每进程最小请求间隔（秒）")
    args = parser.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    target_per_shard = total_target(args.shards)
    target_total = count_store()
    print(f"store starts at {target_total} files; target ~{target_per_shard * args.shards}", flush=True)

    procs: dict[int, subprocess.Popen] = {}
    restarts: dict[int, int] = {i: 0 for i in range(args.shards)}

    def launch(i: int) -> subprocess.Popen:
        out = open(LOGS / f"shard{i}.log", "a", encoding="utf-8")
        return subprocess.Popen(
            [PYTHON, "-B", str(ROOT / "src" / "dl_ashare_v2.py"),
             "--workers", str(args.workers), "--shards", str(args.shards), "--shard", str(i),
             "--pace", str(args.pace)],
            cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT,
        )

    for i in range(args.shards):
        procs[i] = launch(i)
        time.sleep(1.5)   # 错峰启动，避免瞬时并发过高

    last = count_store()
    t_start = time.time()
    while True:
        time.sleep(args.poll)
        now = count_store()
        rate = (now - last) / args.poll
        last = now
        elapsed = (time.time() - t_start) / 60
        alive = [i for i, p in procs.items() if p.poll() is None]
        hb = heartbeats()

        # 重新拉起退出的分片
        for i, p in list(procs.items()):
            if p.poll() is not None:
                if restarts[i] < args.max_restarts:
                    restarts[i] += 1
                    procs[i] = launch(i)
                    print(f"  shard {i} exited (code {p.returncode}); restart #{restarts[i]}", flush=True)
                else:
                    print(f"  shard {i} reached restart limit", flush=True)

        status = {
            "files": now,
            "rate_codes_per_min": round(rate * 60, 2),
            "elapsed_min": round(elapsed, 1),
            "alive_shards": alive,
            "restarts": restarts,
            "heartbeats": hb,
            "eta_min": round((target_per_shard * args.shards - now) / max(rate * 60, 1e-6), 1) if rate > 0 else None,
        }
        (LOGS / "download_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{elapsed:6.1f}m] files={now} ({rate*60:.1f}/min) alive={alive} restarts={sum(restarts.values())}", flush=True)

        done = all(
            hb.get(i, {}).get("pending", 1) == 0 and hb.get(i, {}).get("finished", False)
            for i in range(args.shards)
        )
        if done and not alive:
            print("all shards reported finished", flush=True)
            break
        if all(restarts[i] >= args.max_restarts for i in range(args.shards)):
            print("all shards exhausted restarts", flush=True)
            break

    print(f"final store size: {count_store()}", flush=True)


if __name__ == "__main__":
    main()
