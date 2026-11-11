"""全 A 股日频行情下载（生产版）。

策略
----
对每只证券分别下载**不复权**与**后复权**两套日线，合并为单只证券的 parquet
文件（``data/ashare_daily/<code>.parquet``），因此：

1. 断点续跑：已存在的文件直接跳过，网络中断后重跑不会重复请求；
2. 并发下载：线程池并发，单只证券内部串行以免触发限流；
3. 失败留档：失败代码写入 ``data/ashare_daily/_failed.json`` 供下一轮重试。

字段
----
``date, open, close, high, low, volume, amount, adj_close``

- ``close`` 为不复权真实价格，用于流动性排序与实际成交价；
- ``adj_close`` 为后复权价，用于计算含分红送转的持有期总收益；
- ``amount = close * volume`` 为成交额（腾讯接口不直接提供，用价格×成交量近似，
  用于横截面流动性排序，误差由全市场价格口径一致性保证不影响排序）。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ashare_api  # noqa: E402
import net  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STORE = DATA / "ashare_daily"
STORE.mkdir(parents=True, exist_ok=True)

START = "2004-01-01"
END = "2099-12-31"

_lock = threading.Lock()
_done = 0
_ok = 0
_new = 0
_failed: list[str] = []


def _fetch_one(code: str, *, force: bool = False) -> str:
    global _done, _ok, _new
    path = STORE / f"{code}.parquet"
    if path.exists() and not force:
        with _lock:
            _done += 1
        return "cached"
    try:
        raw = ashare_api.TencentAdapter.fetch_raw(code, start=START, end=END)
        adj = ashare_api.TencentAdapter.fetch_adj(code, start=START, end=END)[["date", "adj_close"]]
        frame = raw.merge(adj, on="date", how="left")
        frame["adj_close"] = frame["adj_close"].ffill()
        if frame.empty or frame["date"].max() < pd.Timestamp("1995-01-01"):
            raise ashare_api.NoDataError("empty frame")
        frame.to_parquet(path, index=False, compression="zstd")
        with _lock:
            _done += 1
            _ok += 1
            _new += 1
            n = _done
        if n % 100 == 0:
            print(f"  progress {n} (new={_new}) failed={len(_failed)}", flush=True)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _done += 1
            _failed.append(f"{code}:{type(exc).__name__}")
            n = _done
        if n % 100 == 0:
            print(f"  progress {n} (new={_new}) failed={len(_failed)}", flush=True)
        return "fail"


def load_universe(*, include_neeq: bool = False) -> pd.DataFrame:
    path = DATA / "a_share_securities.parquet"
    if not path.exists():
        raise FileNotFoundError(
            "缺少 data/a_share_securities.parquet；请先运行 download_a_share.py 的证券列表步骤"
        )
    sec = pd.read_parquet(path)
    sec["code"] = sec["code"].astype(str).str.zfill(6)
    if not include_neeq:
        sec = sec[sec["exchange"].isin(["SH", "SZ"])]
    return sec.reset_index(drop=True)


def run(codes: list[str], workers: int, rounds: int = 4, force: bool = False) -> list[str]:
    pending = codes
    for rnd in range(rounds):
        _failed.clear()
        print(f"[round {rnd+1}] {len(pending)} codes, {max(2, workers - rnd)} workers", flush=True)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=max(2, workers - rnd)) as pool:
            list(pool.map(lambda c: _fetch_one(c, force=force), pending))
        print(f"  round {rnd+1} done in {time.time()-t0:.0f}s, failed={len(_failed)}", flush=True)
        failed_codes = sorted({item.split(":", 1)[0] for item in _failed})
        if not failed_codes:
            break
        pending = failed_codes
        (STORE / "_failed.json").write_text(json.dumps(failed_codes, indent=2), encoding="utf-8")
        time.sleep(2.0)
    return pending


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-neeq", action="store_true")
    parser.add_argument("--shard", type=int, default=0, help="分片序号（多进程并行用）")
    parser.add_argument("--shards", type=int, default=1, help="分片总数")
    parser.add_argument("--pace", type=float, default=0.05, help="本进程最小请求间隔（秒）")
    args = parser.parse_args()

    net.PACER.set_interval(args.pace)

    sec = load_universe(include_neeq=args.include_neeq)
    codes = sec["code"].tolist()
    if args.shards > 1:
        codes = [c for i, c in enumerate(codes) if i % args.shards == args.shard]
    if args.limit:
        codes = codes[: args.limit]
    print(f"universe={len(sec):,} target={len(codes):,} workers={args.workers} shard={args.shard}/{args.shards}", flush=True)

    remaining = run(codes, args.workers, force=args.force)
    manifest = {
        "source": "Tencent quote API (kline/kline raw + fqkline/get hfq)",
        "retrieved_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "shard": args.shard,
        "shards": args.shards,
        "requested": len(codes),
        "downloaded_new": _new,
        "still_failed": len(remaining),
        "failed_codes": remaining[:300],
        "store": str(STORE.relative_to(ROOT)),
    }
    (DATA / f"ashare_download_manifest_shard{args.shard}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 心跳：供 run_download_all.py 判断分片是否真正完成
    pending = sum(1 for c in codes if not (STORE / f"{c}.parquet").exists())
    (ROOT / "logs" / f"hb_shard{args.shard}.json").write_text(
        json.dumps({"pending": pending, "finished": True, "requested": len(codes),
                    "ts": pd.Timestamp.now("UTC").isoformat()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total = len(list(STORE.glob("*.parquet")))
    print(f"shard {args.shard} done: pending={pending}, store now holds {total:,} securities", flush=True)


if __name__ == "__main__":
    main()
