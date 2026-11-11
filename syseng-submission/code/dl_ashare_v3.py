"""全 A 股日频行情下载（预筛 + 全历史）。

两阶段策略
----------
**阶段一（预筛）**：只请求最近 640 个交易日，判断腾讯行情是否仍有该证券的数据。
退市/摘牌证券在腾讯接口上通常已无数据，这一步能以极低成本把它们与"服务端限流"
区分开，避免在无效代码上反复重试把全局节流器推到上限。

**阶段二（全历史）**：仅对预筛通过的证券按 640 根 K 线窗口向前分块回补，同时
下载不复权（流动性/涨跌停判定）与后复权（总收益）两套价格。

工程要点
--------
- 单进程 + 线程池，``net.PACER`` 全局节流，避免多进程各自为政导致实际并发超标；
- 每只证券有**硬性截止时间**，超时即放弃并记入 ``_skipped.json``，不阻塞队列；
- 已存在的 parquet 直接跳过，支持断点续跑。
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
DEADLINE_SEC = 90.0

_lock = threading.Lock()
_state = {"done": 0, "new": 0, "skipped": 0}
_skipped: list[str] = []


def _viable(code: str) -> bool:
    """阶段一：用最近 640 个交易日判断腾讯是否有该证券数据。"""
    try:
        frame = ashare_api.TencentAdapter._one(
            ashare_api.to_tencent_symbol(code), "2022-01-01", "2099-12-31", adjusted=False
        )
    except Exception:  # noqa: BLE001
        return False
    return not frame.empty


def _download_one(code: str) -> str:
    path = STORE / f"{code}.parquet"
    if path.exists():
        with _lock:
            _state["done"] += 1
        return "cached"
    deadline = time.monotonic() + DEADLINE_SEC
    try:
        if not _viable(code):
            with _lock:
                _state["done"] += 1
                _state["skipped"] += 1
                _skipped.append(code)
            return "unavailable"
        if time.monotonic() > deadline:
            raise TimeoutError("deadline before full history")
        raw = ashare_api.TencentAdapter.fetch_raw(code, start=START, end=END)
        adj = ashare_api.TencentAdapter.fetch_adj(code, start=START, end=END)[["date", "adj_close"]]
        frame = raw.merge(adj, on="date", how="left")
        frame["adj_close"] = frame["adj_close"].ffill()
        if frame.empty:
            raise ashare_api.NoDataError("empty frame")
        frame.to_parquet(path, index=False, compression="zstd")
        with _lock:
            _state["done"] += 1
            _state["new"] += 1
        return "ok"
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _state["done"] += 1
            _state["skipped"] += 1
            _skipped.append(f"{code}:{type(exc).__name__}")
        return "fail"


def load_universe(*, include_neeq: bool = False) -> pd.DataFrame:
    sec = pd.read_parquet(DATA / "a_share_securities.parquet")
    sec["code"] = sec["code"].astype(str).str.zfill(6)
    if not include_neeq:
        sec = sec[sec["exchange"].isin(["SH", "SZ"])]
    return sec.reset_index(drop=True)


def run(codes: list[str], workers: int) -> None:
    print(f"downloading {len(codes)} codes with {workers} workers, "
          f"pacer={net.PACER.interval:.3f}s, deadline={DEADLINE_SEC:.0f}s", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, c): c for c in codes}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            try:
                fut.result()
            except Exception:  # noqa: BLE001
                pass
            if i % 100 == 0:
                with _lock:
                    st = dict(_state)
                elapsed = time.time() - t0
                print(f"  {i}/{len(codes)} done, new={st['new']} skipped={st['skipped']} "
                      f"({i/elapsed:.2f} codes/s, pacer={net.PACER.interval:.3f})", flush=True)
    print(f"finished in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pace", type=float, default=0.12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-neeq", action="store_true")
    parser.add_argument("--mb-per", type=int, default=2)
    args = parser.parse_args()

    net.PACER.set_interval(args.pace)
    route = net.probe_and_pin()
    print(f"route pinned to: {'direct' if route is None else route}", flush=True)
    sec = load_universe(include_neeq=args.include_neeq)
    codes = sec["code"].tolist()
    if args.limit:
        codes = codes[: args.limit]
    remaining = [c for c in codes if not (STORE / f"{c}.parquet").exists()]
    print(f"universe={len(codes):,} remaining={len(remaining):,}", flush=True)

    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "hb_download.json").write_text(
        json.dumps({"started": pd.Timestamp.now("UTC").isoformat(), "remaining": len(remaining)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    run(remaining, args.workers)

    with _lock:
        st = dict(_state)
    (DATA / "ashare_download_state.json").write_text(
        json.dumps({"remaining_start": len(remaining), **_state,
                    "unavailable_sample": _skipped[:400],
                    "store_size": len(list(STORE.glob('*.parquet'))),
                    "finished_at": pd.Timestamp.now("UTC").isoformat()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"store now {len(list(STORE.glob('*.parquet'))):,} securities; new={st['new']} skipped={st['skipped']}")


if __name__ == "__main__":
    main()
