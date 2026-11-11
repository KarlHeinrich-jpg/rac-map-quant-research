"""国际对照资产行情下载（Yahoo Finance 公开 chart 接口）。

``yfinance`` 在当前网络出口（共享代理 IP）上触发 ``YFRateLimitError``。
本模块直接调用 Yahoo 公开的 ``v8/finance/chart`` 接口，自带磁盘缓存、
自适应节流与多主机回退，稳定性显著高于库封装。

输出
----
``data/intl_daily.parquet``   长表：date, ticker, close(adjclose), volume
``data/intl_manifest.json``   下载清单与元信息
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from pathlib import Path

import pandas as pd
import requests

for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    if os.environ.get(_var, "").endswith(":7890"):
        os.environ.pop(_var, None)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = DATA / "raw_cache" / "yahoo"
CACHE.mkdir(parents=True, exist_ok=True)

HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}

ETF10 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "LQD", "DBC"]

GLOBAL = [
    "MDY", "EWJ", "EWG", "EWU", "EWZ", "EWW", "FXI", "INDA", "IEF", "SHY",
    "TIP", "HYG", "EMB", "BND", "SLV", "USO", "RWX", "IYR", "XLE", "XLF",
    "XLK", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU", "XLC", "SMH", "IBB",
]

CHINA_OFFSHORE = ["MCHI", "KWEB", "ASHR", "GXC", "PGJ"]

ALL_TICKERS = sorted(set(ETF10 + GLOBAL + CHINA_OFFSHORE))


class _Pacer:
    def __init__(self, interval: float = 0.8, max_interval: float = 8.0) -> None:
        self._lock = threading.Lock()
        self._interval = interval
        self._max = max_interval
        self._next = 0.0
        self._streak = 0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next:
                    self._next = now + self._interval
                    return
                delay = self._next - now
            time.sleep(min(delay, 0.25))

    def success(self) -> None:
        with self._lock:
            self._streak += 1
            if self._streak >= 10:
                self._streak = 0
                self._interval = max(0.3, self._interval * 0.85)

    def failure(self) -> None:
        with self._lock:
            self._streak = 0
            self._interval = min(self._max, self._interval * 1.8)

    @property
    def interval(self) -> float:
        with self._lock:
            return self._interval


PACER = _Pacer()
_LOCAL = threading.local()


def _session() -> requests.Session:
    sess = getattr(_LOCAL, "s", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(HEADERS)
        override = os.environ.get("DSH_EM_PROXY", "http://127.0.0.1:7897")
        proxies = {}
        if override.lower() not in {"none", "direct", ""}:
            proxies = {"http": override, "https": override}
        sess.proxies = proxies
        _LOCAL.s = sess
    return sess


def _cache_file(ticker: str) -> Path:
    return CACHE / f"{ticker.replace('/', '_')}.json"


def fetch_chart(ticker: str, *, refresh: bool = False, rng: str = "max") -> pd.DataFrame:
    path = _cache_file(ticker)
    payload = None
    if path.exists() and not refresh:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        last_err: Exception | None = None
        for attempt in range(6):
            host = HOSTS[attempt % len(HOSTS)]
            PACER.wait()
            try:
                resp = _session().get(
                    f"https://{host}/v8/finance/chart/{ticker}",
                    params={"range": rng, "interval": "1d", "events": "div,split", "includeAdjustedClose": "true"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    PACER.success()
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    break
                last_err = RuntimeError(f"HTTP {resp.status_code}")
                PACER.failure()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                PACER.failure()
            time.sleep(min(0.5 * (2**attempt), 6.0) + random.random() * 0.4)
        if payload is None:
            raise RuntimeError(f"{ticker}: {last_err}")

    result = (payload.get("chart") or {}).get("result") or []
    empty = pd.DataFrame(columns=["date", "ticker", "close", "volume"])
    if not result:
        return empty
    res = result[0]
    ts = res.get("timestamp") or []
    if not ts:
        return empty
    quote = (res.get("indicators") or {}).get("quote", [{}])[0]
    adj = ((res.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(ts, unit="s", utc=True).tz_convert("UTC").normalize().tz_localize(None),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    if adj is not None and len(adj) == len(frame):
        frame["close"] = adj  # 复权价（含分红），与 A 股后复权口径一致
    frame["ticker"] = ticker
    frame = frame.dropna(subset=["close"])
    return frame[["date", "ticker", "close", "volume"]]


def download(tickers: list[str], *, workers: int = 3, refresh: bool = False) -> pd.DataFrame:
    import concurrent.futures as cf

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_chart, t, refresh=refresh): t for t in tickers}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            ticker = futures[fut]
            try:
                frame = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{ticker}: {str(exc)[:60]}")
                continue
            if not frame.empty:
                frames.append(frame)
            if i % 10 == 0:
                print(f"  {i}/{len(tickers)} done, pacer={PACER.interval:.2f}s", flush=True)
    if failed:
        print(f"failed: {failed}", flush=True)
    if not frames:
        raise RuntimeError("no data downloaded")
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--tickers", default=None, help="逗号分隔，默认全部")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else ALL_TICKERS
    print(f"downloading {len(tickers)} tickers", flush=True)
    out = download(tickers, workers=args.workers, refresh=args.refresh)
    out.to_parquet(DATA / "intl_daily.parquet", index=False, compression="zstd")
    present = sorted(out["ticker"].unique().tolist())
    manifest = {
        "source": "Yahoo Finance public chart API v8, adjclose (total-return adjusted)",
        "retrieved_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "requested": tickers,
        "present": present,
        "missing": [t for t in tickers if t not in present],
        "n_rows": int(len(out)),
        "date_min": str(out["date"].min().date()),
        "date_max": str(out["date"].max().date()),
        "groups": {"etf10": ETF10, "global_multi_asset": GLOBAL, "china_offshore": CHINA_OFFSHORE},
    }
    (DATA / "intl_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(out):,} tickers={len(present)} range={manifest['date_min']}..{manifest['date_max']}", flush=True)
    print("missing:", manifest["missing"], flush=True)


if __name__ == "__main__":
    main()
