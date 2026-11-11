"""下载全 A 股日频后复权行情并整理为面板数据。

输出（``data/`` 目录）
---------------------
``a_share_securities.parquet``  证券主表（代码、名称、交易所、上市日、行业）
``a_share_daily.parquet``       长表日频面板：code, date, close(后复权), volume, amount, turnover
``a_share_monthly.parquet``     月末面板：code, month_end 及各月聚合量
``a_share_index.parquet``       指数日频行情

设计要点
--------
1. 幸存者偏差控制：证券列表来自东方财富 F10 基本信息表，包含已退市与已摘牌
   证券（名称含“退”“摘牌”），因此样本内包含完整的退市历史，而非仅存续公司。
2. 点时信息：行情按交易日期索引；任何特征只用该日及之前的数据构造。
3. 复权口径：使用后复权价（``fqt=2``）计算收益，避免前复权导致的未来信息泄漏。
4. 网络稳定性：东方财富行情接口在高并发下会主动断开连接，故采用“多轮下载 +
   失败重试队列 + 自适应节流”策略；成功响应写入磁盘缓存，重复运行不重复请求。
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

import em_api  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOGS = ROOT / "logs"
DATA.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

BEG = "19901219"
END = "20991231"

# 沪深 A 股有效前缀。F10 基本信息表含少量非上市交易代码（如 000027 深物业优先股），
# 由下载阶段的“有效行情探测”进一步剔除。
SH_PREFIX = ("600", "601", "603", "605", "688", "689")
SZ_PREFIX = ("000", "001", "002", "003", "300", "301")

_EXCLUDE_NAME_PAT = r"(?:摘牌|退$|退市)"
_BSHARE_PAT = r"^(?:900|200)"


def select_universe(uni: pd.DataFrame, *, include_neeq: bool = False) -> pd.DataFrame:
    """筛选进入面板的证券。

    默认只保留沪深 A 股：北交所与新三板（4xx/8xx/920 前缀）及 B 股（900/200 前缀）
    的流动性与投资者结构同主板差异过大，且新三板含大量从未连续竞价交易的代码。
    退市证券保留，用于控制幸存者偏差。
    """
    frame = uni.copy()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if include_neeq:
        keep = frame["exchange"].isin(["SH", "SZ", "BJ"])
    else:
        keep = frame["code"].str.startswith(SH_PREFIX + SZ_PREFIX)
    frame = frame[keep]
    frame = frame[~frame["code"].str.match(_BSHARE_PAT)]
    frame = frame[frame["list_date"].notna()]
    frame = frame[frame["list_date"] <= pd.Timestamp("2026-12-31")]
    frame["is_delisted"] = frame["name"].str.contains(_EXCLUDE_NAME_PAT, na=False, regex=True)
    return frame.reset_index(drop=True)


_lock = threading.Lock()
_done = 0
_failed: list[str] = []


def _fetch_one(row: tuple[str, str]) -> pd.DataFrame | None:
    global _done
    code, secid = row
    try:
        frame = em_api.fetch_kline(secid, beg=BEG, end=END, fqt=2)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _done += 1
            _failed.append(f"{code}:{type(exc).__name__}")
            n = _done
        if n % 500 == 0:
            print(f"  progress {n}", flush=True)
        return None
    with _lock:
        _done += 1
        n = _done
    if n % 500 == 0:
        print(f"  progress {n}", flush=True)
    if frame.empty:
        return None
    frame = frame[["date", "close", "volume", "amount", "turnover"]].copy()
    frame.insert(0, "code", code)
    return frame


def download_all(
    securities: pd.DataFrame,
    workers: int = 6,
    limit: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """多轮下载直至收敛。返回 (面板, 仍未成功的代码列表)。"""
    tasks = list(zip(securities["code"].tolist(), securities["secid"].tolist()))
    if limit:
        tasks = tasks[:limit]
    print(f"downloading {len(tasks)} securities", flush=True)

    t0 = time.time()
    chunks: list[pd.DataFrame] = []
    pending = tasks
    round_no = 0
    while pending and round_no < 5:
        round_workers = max(2, workers - round_no * 2)
        with _lock:
            _failed.clear()
        print(
            f"[round {round_no + 1}] {len(pending)} codes, {round_workers} workers, "
            f"pacer={em_api.PACER.interval:.3f}s",
            flush=True,
        )
        with cf.ThreadPoolExecutor(max_workers=round_workers) as pool:
            for result in pool.map(_fetch_one, pending):
                if result is not None and not result.empty:
                    chunks.append(result)
        failed_codes = {item.split(":", 1)[0] for item in _failed}
        pending = [item for item in pending if item[0] in failed_codes]
        round_no += 1
        if pending:
            print(f"  {len(pending)} codes pending retry", flush=True)
            time.sleep(3.0)

    if not chunks:
        raise RuntimeError("no kline data downloaded")
    panel = pd.concat(chunks, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"]).reset_index(drop=True)
    print(
        f"downloaded {len(panel):,} rows for {panel['code'].nunique():,} codes in {time.time()-t0:.0f}s",
        flush=True,
    )
    remaining = [item[0] for item in pending]
    if remaining:
        print(f"still pending after retries: {len(remaining)} (e.g. {remaining[:10]})", flush=True)
    return panel, remaining


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """由日频面板构造月末面板。"""
    frame = daily.copy()
    frame["ym"] = frame["date"].dt.to_period("M")
    frame = frame.sort_values(["code", "date"])

    grouped = frame.groupby(["code", "ym"], sort=True)
    monthly = grouped.agg(
        month_end=("date", "last"),
        close=("close", "last"),
        n_days=("date", "size"),
        amount_mean=("amount", "mean"),
        amount_sum=("amount", "sum"),
        volume_mean=("volume", "mean"),
        turnover_mean=("turnover", "mean"),
    ).reset_index()
    tail_amount = (
        frame.groupby(["code", "ym"])["amount"]
        .apply(lambda s: float(s.tail(20).mean()))
        .rename("amount_tail20")
        .reset_index()
    )
    monthly = monthly.merge(tail_amount, on=["code", "ym"], how="left")
    monthly["ym"] = monthly["ym"].astype(str)
    return monthly


def build_daily_returns(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.sort_values(["code", "date"]).copy()
    frame["ret"] = frame.groupby("code")["close"].pct_change()
    return frame[["code", "date", "close", "ret", "amount", "turnover"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-neeq", action="store_true", help="并入北交所与新三板（默认关闭）")
    parser.add_argument("--skip-download", action="store_true", help="只重建聚合文件，不访问网络")
    args = parser.parse_args()

    print("fetching security list ...", flush=True)
    uni = em_api.fetch_security_list()
    securities = select_universe(uni, include_neeq=args.include_neeq)
    securities.to_parquet(DATA / "a_share_securities.parquet", index=False)
    print(f"universe after filter: {len(securities):,}", flush=True)
    print(securities["exchange"].value_counts().to_string(), flush=True)
    print(f"delisted (name-flagged): {int(securities['is_delisted'].sum()):,}", flush=True)

    if args.skip_download:
        cached = em_api.CACHE_DIR.glob("*.json")
        print(f"cache files available: {len(list(cached))}", flush=True)
        return

    daily, remaining = download_all(securities, workers=args.workers, limit=args.limit)
    daily.to_parquet(DATA / "a_share_daily.parquet", index=False, compression="zstd")

    monthly = build_monthly(daily)
    monthly.to_parquet(DATA / "a_share_monthly.parquet", index=False, compression="zstd")

    rets = build_daily_returns(daily)
    rets.to_parquet(DATA / "a_share_daily_returns.parquet", index=False, compression="zstd")

    print("fetching indices ...", flush=True)
    idx = em_api.fetch_indices()
    idx.to_parquet(DATA / "a_share_index.parquet", index=False, compression="zstd")

    manifest = {
        "source": "Eastmoney public quote API (push2his kline, fqt=2 backward-adjusted)",
        "retrieved_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "universe_rule": "SH/SZ A shares; B shares excluded; delisted securities retained",
        "include_neeq": bool(args.include_neeq),
        "n_securities": int(securities["code"].nunique()),
        "n_daily_rows": int(len(daily)),
        "date_min": str(daily["date"].min().date()),
        "date_max": str(daily["date"].max().date()),
        "n_failed": len(remaining),
        "failed_codes": remaining[:300],
    }
    (DATA / "a_share_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "failed_codes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
