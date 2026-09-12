"""Download and transform public market data for the RAC-MAP prototype.

The raw download is kept separately from the feature table. The feature table
contains only information available at the close of date t; portfolio returns
are evaluated on t+1, which avoids look-ahead in the backtest.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


ASSETS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "LQD", "DBC"]
BENCHMARK = "SPY"
DEFAULT_START = "2010-01-01"
DEFAULT_END = "2026-09-08"  # exclusive; includes the latest complete trading day


def _download(tickers: Iterable[str], start: str, end: str, retries: int = 3) -> pd.DataFrame:
    tickers = list(tickers)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            frame = yf.download(
                tickers=tickers,
                start=start,
                end=end,
                auto_adjust=True,
                actions=False,
                progress=False,
                group_by="column",
                threads=False,
            )
            if frame.empty:
                raise RuntimeError("Yahoo Finance returned an empty frame")
            return frame
        except Exception as exc:  # network providers can fail transiently
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download {tickers}: {last_error}")


def _to_long(frame: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance's single- and multi-ticker column layouts."""
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = set(frame.columns.get_level_values(0))
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(level0):
            frame = frame.stack(level=1, future_stack=True).rename_axis(index=["date", "tic"]).reset_index()
        else:
            frame = frame.stack(level=0, future_stack=True).rename_axis(index=["date", "tic"]).reset_index()
    else:
        tic = tickers[0]
        frame = frame.copy()
        frame["tic"] = tic
        frame = frame.rename_axis("date").reset_index()
    frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
    rename = {"adj_close": "close", "price": "close"}
    frame = frame.rename(columns=rename)
    required = ["date", "tic", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns from Yahoo Finance response: {missing}; got {frame.columns.tolist()}")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None).dt.normalize()
    for col in required[2:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.sort_values(["date", "tic"]).reset_index(drop=True)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build causal technical and cross-sectional market-regime features."""
    prices = prices.copy().sort_values(["tic", "date"])
    grouped = prices.groupby("tic", group_keys=False)
    prices["ret_1d"] = grouped["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    prices["mom_20"] = grouped["close"].pct_change(20)
    prices["mom_60"] = grouped["close"].pct_change(60)
    prices["vol_20"] = grouped["ret_1d"].transform(lambda x: x.rolling(20, min_periods=10).std() * np.sqrt(252))
    prices["sma_ratio_20"] = prices["close"] / grouped["close"].transform(lambda x: x.rolling(20, min_periods=5).mean()) - 1.0
    prices["rsi_14"] = grouped["close"].transform(_rsi) / 100.0
    log_volume = np.log1p(prices["volume"])
    prices["volume_z"] = log_volume.groupby(prices["tic"]).transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean()) / x.rolling(60, min_periods=20).std()
    )

    market = prices.pivot(index="date", columns="tic", values="ret_1d").sort_index()
    market_ret = market.mean(axis=1)
    market_vol = market_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    market_trend = market_ret.rolling(60, min_periods=20).mean()
    market_level = (1 + market_ret.fillna(0)).cumprod()
    market_dd = market_level / market_level.rolling(60, min_periods=20).max() - 1.0
    regime = (market_trend / (market_vol + 1e-6)).clip(-5, 5)
    regime_frame = pd.DataFrame(
        {
            "market_ret": market_ret,
            "market_vol_20": market_vol,
            "market_trend_60": market_trend,
            "market_drawdown_60": market_dd,
            "regime_score": regime,
        }
    ).reset_index()
    prices = prices.merge(regime_frame, on="date", how="left")

    feature_cols = [
        "ret_1d", "mom_20", "mom_60", "vol_20", "sma_ratio_20", "rsi_14", "volume_z",
        "market_ret", "market_vol_20", "market_trend_60", "market_drawdown_60", "regime_score",
    ]
    prices[feature_cols] = prices[feature_cols].replace([np.inf, -np.inf], np.nan)
    # A 60-day warm-up is removed so every retained row has a fully observed state.
    prices = prices.dropna(subset=feature_cols).sort_values(["date", "tic"]).reset_index(drop=True)
    return prices


def download_dataset(
    output_dir: str | Path,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    tickers: Iterable[str] = ASSETS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download raw OHLCV and save causal features plus provenance metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tickers = list(tickers)
    raw = _to_long(_download(tickers, start, end), tickers)
    # Keep only dates with all assets to make portfolio comparisons identical.
    complete_dates = raw.groupby("date")["tic"].nunique()
    complete_dates = complete_dates[complete_dates == len(tickers)].index
    raw = raw[raw["date"].isin(complete_dates)].copy()
    features = build_features(raw)
    complete_feature_dates = features.groupby("date")["tic"].nunique()
    complete_feature_dates = complete_feature_dates[complete_feature_dates == len(tickers)].index
    features = features[features["date"].isin(complete_feature_dates)].copy()

    raw_path = output_dir / "prices_daily.csv"
    feat_path = output_dir / "features_daily.parquet"
    # Compatibility exports for the original repositories' expected schemas.
    masa_path = output_dir / f"ETF_{len(tickers)}_1d.csv"
    finrl_path = output_dir / "finrl_processed.csv"
    meta_path = output_dir / "metadata.json"
    raw.to_csv(raw_path, index=False, date_format="%Y-%m-%d")
    features.to_parquet(feat_path, index=False)
    raw.rename(columns={"tic": "stock"}).to_csv(masa_path, index=False, date_format="%Y-%m-%d")
    features.rename(columns={"stock": "tic"}).to_csv(finrl_path, index=False, date_format="%Y-%m-%d")
    metadata = {
        "source": "Yahoo Finance chart API via yfinance",
        "source_url": "https://finance.yahoo.com/",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start,
        "requested_end_exclusive": end,
        "actual_start": str(features["date"].min().date()),
        "actual_end": str(features["date"].max().date()),
        "tickers": tickers,
        "raw_rows": int(len(raw)),
        "feature_rows": int(len(features)),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "feature_sha256": hashlib.sha256(feat_path.read_bytes()).hexdigest(),
        "masa_compatibility_export": str(masa_path.name),
        "finrl_compatibility_export": str(finrl_path.name),
        "note": "Public-data prototype. Yahoo data may be revised; pin these files for a paper replication package.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return raw, features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="work/rac_map/data")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    args = parser.parse_args()
    raw, features = download_dataset(args.output_dir, args.start, args.end)
    print(f"saved {len(raw):,} raw rows and {len(features):,} feature rows")
    print(f"feature dates: {features.date.min().date()} to {features.date.max().date()}")
