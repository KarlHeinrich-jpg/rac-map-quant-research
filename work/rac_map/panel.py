"""Panel conversion and deterministic backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "ret_1d", "mom_20", "mom_60", "vol_20", "sma_ratio_20", "rsi_14", "volume_z",
]
MARKET_COLUMNS = [
    "market_ret", "market_vol_20", "market_trend_60", "market_drawdown_60", "regime_score",
]
MACRO_COLUMNS = ["macro_vix", "macro_fedfunds", "macro_dgs10", "macro_t10y2y", "macro_baa10y"]


@dataclass
class MarketPanel:
    dates: pd.DatetimeIndex
    tickers: list[str]
    close: np.ndarray
    returns: np.ndarray
    features: np.ndarray
    market: np.ndarray

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def n_features(self) -> int:
        return self.features.shape[2]


def load_panel(path: str | Path) -> MarketPanel:
    frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, parse_dates=["date"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "tic"])
    tickers = sorted(frame["tic"].unique().tolist())
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    expected = len(dates) * len(tickers)
    frame = frame.set_index(["date", "tic"]).reindex(pd.MultiIndex.from_product([dates, tickers], names=["date", "tic"]))
    if frame.isna().any().any():
        raise ValueError("Feature file is not a complete date x ticker panel")
    close = frame["close"].to_numpy(dtype=np.float64).reshape(len(dates), len(tickers))
    returns = frame["ret_1d"].to_numpy(dtype=np.float64).reshape(len(dates), len(tickers))
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32).reshape(len(dates), len(tickers), len(FEATURE_COLUMNS))
    available_macro = [c for c in MACRO_COLUMNS if c in frame.columns]
    state_columns = MARKET_COLUMNS + available_macro
    market = frame[state_columns].groupby(level="date").first().reindex(dates).to_numpy(dtype=np.float32)
    if len(frame) != expected:
        raise ValueError("Unexpected panel size")
    return MarketPanel(dates, tickers, close, returns, features, market)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) / max(temperature, 1e-6)
    values = values - np.nanmax(values)
    exp = np.exp(np.clip(values, -40, 40))
    return exp / (exp.sum() + 1e-12)


def project_long_only(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = values.sum()
    return values / total if total > 1e-12 else np.ones_like(values) / len(values)


def performance_metrics(values: np.ndarray, dates: pd.DatetimeIndex, turnover: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    rets = values[1:] / values[:-1] - 1.0
    n = max(len(rets), 1)
    years = n / 252.0
    total = values[-1] / values[0] - 1.0
    cagr = (values[-1] / values[0]) ** (1.0 / max(years, 1 / 252)) - 1.0
    vol = np.std(rets, ddof=1) * np.sqrt(252) if n > 1 else 0.0
    sharpe = np.mean(rets) / (np.std(rets, ddof=1) + 1e-12) * np.sqrt(252) if n > 1 else 0.0
    downside = np.sqrt(np.mean(np.minimum(rets, 0.0) ** 2)) * np.sqrt(252) if n else 0.0
    sortino = np.mean(rets) / (downside / np.sqrt(252) + 1e-12) * np.sqrt(252) if n else 0.0
    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    max_dd = float(drawdown.min())
    q05 = np.quantile(rets, 0.05) if n else 0.0
    tail = rets[rets <= q05]
    cvar = float(tail.mean()) if len(tail) else float(q05)
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "annual_volatility": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(cagr / (abs(max_dd) + 1e-12)),
        "cvar_95_daily": cvar,
        "avg_turnover": float(np.mean(turnover)) if turnover is not None and len(turnover) else 0.0,
        "final_value": float(values[-1]),
    }


def run_backtest(
    panel: MarketPanel,
    policy: Any,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    initial_value: float = 1_000_000.0,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a policy using close(t) signals and close(t+1) returns."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    mask = (panel.dates >= start) & (panel.dates <= end)
    indices = np.flatnonzero(mask)
    if len(indices) < 3:
        raise ValueError(f"Backtest window too short: {start} to {end}")
    lo, hi = indices[0], indices[-1]
    if hasattr(policy, "reset"):
        policy.reset(panel.n_assets)
    values = [float(initial_value)]
    weights = np.ones(panel.n_assets) / panel.n_assets
    turnovers = [0.0]
    rows = [{"date": panel.dates[lo], "value": values[0], "return": 0.0, "turnover": 0.0, "weights": weights.copy()}]
    for t in range(lo, hi):
        context = {
            "date": panel.dates[t],
            "features": panel.features[t],
            "market": panel.market[t],
            "returns_history": panel.returns[max(0, t - 252) : t + 1],
            "previous_weights": weights.copy(),
            "value": values[-1],
        }
        target = policy.get_weights(context) if hasattr(policy, "get_weights") else policy(context)
        target = project_long_only(np.asarray(target).reshape(-1))
        turnover_t = 0.5 * float(np.abs(target - weights).sum())
        asset_returns = panel.returns[t + 1]
        next_ret = float(np.dot(target, asset_returns))
        net_ret = (1.0 + next_ret) * (1.0 - transaction_cost * turnover_t) - 1.0
        context["realized_asset_returns"] = asset_returns.copy()
        post_return_weights = project_long_only(target * (1.0 + asset_returns))
        values.append(values[-1] * (1.0 + net_ret))
        turnovers.append(turnover_t)
        rows.append({"date": panel.dates[t + 1], "value": values[-1], "return": net_ret, "turnover": turnover_t, "weights": target.copy()})
        if hasattr(policy, "observe"):
            policy.observe(next_ret, target, context)
        # Holdings drift with asset returns between rebalancing dates. The
        # next trade's turnover is measured against these post-return weights.
        weights = post_return_weights
    result = pd.DataFrame(rows)
    result["date"] = pd.to_datetime(result["date"])
    metrics = performance_metrics(np.asarray(values), pd.DatetimeIndex(result["date"]), np.asarray(turnovers))
    return result, metrics
