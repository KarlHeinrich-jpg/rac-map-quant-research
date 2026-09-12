"""GARCH(1,1)-volatility and volatility-targeted portfolio baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from .panel import load_panel, project_long_only, run_backtest, softmax
except ImportError:
    from panel import load_panel, project_long_only, run_backtest, softmax


def _garch_nll(params: np.ndarray, returns: np.ndarray) -> float:
    omega, alpha, beta = np.exp(params)
    if alpha + beta >= 0.999:
        return 1e8 + 1e8 * (alpha + beta - 0.999)
    var = float(np.var(returns))
    nll = 0.0
    for r in returns:
        var = omega + alpha * (r * r) + beta * var
        nll += 0.5 * (np.log(var + 1e-12) + (r * r) / (var + 1e-12))
    return float(nll)


def fit_garch_vol(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 60:
        return float(np.std(returns, ddof=1) if len(returns) > 1 else 0.05)
    var = max(float(np.var(returns)), 1e-6)
    x0 = np.log([var * 0.05, 0.08, 0.9])
    fit = minimize(_garch_nll, x0, args=(returns,), method="Nelder-Mead", options={"maxiter": 350, "xatol": 1e-7, "fatol": 1e-7})
    omega, alpha, beta = np.exp(fit.x)
    last_var = var
    for r in returns:
        last_var = omega + alpha * (r * r) + beta * last_var
    forecast = omega + alpha * (returns[-1] ** 2) + beta * last_var
    return float(np.sqrt(max(forecast, 1e-8)))


class GARCHVolPolicy:
    def __init__(self, forecast_vol: np.ndarray, momentum: np.ndarray | None = None):
        self.forecast_vol = np.maximum(np.asarray(forecast_vol, dtype=float), 1e-4)
        self.momentum = np.zeros_like(self.forecast_vol) if momentum is None else np.asarray(momentum, dtype=float)
        self.n_assets = len(self.forecast_vol)

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        # GARCH forecasts are frozen from the training window; momentum only
        # supplies a transparent expected-return tilt available at t.
        current_mom = np.asarray(context["features"][:, 2], dtype=float)
        score = 0.6 * current_mom / np.maximum(self.forecast_vol, 1e-4) - 0.4 * self.forecast_vol
        return project_long_only(softmax(np.clip(score, -5.0, 5.0), temperature=0.5))


def make_windows(panel: Any, first_test_year: int = 2018, last_test_year: int = 2025) -> list[dict[str, Any]]:
    out = []
    for year in range(first_test_year, last_test_year + 1):
        out.append({"test_year": year, "train_start": pd.Timestamp(f"{year-6}-01-01"),
                    "train_end": pd.Timestamp(f"{year-2}-12-31"), "valid_start": pd.Timestamp(f"{year-1}-01-01"),
                    "valid_end": pd.Timestamp(f"{year-1}-12-31"), "test_start": pd.Timestamp(f"{year}-01-01"),
                    "test_end": pd.Timestamp(f"{year}-12-31")})
    return out


def run_garch_rolling(data_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    panel = load_panel(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in make_windows(panel):
        def idx(d: pd.Timestamp, side: str) -> int:
            h = np.flatnonzero(panel.dates >= d) if side == "left" else np.flatnonzero(panel.dates <= d)
            return int(h[0] if side == "left" else h[-1])
        ts, te = idx(w["train_start"], "left"), idx(w["train_end"], "right")
        vols = np.asarray([fit_garch_vol(panel.returns[ts:te, a]) for a in range(panel.n_assets)])
        policy = GARCHVolPolicy(vols)
        _, valid = run_backtest(panel, policy, w["valid_start"], w["valid_end"], transaction_cost=0.001)
        _, metrics = run_backtest(panel, policy, w["test_start"], min(w["test_end"], panel.dates.max()), transaction_cost=0.001)
        rows.append({"seed": -1, "model": "garch_momentum", "validation_sharpe": valid["sharpe"],
                     "garch_vol_mean": float(vols.mean()), "test_year": w["test_year"],
                     "train_start": str(w["train_start"].date()), "train_end": str(w["train_end"].date()),
                     "valid_start": str(w["valid_start"].date()), "valid_end": str(w["valid_end"].date()),
                     "test_start": str(w["test_start"].date()), "test_end": str(min(w["test_end"], panel.dates.max()).date()), **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "garch_rolling_metrics.csv", index=False)
    frame.groupby("model")[["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "cvar_95_daily", "avg_turnover"]].agg(["mean", "median", "std"]).reset_index().to_csv(output_dir / "garch_rolling_summary.csv", index=False)
    (output_dir / "garch_rolling_manifest.json").write_text(json.dumps({
        "data_path": str(Path(data_path).resolve()), "model": "GARCH(1,1) conditional volatility forecast plus lagged 60-day momentum tilt",
        "window_rule": "5 calendar years train, 1 validation, 1 test; annual refit", "transaction_cost": 0.001,
        "implementation": "Gaussian quasi-maximum likelihood via scipy; no simulated data", "research_status": "baseline output; inspect and freeze before manuscript claims"
    }, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="work/rac_map/data/features_macro.parquet")
    parser.add_argument("--output-dir", default="work/rac_map/results/garch_rolling")
    args = parser.parse_args()
    f = run_garch_rolling(args.data, args.output_dir)
    print(f.groupby("model")[["cagr", "sharpe", "sortino"]].mean().round(4).to_string())
