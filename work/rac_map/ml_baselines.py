"""Rolling non-agent predictive baselines for the RAC-MAP study.

Each model predicts next-day ETF returns from information available at close t.
Predictions are converted to long-only portfolio weights and evaluated with the
same close(t)->close(t+1) execution rule and transaction cost as RAC-MAP.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

try:
    from .panel import load_panel, performance_metrics, project_long_only, run_backtest, softmax
except ImportError:
    from panel import load_panel, performance_metrics, project_long_only, run_backtest, softmax


FEATURE_COUNT = 7


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    value: float | int | None = None


def _make_model(spec: ModelSpec, seed: int) -> Any:
    if spec.kind == "ols":
        return LinearRegression()
    if spec.kind == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=float(spec.value)))])
    if spec.kind == "lasso":
        return Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=float(spec.value), max_iter=5000, random_state=seed))])
    if spec.kind == "elastic_net":
        return Pipeline([("scale", StandardScaler()), ("model", ElasticNet(alpha=float(spec.value), l1_ratio=0.5, max_iter=5000, random_state=seed))])
    if spec.kind == "pls":
        return Pipeline([("scale", StandardScaler()), ("model", PLSRegression(n_components=int(spec.value), scale=False, max_iter=500))])
    if spec.kind == "svr":
        return Pipeline([("scale", StandardScaler()), ("model", SVR(C=float(spec.value), epsilon=0.0005, gamma="scale"))])
    if spec.kind == "gbr":
        return GradientBoostingRegressor(
            n_estimators=80, learning_rate=0.03, max_depth=int(spec.value),
            min_samples_leaf=10, loss="huber", random_state=seed,
        )
    raise ValueError(f"unknown model kind: {spec.kind}")


def _feature_matrix(panel: Any, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Build one row per date-asset with next-day return target."""
    if end - start < 20:
        raise ValueError("training window too short")
    x = panel.features[start:end - 1].astype(np.float64)
    m = panel.market[start:end - 1].astype(np.float64)
    n_t, n_a, n_f = x.shape
    xx = np.concatenate([x, np.repeat(m[:, None, :], n_a, axis=1)], axis=2)
    yy = panel.returns[start + 1:end].astype(np.float64)
    return xx.reshape(n_t * n_a, n_f + m.shape[1]), yy.reshape(-1)


class MLReturnPolicy:
    """Predictive return model with volatility-scaled long-only allocation."""

    def __init__(self, models: list[Any], feature_mean: np.ndarray, feature_std: np.ndarray,
                 target_std: np.ndarray, temperature: float = 0.35):
        self.models = models
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_std = np.maximum(target_std, 1e-5)
        self.temperature = temperature
        self.n_assets = len(models)

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        features = np.asarray(context["features"], dtype=np.float64)
        market = np.asarray(context["market"], dtype=np.float64)
        x = np.concatenate([features, np.repeat(market[None, :], self.n_assets, axis=0)], axis=1)
        x = (x - self.feature_mean) / self.feature_std
        pred = np.asarray([float(np.asarray(model.predict(x[i].reshape(1, -1))).reshape(-1)[0]) for i, model in enumerate(self.models)])
        scale = self.target_std
        score = np.clip(pred / scale, -4.0, 4.0)
        return project_long_only(softmax(score, temperature=self.temperature))


def fit_ml_policy(panel: Any, train_start: int, train_end: int, spec: ModelSpec, seed: int = 42) -> MLReturnPolicy:
    x, y = _feature_matrix(panel, train_start, train_end)
    n_features = x.shape[1]
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0, ddof=1), 1e-6)
    x_std = (x - mean) / std
    n_assets = panel.n_assets
    n_obs = train_end - train_start - 1
    y_panel = y.reshape(n_obs, n_assets)
    models: list[Any] = []
    target_std: list[float] = []
    for asset in range(n_assets):
        model = _make_model(spec, seed + asset)
        # Pipelines standardize internally; tree and OLS are also safe on the
        # standardized matrix. Target scale is used only for portfolio scores.
        model.fit(x_std[asset::n_assets], y_panel[:, asset])
        models.append(model)
        target_std.append(float(np.std(y_panel[:, asset], ddof=1)))
    return MLReturnPolicy(models, mean, std, np.asarray(target_std))


def make_windows(panel: Any, first_test_year: int = 2018, last_test_year: int = 2025) -> list[dict[str, Any]]:
    windows = []
    for test_year in range(first_test_year, last_test_year + 1):
        train_start = pd.Timestamp(f"{test_year - 6}-01-01")
        train_end = pd.Timestamp(f"{test_year - 2}-12-31")
        valid_start = pd.Timestamp(f"{test_year - 1}-01-01")
        valid_end = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        if test_start < panel.dates.min() or test_start > panel.dates.max():
            continue
        windows.append({"test_year": test_year, "train_start": train_start, "train_end": train_end,
                        "valid_start": valid_start, "valid_end": valid_end,
                        "test_start": test_start, "test_end": min(test_end, panel.dates.max())})
    return windows


def run_ml_rolling(data_path: str | Path, output_dir: str | Path, seeds: list[int] | None = None) -> pd.DataFrame:
    panel = load_panel(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_grids = {
        "ols": [ModelSpec("ols", "ols")],
        "ridge": [ModelSpec("ridge", "ridge", 1.0), ModelSpec("ridge", "ridge", 10.0)],
        "lasso": [ModelSpec("lasso", "lasso", 0.0001), ModelSpec("lasso", "lasso", 0.001)],
        "elastic_net": [ModelSpec("elastic_net", "elastic_net", 0.0001), ModelSpec("elastic_net", "elastic_net", 0.001)],
        "pls": [ModelSpec("pls", "pls", 2), ModelSpec("pls", "pls", 4)],
        "svr": [ModelSpec("svr", "svr", 0.1), ModelSpec("svr", "svr", 1.0)],
        "gbr": [ModelSpec("gbr", "gbr", 1), ModelSpec("gbr", "gbr", 2)],
    }
    rows: list[dict[str, Any]] = []
    seeds = seeds or [7, 19, 42, 73, 101]
    for window in make_windows(panel):
        def idx(date: pd.Timestamp, side: str) -> int:
            hits = np.flatnonzero(panel.dates >= date) if side == "left" else np.flatnonzero(panel.dates <= date)
            return int(hits[0] if side == "left" else hits[-1])
        train_start, train_end = idx(window["train_start"], "left"), idx(window["train_end"], "right")
        valid_start, valid_end = window["valid_start"], window["valid_end"]
        test_start, test_end = window["test_start"], window["test_end"]
        common = {"test_year": window["test_year"], "train_start": str(window["train_start"].date()),
                  "train_end": str(window["train_end"].date()), "valid_start": str(valid_start.date()),
                  "valid_end": str(valid_end.date()), "test_start": str(test_start.date()),
                  "test_end": str(test_end.date())}
        for seed in seeds:
            for name, grid in model_grids.items():
                candidates = []
                for spec in grid:
                    policy = fit_ml_policy(panel, train_start, train_end, spec, seed=seed)
                    _, valid_metrics = run_backtest(panel, policy, valid_start, valid_end, transaction_cost=0.001)
                    candidates.append((valid_metrics["sharpe"], spec))
                valid_sharpe, best_spec = max(candidates, key=lambda z: z[0])
                policy = fit_ml_policy(panel, train_start, train_end, best_spec, seed=seed)
                _, metrics = run_backtest(panel, policy, test_start, test_end, transaction_cost=0.001)
                rows.append({"seed": seed, "model": name, "validation_sharpe": valid_sharpe,
                             "selected_spec": f"{best_spec.kind}:{best_spec.value}", **common, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ml_rolling_metrics.csv", index=False)
    model_year = frame.groupby(["model", "test_year"])[["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "cvar_95_daily", "avg_turnover"]].mean().reset_index()
    summary = model_year.groupby("model")[["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "cvar_95_daily", "avg_turnover"]].agg(["mean", "median", "std"]).reset_index()
    summary.to_csv(output_dir / "ml_rolling_summary.csv", index=False)
    manifest = {"data_path": str(Path(data_path).resolve()), "window_rule": "5 calendar years train, 1 validation, 1 test; annual refit",
                "models": list(model_grids), "seeds": seeds, "transaction_cost": 0.001,
                "feature_set": "ETF features plus market and lagged FRED macro state variables",
                "model_selection": "pre-specified hyperparameter grids selected by validation Sharpe; test not used",
                "research_status": "non-agent predictive baseline output; inspect and freeze before manuscript claims"}
    (output_dir / "ml_rolling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="work/rac_map/data/features_macro.parquet")
    parser.add_argument("--output-dir", default="work/rac_map/results/ml_rolling")
    parser.add_argument("--seeds", default="7,19,42,73,101")
    args = parser.parse_args()
    frame = run_ml_rolling(args.data, args.output_dir, [int(s) for s in args.seeds.split(",")])
    print(frame.groupby("model")[["cagr", "sharpe", "sortino"]].mean().round(4).to_string())
