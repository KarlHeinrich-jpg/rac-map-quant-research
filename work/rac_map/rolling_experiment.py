"""Strict annual rolling-window experiment for RAC-MAP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .joint_gate import JointFitConfig, fit_joint_rac_map
    from .panel import load_panel, run_backtest
    from .policies import BuyAndHoldPolicy, EqualWeightPolicy, MASACompatiblePolicy, RACMAPPolicy, RiskParityPolicy, StaticMultiAgentPolicy
except ImportError:
    from joint_gate import JointFitConfig, fit_joint_rac_map
    from panel import load_panel, run_backtest
    from policies import BuyAndHoldPolicy, EqualWeightPolicy, MASACompatiblePolicy, RACMAPPolicy, RiskParityPolicy, StaticMultiAgentPolicy


def _index(panel, date: pd.Timestamp, side: str = "left") -> int:
    hits = np.flatnonzero(panel.dates >= date) if side == "left" else np.flatnonzero(panel.dates <= date)
    if len(hits) == 0:
        raise ValueError(f"date outside panel: {date}")
    return int(hits[0] if side == "left" else hits[-1])


def make_windows(panel, first_test_year: int = 2018, last_test_year: int = 2025) -> list[dict[str, object]]:
    windows = []
    for test_year in range(first_test_year, last_test_year + 1):
        train_start, train_end = pd.Timestamp(f"{test_year - 6}-01-01"), pd.Timestamp(f"{test_year - 2}-12-31")
        valid_start, valid_end = pd.Timestamp(f"{test_year - 1}-01-01"), pd.Timestamp(f"{test_year - 1}-12-31")
        test_start, test_end = pd.Timestamp(f"{test_year}-01-01"), pd.Timestamp(f"{test_year}-12-31")
        if test_start < panel.dates.min() or test_start > panel.dates.max():
            continue
        windows.append({"test_year": test_year, "train_start": train_start, "train_end": train_end,
                        "valid_start": valid_start, "valid_end": valid_end, "test_start": test_start,
                        "test_end": min(test_end, panel.dates.max())})
    return windows


def run_rolling(data_path: str | Path, output_dir: str | Path, seeds: list[int] | None = None, epochs: int = 40) -> pd.DataFrame:
    panel = load_panel(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [7, 19, 42, 73, 101]
    rows: list[dict[str, object]] = []
    candidate_grid = [(0.05, 0.01), (0.05, 0.03), (0.15, 0.01), (0.15, 0.03)]
    for window in make_windows(panel):
        train_start, train_end = _index(panel, window["train_start"], "left"), _index(panel, window["train_end"], "right")
        valid_start, valid_end = pd.Timestamp(window["valid_start"]), pd.Timestamp(window["valid_end"])
        test_start, test_end = pd.Timestamp(window["test_start"]), pd.Timestamp(window["test_end"])
        common = {"test_year": window["test_year"], "train_start": str(window["train_start"].date()), "train_end": str(window["train_end"].date()),
                  "valid_start": str(valid_start.date()), "valid_end": str(valid_end.date()), "test_start": str(test_start.date()), "test_end": str(test_end.date())}
        baselines = {"buy_hold": BuyAndHoldPolicy(), "equal_weight": EqualWeightPolicy(), "risk_parity": RiskParityPolicy(),
                     "static_multi_agent": StaticMultiAgentPolicy(), "masa_compatible": MASACompatiblePolicy(), "rac_map": RACMAPPolicy()}
        for model_name, policy in baselines.items():
            _, metrics = run_backtest(panel, policy, test_start, test_end, transaction_cost=0.001)
            rows.append({"seed": -1, "validation_sharpe": np.nan, "risk_penalty": np.nan, "turnover_penalty": np.nan, "model": model_name, **common, **metrics})
        for seed in seeds:
            candidates = []
            for risk_penalty, turnover_penalty in candidate_grid:
                candidate = fit_joint_rac_map(panel, train_start, train_end, JointFitConfig(epochs=epochs, seed=seed, risk_penalty=risk_penalty, turnover_penalty=turnover_penalty))
                _, validation = run_backtest(panel, candidate, valid_start, valid_end, transaction_cost=0.001)
                candidates.append((validation["sharpe"], risk_penalty, turnover_penalty, candidate))
            validation_sharpe, risk_penalty, turnover_penalty, learned = max(candidates, key=lambda row: row[0])
            _, metrics = run_backtest(panel, learned, test_start, test_end, transaction_cost=0.001)
            rows.append({"seed": seed, "validation_sharpe": validation_sharpe, "risk_penalty": risk_penalty, "turnover_penalty": turnover_penalty, "model": "joint_rac_map", **common, **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "rolling_metrics.csv", index=False)
    metric_columns = ["cagr", "sharpe", "sortino", "max_drawdown", "cvar_95_daily", "avg_turnover"]
    # Report dispersion across market years. Seeds are first averaged within
    # year so they do not inflate the effective sample size.
    model_year = frame.groupby(["model", "test_year"])[metric_columns].mean().reset_index()
    summary = model_year.groupby("model")[metric_columns].agg(["mean", "median", "std"]).reset_index()
    summary.to_csv(output_dir / "rolling_summary.csv", index=False)
    manifest = {"data_path": str(Path(data_path).resolve()), "data_end_used": str(panel.dates.max().date()),
                "window_rule": "5 calendar years train, 1 calendar year validation, 1 calendar year test; annual refit",
                "seeds": seeds, "epochs": epochs, "transaction_cost": 0.001, "models": sorted(frame["model"].unique().tolist()),
                "validation_use": "selects risk_penalty and turnover_penalty from a pre-specified 2 x 2 grid by validation Sharpe; test data are not used",
                "inference_unit": "test year after averaging learned-model results over seeds; deterministic baselines are stored once per year",
                "candidate_grid": [{"risk_penalty": r, "turnover_penalty": t} for r, t in candidate_grid],
                "research_status": "rolling multi-seed empirical output; inspect data provenance and freeze before manuscript claims"}
    (output_dir / "rolling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="work/rac_map/data/features_macro.parquet")
    parser.add_argument("--output-dir", default="work/rac_map/results/rolling")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", default="7,19,42,73,101")
    args = parser.parse_args()
    result = run_rolling(args.data, args.output_dir, [int(s) for s in args.seeds.split(",")], args.epochs)
    print(result.groupby("model")[["cagr", "sharpe", "sortino"]].mean().round(4).to_string())
