"""Merge rule, ML, GARCH, and RAC-MAP rolling outputs into one audit package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def merge(rule_path: str, ml_path: str, garch_path: str, output_dir: str) -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(rule_path), pd.read_csv(ml_path), pd.read_csv(garch_path)]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame.to_csv(out / "full_rolling_metrics.csv", index=False)
    metrics = ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "cvar_95_daily", "avg_turnover"]
    model_year = frame.groupby(["model", "test_year"])[metrics].mean().reset_index()
    model_year.to_csv(out / "full_model_year_metrics.csv", index=False)
    summary = model_year.groupby("model")[metrics].agg(["mean", "median", "std"]).reset_index()
    summary.to_csv(out / "full_rolling_summary.csv", index=False)
    manifest = {
        "inputs": [str(Path(rule_path).resolve()), str(Path(ml_path).resolve()), str(Path(garch_path).resolve())],
        "outputs": ["full_rolling_metrics.csv", "full_model_year_metrics.csv", "full_rolling_summary.csv"],
        "aggregation": "learned models averaged across seeds within test year; deterministic models stored once per year",
        "test_years": sorted(frame.test_year.dropna().astype(int).unique().tolist()),
        "models": sorted(frame.model.unique().tolist()),
        "transaction_cost": 0.001,
        "research_status": "combined rolling output; freeze only after final statistical and manuscript audit",
    }
    (out / "full_rolling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rule", default="work/rac_map/results/rolling_v3/rolling_metrics.csv")
    p.add_argument("--ml", default="work/rac_map/results/ml_rolling_v2/ml_rolling_metrics.csv")
    p.add_argument("--garch", default="work/rac_map/results/garch_rolling/garch_rolling_metrics.csv")
    p.add_argument("--output-dir", default="work/rac_map/results/full_v1")
    args = p.parse_args()
    f = merge(args.rule, args.ml, args.garch, args.output_dir)
    print(f.groupby("model")["sharpe"].mean().sort_values(ascending=False).round(4).to_string())
