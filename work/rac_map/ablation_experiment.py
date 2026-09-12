"""RAC-MAP ablations under the same annual rolling protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .joint_gate import JointFitConfig, fit_joint_rac_map
    from .panel import load_panel, run_backtest
    from .policies import RACMAPPolicy
except ImportError:
    from joint_gate import JointFitConfig, fit_joint_rac_map
    from panel import load_panel, run_backtest
    from policies import RACMAPPolicy


def run_ablations(data_path: str | Path, output_dir: str | Path, seeds: list[int] | None = None, epochs: int = 20) -> pd.DataFrame:
    panel = load_panel(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [7, 19, 42, 73, 101]
    rows = []
    for year in range(2018, 2026):
        train_start_date, train_end_date = pd.Timestamp(f"{year-6}-01-01"), pd.Timestamp(f"{year-2}-12-31")
        valid_start, valid_end = pd.Timestamp(f"{year-1}-01-01"), pd.Timestamp(f"{year-1}-12-31")
        test_start, test_end = pd.Timestamp(f"{year}-01-01"), min(pd.Timestamp(f"{year}-12-31"), panel.dates.max())
        idx_train = np.flatnonzero(panel.dates <= train_end_date)
        ts = int(np.flatnonzero(panel.dates >= train_start_date)[0]); te = int(idx_train[-1])
        deterministic = {
            "ablation_no_regime": RACMAPPolicy(use_regime=False),
            "ablation_no_confidence": RACMAPPolicy(use_confidence=False),
            "ablation_no_risk_score": RACMAPPolicy(use_risk=False),
        }
        for name, policy in deterministic.items():
            _, valid = run_backtest(panel, policy, valid_start, valid_end, transaction_cost=0.001)
            _, metrics = run_backtest(panel, policy, test_start, test_end, transaction_cost=0.001)
            rows.append({"seed": -1, "model": name, "test_year": year, "validation_sharpe": valid["sharpe"], **metrics})
        for seed in seeds:
            policy = fit_joint_rac_map(panel, ts, te, JointFitConfig(epochs=epochs, seed=seed, risk_penalty=0.0, turnover_penalty=0.02))
            _, valid = run_backtest(panel, policy, valid_start, valid_end, transaction_cost=0.001)
            _, metrics = run_backtest(panel, policy, test_start, test_end, transaction_cost=0.001)
            rows.append({"seed": seed, "model": "joint_rac_map_no_cvar", "test_year": year, "validation_sharpe": valid["sharpe"], **metrics})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ablation_metrics.csv", index=False)
    metrics = ["cagr", "sharpe", "sortino", "max_drawdown", "calmar", "cvar_95_daily", "avg_turnover"]
    year = frame.groupby(["model", "test_year"])[metrics].mean().reset_index()
    year.to_csv(output_dir / "ablation_model_year_metrics.csv", index=False)
    year.groupby("model")[metrics].agg(["mean", "median", "std"]).reset_index().to_csv(output_dir / "ablation_summary.csv", index=False)
    (output_dir / "ablation_manifest.json").write_text(json.dumps({"data_path": str(Path(data_path).resolve()), "seeds": seeds,
        "epochs": epochs, "window_rule": "5 train, 1 validation, 1 test calendar years; annual refit",
        "ablations": list(deterministic) + ["joint_rac_map_no_cvar"], "transaction_cost": 0.001,
        "joint_no_cvar": "risk_penalty=0 while retaining turnover penalty", "research_status": "ablation output; inspect before manuscript claims"}, indent=2), encoding="utf-8")
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="work/rac_map/data/features_macro.parquet")
    p.add_argument("--output-dir", default="work/rac_map/results/ablations")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", default="7,19,42,73,101")
    a = p.parse_args()
    f = run_ablations(a.data, a.output_dir, [int(s) for s in a.seeds.split(",")], a.epochs)
    print(f.groupby("model")[["cagr", "sharpe", "sortino"]].mean().round(4).to_string())
