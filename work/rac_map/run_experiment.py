"""Run data-backed baselines, MASA-compatible policies and RAC-MAP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO, TD3

try:
    from .panel import load_panel, run_backtest
    from .policies import (
        BuyAndHoldPolicy,
        EqualWeightPolicy,
        MASACompatiblePolicy,
        MASATD3Policy,
        RACMAPPolicy,
        RiskParityPolicy,
        SB3Policy,
        StaticMultiAgentPolicy,
    )
    from .train_models import train_models
except ImportError:  # direct script execution
    from panel import load_panel, run_backtest
    from policies import (
        BuyAndHoldPolicy,
        EqualWeightPolicy,
        MASACompatiblePolicy,
        MASATD3Policy,
        RACMAPPolicy,
        RiskParityPolicy,
        SB3Policy,
        StaticMultiAgentPolicy,
    )
    from train_models import train_models


def run_experiment(data_path: str | Path, model_dir: str | Path, result_dir: str | Path, timesteps: int, seed: int, train: bool = True) -> pd.DataFrame:
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    ppo_path = model_dir / "finrl_ppo.zip"
    masa_profit_path = model_dir / "masa_profit_td3.zip"
    masa_risk_path = model_dir / "masa_risk_td3.zip"
    if train or not (ppo_path.exists() and masa_profit_path.exists() and masa_risk_path.exists()):
        train_models(data_path, model_dir, timesteps=timesteps, seed=seed)

    panel = load_panel(data_path)
    ppo = PPO.load(str(ppo_path), device="cpu")
    masa_profit = TD3.load(str(masa_profit_path), device="cpu")
    masa_risk = TD3.load(str(masa_risk_path), device="cpu")
    policies = {
        "buy_hold": BuyAndHoldPolicy(),
        "equal_weight": EqualWeightPolicy(),
        "risk_parity": RiskParityPolicy(),
        "static_multi_agent": StaticMultiAgentPolicy(),
        "masa_compatible": MASACompatiblePolicy(),
        "finrl_ppo": SB3Policy(ppo, panel.n_assets, panel.n_features, panel.market.shape[1]),
        "masa_td3": MASATD3Policy(masa_profit, masa_risk, panel.n_assets),
        "rac_map": RACMAPPolicy(),
        "rac_map_no_regime": RACMAPPolicy(use_regime=False),
        "rac_map_no_confidence": RACMAPPolicy(use_confidence=False),
        "rac_map_no_risk": RACMAPPolicy(use_risk=False),
    }
    periods = {
        "validation_2022_2023": ("2022-01-01", "2023-12-31"),
        "test_2024_2025": ("2024-01-01", "2025-12-31"),
        "external_2026_ytd": ("2026-01-01", str(panel.dates.max().date())),
    }
    metric_rows: list[dict[str, object]] = []
    equity_frames: list[pd.DataFrame] = []
    for period, (start, end) in periods.items():
        if pd.Timestamp(start) > panel.dates.max() or pd.Timestamp(end) < panel.dates.min():
            continue
        end = min(pd.Timestamp(end), panel.dates.max()).strftime("%Y-%m-%d")
        for name, policy in policies.items():
            try:
                equity, metrics = run_backtest(panel, policy, start, end)
            except ValueError:
                continue
            metric_rows.append({"period": period, "model": name, **metrics})
            equity_frames.append(equity[["date", "value"]].rename(columns={"value": name}).assign(period=period))

    metrics_df = pd.DataFrame(metric_rows).sort_values(["period", "sharpe"], ascending=[True, False])
    metrics_df.to_csv(result_dir / "metrics.csv", index=False)
    manifest_path = model_dir / "manifest.json"
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    (result_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "data_path": str(data_path.resolve()),
                "model_dir": str(model_dir.resolve()),
                "transaction_cost": 0.001,
                "requested_train_timesteps": timesteps,
                "loaded_model_manifest": loaded_manifest,
                "seed": seed,
                "periods": periods,
                "policy_names": list(policies),
                "research_status": "smoke-test outputs; increase seeds/timesteps and freeze data before inference",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metric_rows, handle, indent=2)
    for period in periods:
        frames = [x.drop(columns="period") for x in equity_frames if x["period"].iloc[0] == period]
        if not frames:
            continue
        equity_df = frames[0]
        for frame in frames[1:]:
            equity_df = equity_df.merge(frame, on="date", how="outer")
        equity_df = equity_df.sort_values("date")
        equity_df.to_csv(result_dir / f"equity_{period}.csv", index=False)
        plt.figure(figsize=(12, 6))
        for name in policies:
            if name in equity_df:
                plt.plot(equity_df["date"], equity_df[name] / equity_df[name].iloc[0], label=name)
        plt.title(f"RAC-MAP prototype equity curves: {period}")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(result_dir / f"equity_{period}.png", dpi=160)
        plt.close()
    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="work/rac_map/data/features_daily.parquet")
    parser.add_argument("--model-dir", default="work/rac_map/models")
    parser.add_argument("--result-dir", default="work/rac_map/results")
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()
    metrics = run_experiment(args.data, args.model_dir, args.result_dir, args.timesteps, args.seed, train=not args.no_train)
    print(metrics.to_string(index=False))
