"""Train FinRL-compatible PPO and MASA-compatible TD3 baselines.

The defaults are deliberately small enough for a smoke test. Increase
``--timesteps`` and use multiple seeds before treating any number as evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, TD3

try:
    from .env import PortfolioEnv
    from .panel import load_panel
except ImportError:  # direct script execution
    from env import PortfolioEnv
    from panel import load_panel


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_indices(panel, train_end: str, valid_end: str, test_end: str) -> dict[str, tuple[int, int]]:
    dates = panel.dates
    train_start = int(np.flatnonzero(dates >= pd.Timestamp("2010-01-01"))[0])
    train_end_idx = int(np.flatnonzero(dates <= pd.Timestamp(train_end))[-1])
    valid_start = train_end_idx
    valid_end_idx = int(np.flatnonzero(dates <= pd.Timestamp(valid_end))[-1])
    test_start = valid_end_idx
    test_end_idx = int(np.flatnonzero(dates <= pd.Timestamp(test_end))[-1])
    return {"train": (train_start, train_end_idx), "valid": (valid_start, valid_end_idx), "test": (test_start, test_end_idx)}


def train_models(data_path: str | Path, model_dir: str | Path, timesteps: int = 5_000, seed: int = 42) -> dict[str, str]:
    _seed_everything(seed)
    panel = load_panel(data_path)
    dates = panel.dates
    train_start = int(np.flatnonzero(dates >= pd.Timestamp("2010-01-01"))[0])
    train_end = int(np.flatnonzero(dates <= pd.Timestamp("2021-12-31"))[-1])
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def make_env(reward_mode: str):
        return PortfolioEnv(
            panel,
            start_idx=train_start,
            end_idx=train_end,
            reward_mode=reward_mode,
            transaction_cost=0.001,
            risk_aversion=0.1,
            turnover_aversion=0.01,
        )

    # FinRL's canonical stock-trading examples use PPO with a profit-oriented reward.
    finrl_env = make_env("finrl")
    ppo_steps = max(256, int(timesteps))
    ppo = PPO(
        "MlpPolicy",
        finrl_env,
        seed=seed,
        n_steps=min(256, max(16, train_end - train_start)),
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        device="cpu",
    )
    ppo.learn(total_timesteps=ppo_steps, progress_bar=False)
    ppo_path = model_dir / "finrl_ppo"
    ppo.save(str(ppo_path))

    # MASA uses TD3 actors with a profit/risk controller. Two TD3 actors are
    # trained under different rewards and combined by MASACompatiblePolicy.
    masa_paths: dict[str, str] = {}
    for name, reward_mode, offset in (("profit", "finrl", 0), ("risk", "masa", 1)):
        td3_env = make_env(reward_mode)
        td3 = TD3(
            "MlpPolicy",
            td3_env,
            seed=seed + offset,
            learning_rate=1e-3,
            buffer_size=min(50_000, max(2_000, timesteps * 10)),
            learning_starts=min(100, max(1, timesteps // 10)),
            batch_size=64,
            train_freq=(1, "episode"),
            gradient_steps=1,
            policy_delay=2,
            verbose=0,
            device="cpu",
        )
        td3.learn(total_timesteps=ppo_steps, progress_bar=False)
        path = model_dir / f"masa_{name}_td3"
        td3.save(str(path))
        masa_paths[name] = str(path) + ".zip"

    manifest = {
        "data_path": str(Path(data_path).resolve()),
        "train_start": str(dates[train_start].date()),
        "train_end": str(dates[train_end].date()),
        "timesteps_per_model": ppo_steps,
        "seed": seed,
        "finrl_ppo": str(ppo_path) + ".zip",
        **{f"masa_{k}_td3": v for k, v in masa_paths.items()},
        "implementation_note": "FinRL-compatible PPO and MASA-inspired dual-TD3 reproduction in current SB3/Gymnasium; not byte-identical upstream execution.",
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"finrl_ppo": str(ppo_path) + ".zip", **{f"masa_{k}_td3": v for k, v in masa_paths.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="work/rac_map/data/features_daily.parquet")
    parser.add_argument("--model-dir", default="work/rac_map/models")
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # Import here to keep module import light for users only running --help.
    import pandas as pd

    print(json.dumps(train_models(args.data, args.model_dir, args.timesteps, args.seed), indent=2))
