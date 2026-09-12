from __future__ import annotations

import numpy as np
from pathlib import Path

from work.rac_map.env import PortfolioEnv
from work.rac_map.panel import load_panel, run_backtest
from work.rac_map.policies import MASACompatiblePolicy, RACMAPPolicy
from work.rac_map.rac_map_gate import RACMAPGate
from work.rac_map.joint_gate import JointFitConfig, fit_joint_rac_map


DATA = str(Path(__file__).parent / "data" / "features_daily.parquet")


def test_panel_and_rac_map_weights():
    panel = load_panel(DATA)
    policy = RACMAPPolicy()
    policy.reset(panel.n_assets)
    weights = policy.get_weights(
        {"features": panel.features[100], "market": panel.market[100], "previous_weights": np.ones(panel.n_assets) / panel.n_assets}
    )
    assert weights.shape == (panel.n_assets,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)


def test_environment_contract():
    panel = load_panel(DATA)
    env = PortfolioEnv(panel, start_idx=100, end_idx=110, reward_mode="rac_map")
    obs, info = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    next_obs, reward, terminated, truncated, step_info = env.step(np.zeros(panel.n_assets))
    assert next_obs.shape == obs.shape
    assert np.isfinite(reward)
    assert not truncated
    assert "turnover" in step_info


def test_backtest_is_finite():
    panel = load_panel(DATA)
    equity, metrics = run_backtest(panel, MASACompatiblePolicy(), "2024-01-01", "2024-03-31")
    assert len(equity) > 10
    assert np.isfinite(metrics["sharpe"])


def test_learnable_gate_contract():
    import torch

    gate = RACMAPGate(n_agents=5, market_dim=5)
    actions = torch.randn(2, 5, 10)
    action, weights = gate(
        actions,
        torch.randn(2, 5),
        torch.rand(2, 5),
        torch.rand(2, 5),
        torch.rand(2, 5) + 0.1,
    )
    assert action.shape == (2, 10)
    assert weights.shape == (2, 5)
    assert torch.allclose(weights.sum(-1), torch.ones(2), atol=1e-5)


def test_joint_rac_map_training_contract():
    panel = load_panel(DATA)
    policy = fit_joint_rac_map(panel, 100, 180, JointFitConfig(epochs=2, seed=7))
    weights = policy.get_weights({"features": panel.features[180], "market": panel.market[180]})
    assert weights.shape == (panel.n_assets,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
