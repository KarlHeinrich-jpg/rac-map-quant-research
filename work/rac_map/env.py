"""Gymnasium portfolio environment used by FinRL-compatible and RAC-MAP agents."""

from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

try:
    from .panel import MarketPanel, softmax
except ImportError:  # direct script execution
    from panel import MarketPanel, softmax


class PortfolioEnv(Env):
    """Long-only allocation environment with close-to-close execution.

    The contract mirrors the portfolio-allocation environments in FinRL and
    the long-only action normalization in MASA. An action is a vector of
    logits in [-1, 1], normalized to portfolio weights with softmax.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: MarketPanel,
        start_idx: int = 0,
        end_idx: int | None = None,
        transaction_cost: float = 0.001,
        reward_mode: str = "finrl",
        cvar_window: int = 60,
        risk_aversion: float = 0.1,
        turnover_aversion: float = 0.01,
    ) -> None:
        super().__init__()
        self.panel = panel
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx if end_idx is not None else len(panel.dates) - 1)
        if self.end_idx - self.start_idx < 2:
            raise ValueError("Environment requires at least three dates")
        self.transaction_cost = float(transaction_cost)
        self.reward_mode = reward_mode
        self.cvar_window = int(cvar_window)
        self.risk_aversion = float(risk_aversion)
        self.turnover_aversion = float(turnover_aversion)
        obs_dim = panel.n_assets * panel.n_features + panel.n_assets + panel.market.shape[1]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(panel.n_assets,), dtype=np.float32)
        self.t = self.start_idx
        self.weights = np.ones(panel.n_assets, dtype=np.float64) / panel.n_assets
        self.value = 1_000_000.0
        self.returns_memory: list[float] = []
        self.turnover_memory: list[float] = []

    def _observation(self) -> np.ndarray:
        obs = np.concatenate(
            [
                np.nan_to_num(self.panel.features[self.t].reshape(-1), nan=0.0, posinf=0.0, neginf=0.0),
                self.weights,
                np.nan_to_num(self.panel.market[self.t], nan=0.0, posinf=0.0, neginf=0.0),
            ]
        )
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = self.start_idx
        self.weights = np.ones(self.panel.n_assets, dtype=np.float64) / self.panel.n_assets
        self.value = 1_000_000.0
        self.returns_memory = []
        self.turnover_memory = []
        return self._observation(), {"date": self.panel.dates[self.t], "value": self.value}

    def step(self, action: np.ndarray):
        logits = np.asarray(action, dtype=np.float64).reshape(-1)
        if logits.shape[0] != self.panel.n_assets:
            raise ValueError(f"Expected {self.panel.n_assets} actions, got {logits.shape[0]}")
        target = softmax(logits)
        turnover = 0.5 * float(np.abs(target - self.weights).sum())
        next_ret = float(np.dot(target, self.panel.returns[self.t + 1]))
        net_ret = (1.0 + next_ret) * (1.0 - self.transaction_cost * turnover) - 1.0
        self.value *= 1.0 + net_ret
        self.returns_memory.append(net_ret)
        self.turnover_memory.append(turnover)
        recent = np.asarray(self.returns_memory[-self.cvar_window :], dtype=np.float64)
        q05 = float(np.quantile(recent, 0.05)) if len(recent) >= 5 else 0.0
        cvar = float(recent[recent <= q05].mean()) if len(recent) >= 5 and np.any(recent <= q05) else q05
        vol = float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
        reward = float(np.log(max(1.0 + net_ret, 1e-8)))
        if self.reward_mode in {"masa", "rac_map"}:
            reward -= self.risk_aversion * max(0.0, -cvar)
        if self.reward_mode == "rac_map":
            reward -= self.turnover_aversion * turnover
        self.weights = target
        self.t += 1
        terminated = self.t >= self.end_idx
        info = {
            "date": self.panel.dates[self.t],
            "value": self.value,
            "net_return": net_ret,
            "turnover": turnover,
            "cvar": cvar,
            "volatility": vol,
            "weights": target.copy(),
        }
        return self._observation(), reward, terminated, False, info
