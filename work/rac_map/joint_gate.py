"""Differentiable joint training of RAC-MAP experts and its gate.

The training objective uses only next-day returns inside a rolling training
window.  After fitting, the module is frozen and exposed through the same
``get_weights`` protocol as the deterministic backtest policies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


@dataclass
class JointFitConfig:
    epochs: int = 80
    learning_rate: float = 3e-3
    risk_penalty: float = 0.15
    turnover_penalty: float = 0.02
    entropy_penalty: float = 0.002
    seed: int = 42


class JointRACMAPNet(nn.Module):
    def __init__(self, n_assets: int, n_features: int, market_dim: int, n_experts: int = 5, hidden_dim: int = 48):
        super().__init__()
        self.n_assets = n_assets
        self.n_experts = n_experts
        self.expert_params = nn.Parameter(torch.zeros(n_experts, n_features, 1))
        self.expert_bias = nn.Parameter(torch.zeros(n_experts, 1))
        # Economically interpretable starting experts; all coefficients remain
        # jointly trainable with the gate inside each rolling window.
        initial = torch.zeros(n_experts, n_features)
        if n_experts == 5 and n_features >= 7:
            initial[0, [1, 2, 4]] = torch.tensor([0.8, 0.4, 0.2])  # momentum
            initial[1, [1, 4]] = torch.tensor([0.3, 1.0])           # trend
            initial[2, [3, 4]] = torch.tensor([-1.0, 0.1])         # low volatility
            initial[3, [1, 3]] = torch.tensor([-0.5, -1.0])        # defensive
            initial[4, [1, 5]] = torch.tensor([-0.8, -0.2])        # mean reversion
            self.expert_params.data.copy_(initial.unsqueeze(-1))
            self.expert_bias.data[4] = 0.1
        else:
            nn.init.normal_(self.expert_params, std=0.05)
        self.gate = nn.Sequential(
            nn.Linear(market_dim + n_experts * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_experts),
        )

    def expert_actions(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        signals = torch.einsum("baf,efk->beak", features, self.expert_params).squeeze(-1)
        signals = signals + self.expert_bias.view(1, self.n_experts, 1)
        expert_actions = torch.softmax(signals, dim=-1)
        vol = torch.clamp(features[:, :, 3], min=0.02)
        confidence = torch.mean(torch.abs(signals), dim=-1)
        risk = torch.sum(expert_actions * vol.unsqueeze(1), dim=-1)
        return expert_actions, confidence, risk

    def forward(self, features: Tensor, market: Tensor, recent_performance: Tensor | None = None) -> tuple[Tensor, Tensor]:
        expert_actions, confidence, risk = self.expert_actions(features)
        recent = torch.ones_like(confidence) if recent_performance is None else recent_performance
        gate_input = torch.cat([market, confidence, risk, torch.log(torch.clamp(recent, min=1e-6))], dim=-1)
        gate_weights = torch.softmax(self.gate(gate_input), dim=-1)
        action = torch.sum(gate_weights.unsqueeze(-1) * expert_actions, dim=1)
        return action, gate_weights


class JointRACMAPPolicy:
    agent_names = ("momentum", "trend", "low_vol", "defensive", "mean_reversion")

    def __init__(self, model: JointRACMAPNet):
        self.model = model.eval()
        self.n_assets = model.n_assets
        self.last_gate = np.ones(model.n_experts) / model.n_experts
        self.recent_performance = np.ones(model.n_experts)
        self._last_expert_weights: np.ndarray | None = None

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets
        self.last_gate = np.ones(self.model.n_experts) / self.model.n_experts
        self.recent_performance = np.ones(self.model.n_experts)
        self._last_expert_weights = None

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        features = torch.as_tensor(np.array(context["features"], dtype=np.float32, copy=True)[None])
        market = torch.as_tensor(np.array(context["market"], dtype=np.float32, copy=True)[None])
        recent = torch.as_tensor(self.recent_performance[None], dtype=torch.float32)
        with torch.no_grad():
            experts, _, _ = self.model.expert_actions(features)
            action, gate = self.model(features, market, recent)
        self.last_gate = gate[0].cpu().numpy()
        self._last_expert_weights = experts[0].cpu().numpy()
        return action[0].cpu().numpy()

    def observe(self, next_ret: float, target: np.ndarray, context: dict[str, Any]) -> None:
        asset_returns = np.asarray(context.get("realized_asset_returns", []), dtype=float)
        if self._last_expert_weights is None or asset_returns.shape != (self.n_assets,):
            return
        realized = self._last_expert_weights @ asset_returns
        self.recent_performance = 0.95 * self.recent_performance + 0.05 * np.exp(np.clip(realized, -5, 5))


def fit_joint_rac_map(panel: Any, train_start: int, train_end: int, config: JointFitConfig | None = None) -> JointRACMAPPolicy:
    config = config or JointFitConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if train_end - train_start < 10:
        raise ValueError("joint training window is too short")
    device = torch.device("cpu")
    model = JointRACMAPNet(panel.n_assets, panel.n_features, panel.market.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    features = torch.as_tensor(np.array(panel.features[train_start:train_end], copy=True), dtype=torch.float32, device=device)
    market = torch.as_tensor(np.array(panel.market[train_start:train_end], copy=True), dtype=torch.float32, device=device)
    next_returns = torch.as_tensor(np.array(panel.returns[train_start + 1:train_end + 1], copy=True), dtype=torch.float32, device=device)
    transaction_cost = 0.001
    for _ in range(config.epochs):
        experts, _, _ = model.expert_actions(features)
        expert_returns = torch.sum(experts * next_returns.unsqueeze(1), dim=-1)
        recent_rows = [torch.ones(model.n_experts, device=device)]
        for t in range(1, len(expert_returns)):
            recent_rows.append(0.95 * recent_rows[-1] + 0.05 * torch.exp(torch.clamp(expert_returns[t - 1], -5, 5)))
        recent = torch.stack(recent_rows)
        action, gate = model(features, market, recent)
        turnover = torch.abs(action[1:] - action[:-1]).sum(dim=-1) * 0.5
        gross = torch.sum(action * next_returns, dim=-1)
        net = gross.clone()
        net[1:] = net[1:] - transaction_cost * turnover
        threshold = torch.quantile(net, 0.05)
        cvar_loss = -net[net <= threshold].mean()
        entropy = torch.sum(gate * torch.log(torch.clamp(gate, min=1e-8)), dim=-1)
        loss = -net.mean() + config.risk_penalty * cvar_loss + config.turnover_penalty * turnover.mean() + config.entropy_penalty * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return JointRACMAPPolicy(model)
