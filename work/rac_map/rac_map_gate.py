"""Learnable RAC-MAP gate for joint expert/RL training.

The deterministic :class:`policies.RACMAPPolicy` is useful for transparent
backtests. This module is the neural version intended for a later joint TD3 or
PPO experiment: each expert still produces an action, while the gate learns
how market state, confidence, risk and recent expert performance should be
combined.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RACMAPGate(nn.Module):
    def __init__(self, n_agents: int, market_dim: int = 5, hidden_dim: int = 64, temperature: float = 0.7) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.temperature = float(temperature)
        input_dim = market_dim + 3 * self.n_agents
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.n_agents),
        )

    def forward(
        self,
        expert_actions: Tensor,
        market_state: Tensor,
        confidence: Tensor,
        risk: Tensor,
        recent_performance: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return combined action and gate weights.

        ``expert_actions`` accepts ``[batch, agents, assets]`` or
        ``[agents, assets]``. The other tensors are ``[batch, agents]`` (or
        one-dimensional for a single observation). Gate weights always lie on
        the simplex and therefore preserve long-only portfolio feasibility
        after the experts' own action projection.
        """
        squeeze = expert_actions.ndim == 2
        if squeeze:
            expert_actions = expert_actions.unsqueeze(0)
        tensors = [market_state, confidence, risk, recent_performance]
        tensors = [x.unsqueeze(0) if x.ndim == 1 else x for x in tensors]
        gate_input = torch.cat(
            [tensors[0], tensors[1], tensors[2], torch.log(torch.clamp(tensors[3], min=1e-6))], dim=-1
        )
        logits = self.network(gate_input)
        gate = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        action = torch.sum(gate.unsqueeze(-1) * expert_actions, dim=1)
        if squeeze:
            return action.squeeze(0), gate.squeeze(0)
        return action, gate


def cvar_penalty(returns: Tensor, alpha: float = 0.05) -> Tensor:
    """Differentiable-free empirical CVaR statistic for a reward penalty."""
    if returns.numel() == 0:
        return returns.new_zeros(())
    flat = returns.reshape(-1)
    quantile = torch.quantile(flat, alpha)
    tail = flat[flat <= quantile]
    return tail.mean() if tail.numel() else quantile

