"""Policy implementations for transparent baselines and RAC-MAP."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .panel import project_long_only, softmax
except ImportError:  # direct script execution
    from panel import project_long_only, softmax


class EqualWeightPolicy:
    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        return np.ones(self.n_assets, dtype=float) / self.n_assets


class BuyAndHoldPolicy(EqualWeightPolicy):
    """Equal-weight initial allocation with weights drifting thereafter."""

    def reset(self, n_assets: int) -> None:
        super().reset(n_assets)
        self._weights: np.ndarray | None = None

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        if self._weights is None:
            self._weights = np.ones(self.n_assets, dtype=float) / self.n_assets
        return self._weights.copy()

    def observe(self, next_ret: float, target: np.ndarray, context: dict[str, Any]) -> None:
        asset_returns = np.asarray(context.get("realized_asset_returns", []), dtype=float)
        if asset_returns.shape == (self.n_assets,) and self._weights is not None:
            drifted = self._weights * (1.0 + asset_returns)
            self._weights = project_long_only(drifted)


class RiskParityPolicy:
    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        vol = np.asarray(context["features"][:, 3], dtype=float)
        inv = 1.0 / np.maximum(vol, 0.05)
        return project_long_only(inv)


class StaticMultiAgentPolicy:
    """MASA-style static mixture without a market observer or adaptive gate."""

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def _agent_weights(self, features: np.ndarray) -> list[np.ndarray]:
        mom20, mom60, vol, sma, rsi = features[:, 1], features[:, 2], features[:, 3], features[:, 4], features[:, 5]
        signals = [
            0.8 * mom20 + 0.4 * mom60 + 0.2 * sma,
            -vol + 0.2 * sma,
            sma + 0.3 * mom20,
            -0.5 * mom20 - vol,
        ]
        return [softmax(s, temperature=0.4) for s in signals]

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        return np.mean(self._agent_weights(context["features"]), axis=0)


class MASACompatiblePolicy(StaticMultiAgentPolicy):
    """Transparent reproduction of MASA's two-agent + market-observer logic.

    The upstream MASA implementation uses TD3 actors, a market observer and a
    controller barrier function. This compatible policy retains those three
    information flows while keeping the experiment runnable in modern Python:
    profit and risk actors, observer-conditioned mixing, and an adaptive risk
    tilt. It is therefore a reproducible approximation, not a claim of byte-
    identical upstream execution.
    """

    def reset(self, n_assets: int) -> None:
        super().reset(n_assets)
        self.agent_performance = np.ones(2, dtype=float)
        self._last_agent_weights: list[np.ndarray] | None = None

    def _agent_weights(self, features: np.ndarray) -> list[np.ndarray]:
        mom20, mom60, vol, sma = features[:, 1], features[:, 2], features[:, 3], features[:, 4]
        profit = softmax(0.8 * mom20 + 0.4 * mom60 + 0.2 * sma, temperature=0.4)
        risk = softmax(-vol - 0.5 * np.maximum(-sma, 0.0), temperature=0.4)
        return [profit, risk]

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        agents = self._agent_weights(context["features"])
        regime = float(context["market"][4])
        bull = 1.0 / (1.0 + np.exp(-2.0 * regime))
        # The observer makes the risk actor dominant in stressed/down regimes.
        logits = np.log(self.agent_performance + 1e-6) + np.array([1.5 * bull, 1.5 * (1.0 - bull)])
        gate = softmax(logits)
        self._last_agent_weights = agents
        return project_long_only(sum(g * a for g, a in zip(gate, agents)))

    def observe(self, next_ret: float, target: np.ndarray, context: dict[str, Any]) -> None:
        if self._last_agent_weights is None:
            return
        asset_returns = np.asarray(context.get("realized_asset_returns", []), dtype=float)
        if asset_returns.shape != (self.n_assets,):
            return
        realized = np.array([float(np.dot(w, asset_returns)) for w in self._last_agent_weights])
        self.agent_performance = 0.95 * self.agent_performance + 0.05 * np.exp(np.clip(realized, -5, 5))


class RACMAPPolicy(StaticMultiAgentPolicy):
    """Regime-adaptive, confidence-weighted multi-agent portfolio policy."""

    agent_names = ("momentum", "trend", "low_vol", "defensive", "mean_reversion")

    def __init__(self, use_regime: bool = True, use_confidence: bool = True, use_risk: bool = True) -> None:
        self.use_regime = use_regime
        self.use_confidence = use_confidence
        self.use_risk = use_risk

    def reset(self, n_assets: int) -> None:
        super().reset(n_assets)
        self.agent_performance = np.ones(len(self.agent_names), dtype=float)
        self._last_agent_weights: list[np.ndarray] | None = None
        self._last_gate = np.ones(len(self.agent_names), dtype=float) / len(self.agent_names)

    def _signals(self, features: np.ndarray) -> np.ndarray:
        mom20, mom60, vol, sma, rsi = features[:, 1], features[:, 2], features[:, 3], features[:, 4], features[:, 5]
        return np.vstack(
            [
                0.8 * mom20 + 0.4 * mom60 + 0.2 * sma,
                1.0 * sma + 0.3 * mom20,
                -1.0 * vol + 0.1 * sma,
                -0.5 * mom20 - 1.0 * vol,
                -0.8 * mom20 + 0.2 * (0.5 - rsi),
            ]
        )

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        features = np.asarray(context["features"], dtype=float)
        signals = self._signals(features)
        vol = np.maximum(features[:, 3], 0.05)
        agent_weights = [softmax(signal, temperature=0.4) for signal in signals]
        confidence = np.asarray([np.mean(np.abs(signal) / vol) for signal in signals])
        confidence = confidence / (np.mean(confidence) + 1e-6)
        agent_risk = np.asarray([np.dot(w, vol) for w in agent_weights])
        regime = float(context["market"][4])
        market_vol = float(context["market"][1])
        bull = 1.0 / (1.0 + np.exp(-2.0 * regime))
        stress = np.clip(2.0 * market_vol, 0.0, 2.0) + (1.0 - bull)
        regime_preference = np.array(
            [1.1 * bull, 0.8 * bull, 0.8 * stress, 1.2 * stress, 0.5 * (1.0 - bull)]
        ) if self.use_regime else np.zeros(len(self.agent_names))
        gate_logits = (
            np.log(self.agent_performance + 1e-6)
            + (0.8 * confidence if self.use_confidence else 0.0)
            - (1.5 * agent_risk if self.use_risk else 0.0)
            + regime_preference
        )
        gate = softmax(gate_logits, temperature=0.7)
        self._last_gate = gate
        self._last_agent_weights = agent_weights
        return project_long_only(sum(g * a for g, a in zip(gate, agent_weights)))

    def observe(self, next_ret: float, target: np.ndarray, context: dict[str, Any]) -> None:
        if self._last_agent_weights is None:
            return
        asset_returns = np.asarray(context.get("realized_asset_returns", []), dtype=float)
        if asset_returns.shape != (self.n_assets,):
            return
        realized = np.array([float(np.dot(w, asset_returns)) for w in self._last_agent_weights])
        self.agent_performance = 0.96 * self.agent_performance + 0.04 * np.exp(np.clip(realized, -5, 5))


class SB3Policy:
    """Adapter for a Stable-Baselines3 model in the common backtest loop."""

    def __init__(self, model: Any, n_assets: int, n_features: int, market_dim: int) -> None:
        self.model = model
        self.n_assets = n_assets
        self.n_features = n_features
        self.market_dim = market_dim

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        obs = np.concatenate(
            [context["features"].reshape(-1), context["previous_weights"], context["market"]]
        ).astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return softmax(np.asarray(action).reshape(-1))


class MASATD3Policy:
    """Combine two trained TD3 actors with the MASA observer-style gate."""

    def __init__(self, profit_model: Any, risk_model: Any, n_assets: int) -> None:
        self.profit_model = profit_model
        self.risk_model = risk_model
        self.n_assets = n_assets
        self.agent_performance = np.ones(2, dtype=float)
        self._last_agent_weights: list[np.ndarray] | None = None

    def reset(self, n_assets: int) -> None:
        self.n_assets = n_assets
        self.agent_performance = np.ones(2, dtype=float)
        self._last_agent_weights = None

    def get_weights(self, context: dict[str, Any]) -> np.ndarray:
        obs = np.concatenate(
            [context["features"].reshape(-1), context["previous_weights"], context["market"]]
        ).astype(np.float32)
        profit_action, _ = self.profit_model.predict(obs, deterministic=True)
        risk_action, _ = self.risk_model.predict(obs, deterministic=True)
        profit = softmax(np.asarray(profit_action).reshape(-1))
        risk = softmax(np.asarray(risk_action).reshape(-1))
        regime = float(context["market"][4])
        bull = 1.0 / (1.0 + np.exp(-2.0 * regime))
        gate = softmax(np.log(self.agent_performance + 1e-6) + np.array([1.5 * bull, 1.5 * (1 - bull)]))
        self._last_agent_weights = [profit, risk]
        return project_long_only(gate[0] * profit + gate[1] * risk)

    def observe(self, next_ret: float, target: np.ndarray, context: dict[str, Any]) -> None:
        if self._last_agent_weights is None:
            return
        asset_returns = np.asarray(context.get("realized_asset_returns", []), dtype=float)
        if asset_returns.shape != (self.n_assets,):
            return
        realized = np.array([float(np.dot(w, asset_returns)) for w in self._last_agent_weights])
        self.agent_performance = 0.95 * self.agent_performance + 0.05 * np.exp(np.clip(realized, -5, 5))
