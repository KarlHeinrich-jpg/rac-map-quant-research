r"""专家信号、组合映射与门控网络。

专家设计
--------
五个专家分别对应中国 A 股文献中稳健性较高的选股逻辑。打分是**线性可解释**的，
载荷方向由金融理论设定，不参与自由训练（见理论部分 5.4 节的识别性讨论）：

============================  ================================================
专家                        打分
============================  ================================================
动量 (momentum)              $0.7\,mom_{12-1}+0.3\,mom_{12}$
短期反转 (reversal)          $-1.0\,mom_{1}$
低波动 (low_vol)             $-1.0\,vol_{60}$
低换手 (low_turnover)        $-0.6\,turnover_{20}+0.4\,mom_{12-1}$
低 beta (low_beta)           $-1.0\,\beta_{120}+0.3\,mom_{12}$
============================  ================================================

组合映射
--------
打分经截面标准化后通过带温度的 softmax 映射为长仓满仓权重
$w_{k,i}\propto\exp(\theta_{k,i}/\tau)$。$\tau$ 越大越接近等权。

门控
----
门控输入包含市场状态、专家置信度、专家风险与专家近期表现，输出专家权重
$g\in\Delta^{K-1}$，最终组合 $w=\sum_k g_k a_k$。

训练目标
--------
**主动收益型**（本文修正，见理论 5.2 节）
``L = -IR(r_active) + λ_R·CVaR5(r_active) + λ_U·turnover + λ_H·H(g) + λ_2‖W‖²``
其中 ``r_active = (w_gate - w_static)ᵀ r``，``w_static`` 为等权专家混合。

**绝对收益型**（原稿目标，用于 E7 对照）
``L = -r_absolute + λ_R·CVaR5(r_absolute) + λ_U·turnover + λ_H·H(g)``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EXPERT_NAMES = ("momentum", "reversal", "low_vol", "low_turnover", "low_beta")

EXPERT_FEATURES = ("mom120", "mom20", "mom250", "vol60", "turnover20", "beta120", "amihud")

EXPERT_LOADINGS = np.array(
    [
        # mom120  mom20  mom250  vol60  turnover20  beta120  amihud
        [0.70, -1.00, 0.30, 0.00, 0.00, 0.00, 0.00],   # momentum: 12-1 动量
        [0.00, -1.00, 0.00, 0.00, 0.00, 0.00, 0.00],   # reversal: 短期反转
        [0.00, 0.00, 0.00, -1.00, 0.00, 0.00, 0.00],   # low_vol: 低波动
        [0.40, 0.00, 0.00, 0.00, -0.60, 0.00, 0.00],   # low_turnover: 低换手
        [0.30, 0.00, 0.00, 0.00, 0.00, -1.00, 0.00],   # low_beta: 低 beta
    ],
    dtype=float,
)


def softmax_rows(scores: np.ndarray, temperature: float = 0.6) -> np.ndarray:
    """按最后一维做带温度的 softmax；非有限值权重置零。"""
    z = np.asarray(scores, dtype=float) / max(temperature, 1e-6)
    valid = np.isfinite(z)
    if not valid.any():
        return np.zeros_like(z)
    zmax = np.max(np.where(valid, z, -np.inf), axis=-1, keepdims=True)
    e = np.where(valid, np.exp(np.clip(z - zmax, -60, 0)), 0.0)
    s = e.sum(axis=-1, keepdims=True)
    return np.divide(e, s, out=np.zeros_like(e), where=s > 1e-12)


def expert_scores(features: dict[str, np.ndarray]) -> np.ndarray:
    """返回 (K, T, N) 的专家打分。"""
    T, N = next(iter(features.values())).shape
    K = EXPERT_LOADINGS.shape[0]
    out = np.zeros((K, T, N))
    for k in range(K):
        acc = np.zeros((T, N))
        wsum = 0.0
        for j, name in enumerate(EXPERT_FEATURES):
            w = EXPERT_LOADINGS[k, j]
            if abs(w) < 1e-12 or name not in features:
                continue
            acc += w * features[name]
            wsum += abs(w)
        out[k] = acc / max(wsum, 1e-12)
    return out


def expert_weights(scores: np.ndarray, universe: np.ndarray, temperature: float = 0.6) -> np.ndarray:
    """专家打分 -> (K, T, N) 长仓满仓权重。"""
    masked = np.where(universe[None, :, :] > 0, scores, np.nan)
    filled = np.where(np.isfinite(masked), masked, -1e6)
    return softmax_rows(filled, temperature=temperature)


def expert_confidence(scores: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """专家置信度：截面打分离散度（越大表示观点越强）。"""
    masked = np.where(universe[None, :, :] > 0, scores, np.nan)
    with np.errstate(invalid="ignore"):
        conf = np.nanstd(masked, axis=-1)
    return np.nan_to_num(conf, nan=0.0)


def expert_risk(weights: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """专家风险：组合加权平均波动率。"""
    return np.nansum(weights * np.nan_to_num(vol, nan=0.0)[None, :, :], axis=-1)


def recent_performance(expert_returns: np.ndarray, decay: float = 0.85) -> np.ndarray:
    """专家近期表现：指数加权，且只用 t 期之前的信息。返回 (K, T)。"""
    K, T = expert_returns.shape
    out = np.ones((K, T))
    state = np.ones(K)
    for t in range(T):
        out[:, t] = state
        state = decay * state + (1.0 - decay) * np.exp(np.clip(expert_returns[:, t], -0.5, 0.5))
    return out


@dataclass
class GateConfig:
    hidden: int = 24
    lr: float = 3e-2
    epochs: int = 200
    risk_penalty: float = 0.30
    turnover_penalty: float = 0.20
    entropy_penalty: float = 0.01
    temperature: float = 1.0
    seed: int = 42
    objective: str = "active"       # "active" | "absolute"
    l2: float = 1e-4
    ir_scale: float = 10.0

    def as_dict(self) -> dict:
        return {
            "hidden": self.hidden, "lr": self.lr, "epochs": self.epochs,
            "risk_penalty": self.risk_penalty, "turnover_penalty": self.turnover_penalty,
            "entropy_penalty": self.entropy_penalty, "temperature": self.temperature,
            "seed": self.seed, "objective": self.objective, "l2": self.l2,
        }


class GatingModel:
    """两层 tanh 门控网络：状态 + 专家置信度/风险/近期表现 -> 专家权重。"""

    def __init__(self, n_regime: int, n_experts: int, config: GateConfig) -> None:
        self.cfg = config
        self.n_experts = n_experts
        rng = np.random.default_rng(config.seed)
        self.d_in = n_regime + 3 * n_experts
        scale = 1.0 / np.sqrt(self.d_in)
        self.W1 = rng.normal(0.0, scale, size=(self.d_in, config.hidden))
        self.b1 = np.zeros(config.hidden)
        self.W2 = rng.normal(0.0, scale, size=(config.hidden, n_experts))
        self.b2 = np.zeros(n_experts)
        self.norm: tuple[np.ndarray, np.ndarray] | None = None

    def params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def make_input(self, regime: np.ndarray, conf: np.ndarray, risk: np.ndarray, perf: np.ndarray) -> np.ndarray:
        parts = [
            np.nan_to_num(regime, nan=0.0),
            np.log(np.clip(conf, 1e-4, None)),
            np.log(np.clip(risk, 1e-4, None)),
            np.log(np.clip(perf, 1e-4, None)),
        ]
        x = np.concatenate(parts, axis=1)
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True) + 1e-8
        return (x - mu) / sd

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (专家权重 g, 隐层 h)。"""
        h = np.tanh(x @ self.W1 + self.b1)
        logits = (h @ self.W2 + self.b2) / max(self.cfg.temperature, 1e-6)
        return softmax_rows(logits, temperature=1.0), h

    def save(self) -> dict:
        return {k: v.copy() for k, v in self.params().items()} | {
            "norm": self.norm, "cfg": self.cfg.as_dict(), "n_experts": self.n_experts, "d_in": self.d_in
        }

    @classmethod
    def load(cls, blob: dict) -> "GatingModel":
        cfg = GateConfig(**blob["cfg"])
        model = cls(blob["d_in"] - 3 * blob["n_experts"], blob["n_experts"], cfg)
        for key, value in blob.items():
            if key in {"W1", "b1", "W2", "b2"}:
                setattr(model, key, np.asarray(value, dtype=float))
        model.norm = blob.get("norm")
        return model


def train_gate(
    regime: np.ndarray,
    conf: np.ndarray,
    risk: np.ndarray,
    perf: np.ndarray,
    expert_w: np.ndarray,
    asset_ret: np.ndarray,
    config: GateConfig,
    asset_vol: np.ndarray | None = None,
) -> tuple[GatingModel, dict]:
    """训练门控网络（numpy 手写 Adam）。

    参数
    ----
    regime    : (T, R) 市场状态变量
    conf      : (T, K) 专家置信度
    risk      : (T, K) 专家风险
    perf      : (T, K) 专家近期表现
    expert_w  : (K, T, N) 专家组合权重
    asset_ret : (T, N) 持有期收益
    """
    T, K = conf.shape
    N = asset_ret.shape[1]
    r = np.nan_to_num(asset_ret)

    model = GatingModel(regime.shape[1], K, config)
    x = model.make_input(regime, conf, risk, perf)

    expert_port_ret = np.einsum("ktn,tn->tk", expert_w, r)     # (T, K)
    static_w = expert_w.mean(axis=0)                            # (T, N)
    static_ret = np.einsum("tn,tn->t", static_w, r)             # (T,)

    state = {k: np.zeros_like(v) for k, v in model.params().items()}
    velocity = {k: np.zeros_like(v) for k, v in model.params().items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history: list[dict] = []
    n_trans = max(T - 1, 1)

    for step in range(1, config.epochs + 1):
        g, h = model.forward(x)                                 # (T, K), (T, H)
        port = np.einsum("tk,ktn->tn", g, expert_w)             # (T, N)
        port_ret = np.einsum("tn,tn->t", port, r)
        active_ret = port_ret - static_ret

        if config.objective == "active":
            target = active_ret
        else:
            target = port_ret

        sd = target.std() + 1e-12
        mean_t = target.mean()
        q = np.quantile(target, 0.05)
        tail_mask = (target <= q).astype(float)
        n_tail = max(tail_mask.sum(), 1.0)
        cvar = float(-(target * tail_mask).sum() / n_tail)

        turn_vec = 0.5 * np.abs(port[1:] - port[:-1]).sum(axis=1)
        turn_mean = float(turn_vec.mean()) if turn_vec.size else 0.0
        ent = float(-(g * np.log(np.clip(g, 1e-12, None))).sum(axis=1).mean())

        if config.objective == "active":
            loss = -mean_t / (config.ir_scale * sd)
            loss += config.risk_penalty * cvar / sd
        else:
            loss = -mean_t * config.ir_scale
            loss += config.risk_penalty * cvar
        loss += config.turnover_penalty * turn_mean
        loss += config.entropy_penalty * ent
        loss += config.l2 * float((model.W1**2).sum() + (model.W2**2).sum())

        # ---------------- 解析梯度 ----------------
        # 全部按“均值型”损失推导，均值的 1/T 因子显式写出。
        d_r = np.zeros(T)
        if config.objective == "active":
            # L1 = -mean/(s*sd)
            #  => dL1/dr_t = -1/(T*s*sd) + mean*(r_t-mean)/(T*s*sd^3)
            d_r += -1.0 / (T * config.ir_scale * sd)
            d_r += mean_t * (target - mean_t) / (T * config.ir_scale * sd**3)
            # L2 = lam_R * cvar/sd
            #  => dL2/dr_t = lam_R*(-mask_t/n_tail)/sd - lam_R*cvar*(r_t-mean)/(T*sd^3)
            d_r += config.risk_penalty * (-tail_mask / n_tail) / sd
            d_r -= config.risk_penalty * cvar * (target - mean_t) / (T * sd**3)
        else:
            # L = -s*mean + lam_R*cvar
            d_r += -config.ir_scale / T
            d_r += config.risk_penalty * (-tail_mask / n_tail)

        d_g = expert_port_ret * d_r[:, None]                     # (T, K)

        # 换手惩罚：L3 = lam_U * mean_t(0.5*sum_i |port_t - port_{t-1}|)
        #  => dL3/dport_{t,i} = lam_U/(T-1) * 0.5*(sign(d_t) - sign(d_{t+1}))
        if turn_vec.size and config.turnover_penalty != 0.0:
            sign = np.sign(port[1:] - port[:-1])
            coef = np.zeros_like(port)
            coef[:-1] -= 0.5 * sign
            coef[1:] += 0.5 * sign
            d_g += config.turnover_penalty / n_trans * np.einsum("tn,ktn->tk", coef, expert_w)

        # 熵惩罚：L4 = lam_H * mean_t(H_t) => dL4/dg_{t,k} = lam_H/T * (log g_{t,k} + 1)
        d_g -= config.entropy_penalty / T * (np.log(np.clip(g, 1e-12, None)) + 1.0)

        # 反向传播
        d_logits = g * (d_g - (d_g * g).sum(axis=1, keepdims=True))
        grads = {
            "W2": h.T @ d_logits + 2 * config.l2 * model.W2,
            "b2": d_logits.sum(axis=0),
        }
        d_h = (d_logits @ model.W2.T) * (1.0 - h**2)
        grads["W1"] = x.T @ d_h + 2 * config.l2 * model.W1
        grads["b1"] = d_h.sum(axis=0)

        for key, par in model.params().items():
            gr = np.clip(grads[key], -10.0, 10.0)
            state[key] = beta1 * state[key] + (1 - beta1) * gr
            velocity[key] = beta2 * velocity[key] + (1 - beta2) * gr**2
            mh = state[key] / (1 - beta1**step)
            vh = velocity[key] / (1 - beta2**step)
            par -= config.lr * mh / (np.sqrt(vh) + eps)

        if step % 50 == 0 or step == 1:
            history.append({
                "step": step, "loss": float(loss), "mean_target": float(mean_t),
                "sd_target": float(sd), "cvar": cvar, "turnover": turn_mean, "entropy": ent,
            })

    g_final, _ = model.forward(x)
    diag = {
        "history": history,
        "gate_mean": g_final.mean(axis=0).tolist(),
        "gate_std": g_final.std(axis=0).tolist(),
        "gate_dispersion": float(g_final.std(axis=1).mean()),
        "objective": config.objective,
    }
    return model, diag


def predict_gate(
    model: GatingModel,
    regime: np.ndarray,
    conf: np.ndarray,
    risk: np.ndarray,
    perf: np.ndarray,
) -> np.ndarray:
    x = model.make_input(regime, conf, risk, perf)
    g, _ = model.forward(x)
    return g
