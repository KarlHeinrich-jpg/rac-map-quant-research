"""梯度数值检验：验证 model.train_gate 中解析梯度与有限差分一致。

方法：把 loss 与梯度计算抽出为纯函数 ``loss_and_grad(params)``，对每个参数做
中心差分，与解析梯度比较最大相对误差。任何相对误差超过 1e-5 即视为实现错误。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import EXPERT_NAMES, GateConfig, softmax_rows  # noqa: E402


def make_problem(seed: int = 0, T: int = 30, K: int = 5, N: int = 40, R: int = 6, H: int = 8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, R + 3 * K))
    # 布局约定为 (K, T, N)：对最后一维（资产）做 softmax 即得到每个专家的组合权重
    logits = rng.normal(size=(K, T, N))
    expert_w = softmax_rows(logits, temperature=1.0)
    asset_ret = rng.normal(0.0, 0.05, size=(T, N))
    params = {
        "W1": rng.normal(0, 0.3, size=(x.shape[1], H)),
        "b1": rng.normal(0, 0.1, size=H),
        "W2": rng.normal(0, 0.3, size=(H, K)),
        "b2": rng.normal(0, 0.1, size=K),
    }
    return x, expert_w, asset_ret, params


def forward_from_params(x, params, temperature=1.0):
    h = np.tanh(x @ params["W1"] + params["b1"])
    logits = (h @ params["W2"] + params["b2"]) / temperature
    return softmax_rows(logits, temperature=1.0), h


def loss_and_grad(x, expert_w, asset_ret, params, cfg: GateConfig):
    T, K = expert_w.shape[1], expert_w.shape[0]
    r = asset_ret
    g, h = forward_from_params(x, params, cfg.temperature)
    port = np.einsum("tk,ktn->tn", g, expert_w)
    port_ret = np.einsum("tn,tn->t", port, r)
    static_ret = np.einsum("tn,tn->t", expert_w.mean(axis=0), r)
    target = port_ret - static_ret if cfg.objective == "active" else port_ret

    sd = target.std() + 1e-12
    mean_t = target.mean()
    q = np.quantile(target, 0.05)
    tail_mask = (target <= q).astype(float)
    n_tail = max(tail_mask.sum(), 1.0)
    cvar = float(-(target * tail_mask).sum() / n_tail)
    turn_vec = 0.5 * np.abs(port[1:] - port[:-1]).sum(axis=1)
    turn_mean = float(turn_vec.mean()) if turn_vec.size else 0.0
    ent = float(-(g * np.log(np.clip(g, 1e-12, None))).sum(axis=1).mean())

    if cfg.objective == "active":
        loss = -mean_t / (cfg.ir_scale * sd)
        loss += cfg.risk_penalty * cvar / sd
    else:
        loss = -mean_t * cfg.ir_scale
        loss += cfg.risk_penalty * cvar
    loss += cfg.turnover_penalty * turn_mean
    loss += cfg.entropy_penalty * ent
    loss += cfg.l2 * float((params["W1"] ** 2).sum() + (params["W2"] ** 2).sum())

    # ---- 解析梯度（与 model.train_gate 完全一致）----
    n_trans = max(T - 1, 1)
    d_r = np.zeros(T)
    if cfg.objective == "active":
        # L1 = -mean/(s*sd)  =>  dL1/dr_t = -1/(T*s*sd) + mean*(r_t-mean)/(T*s*sd^3)
        d_r += -1.0 / (T * cfg.ir_scale * sd)
        d_r += mean_t * (target - mean_t) / (T * cfg.ir_scale * sd**3)
        # L2 = lam_R * cvar/sd
        #  => dL2/dr_t = lam_R*(-mask_t/n_tail)/sd - lam_R*cvar*(r_t-mean)/(T*sd^3)
        d_r += cfg.risk_penalty * (-tail_mask / n_tail) / sd
        d_r -= cfg.risk_penalty * cvar * (target - mean_t) / (T * sd**3)
    else:
        d_r += -cfg.ir_scale / T
        d_r += cfg.risk_penalty * (-tail_mask / n_tail)

    # 组合收益对门控权重的导数：d port_ret_t / d g_{t,k} = 专家 k 在 t 期的组合收益
    expert_port_ret = np.einsum("ktn,tn->tk", expert_w, r)   # (T, K)
    d_g = expert_port_ret * d_r[:, None]

    if turn_vec.size and cfg.turnover_penalty:
        sign = np.sign(port[1:] - port[:-1])
        coef = np.zeros_like(port)
        coef[:-1] -= 0.5 * sign
        coef[1:] += 0.5 * sign
        d_g += cfg.turnover_penalty / n_trans * np.einsum("tn,ktn->tk", coef, expert_w)

    d_g -= cfg.entropy_penalty / T * (np.log(np.clip(g, 1e-12, None)) + 1.0)

    d_logits = g * (d_g - (d_g * g).sum(axis=1, keepdims=True))
    grads = {"W2": h.T @ d_logits + 2 * cfg.l2 * params["W2"], "b2": d_logits.sum(axis=0)}
    d_h = (d_logits @ params["W2"].T) * (1.0 - h**2)
    grads["W1"] = x.T @ d_h + 2 * cfg.l2 * params["W1"]
    grads["b1"] = d_h.sum(axis=0)
    return loss, grads


def check(cfg: GateConfig, seed: int = 0, eps: float = 1e-6, tol: float = 1e-4) -> float:
    x, expert_w, asset_ret, params = make_problem(seed)
    loss, grads = loss_and_grad(x, expert_w, asset_ret, params, cfg)
    worst = 0.0
    for key, par in params.items():
        flat = par.reshape(-1)
        analytic = grads[key].reshape(-1)
        idx = np.linspace(0, flat.size - 1, min(12, flat.size)).astype(int)
        for i in idx:
            orig = flat[i]
            flat[i] = orig + eps
            lp, _ = loss_and_grad(x, expert_w, asset_ret, params, cfg)
            flat[i] = orig - eps
            lm, _ = loss_and_grad(x, expert_w, asset_ret, params, cfg)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(abs(num), abs(analytic[i]), 1e-8)
            rel = abs(num - analytic[i]) / denom
            worst = max(worst, rel)
    status = "OK " if worst < tol else "FAIL"
    print(f"  {status} objective={cfg.objective:8s} risk={cfg.risk_penalty} turn={cfg.turnover_penalty} "
          f"ent={cfg.entropy_penalty} -> max rel err {worst:.3e}")
    return worst


if __name__ == "__main__":
    print("梯度数值检验（中心差分 vs 解析梯度）")
    worst = 0.0
    for objective in ("active", "absolute"):
        for risk, turn, ent, l2 in [
            (0.0, 0.0, 0.0, 0.0),
            (0.30, 0.0, 0.0, 0.0),
            (0.0, 0.20, 0.0, 0.0),
            (0.0, 0.0, 0.01, 0.0),
            (0.0, 0.0, 0.0, 1e-4),
            (0.30, 0.20, 0.01, 1e-4),
        ]:
            cfg = GateConfig(objective=objective, risk_penalty=risk, turnover_penalty=turn,
                             entropy_penalty=ent, l2=l2)
            worst = max(worst, check(cfg))
    print(f"\n总体最大相对误差: {worst:.3e}")
    print("结论:", "梯度实现正确" if worst < 1e-4 else "存在梯度错误，需修正")
