r"""理论核心的显式解与定量分解（命题 5--9 的可计算实现）。

本模块把论文第三部分的显式公式转化为可从数据直接计算的量，用于：
1. 定量分解门控价值的\ **两条渠道**：一阶（收益择时）与二阶（风险配置）；
2. 估计制度转移矩阵与持续性乘数，检验动态拓展的可检验含义；
3. 计算容量约束下的\ **最优状态数** $\hat S^\ast$ 并与实证选择比较。

记号
----
沿用论文：资产 $i=1,\dots,N$；专家 $k=1,\dots,K$；制度 $s=1,\dots,S$；
专家组合权重 $a_{k,t}\in\Delta^{N-1}$；资产条件二阶矩
$\bmu_s=\E[r\mid s]$、$\bSigma_s=\Var(r\mid s)$。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# 制度统计量
# --------------------------------------------------------------------------
@dataclass
class RegimeStats:
    """各制度下专家收益的一阶与二阶统计量。"""

    states: np.ndarray                 # (T,) 制度标签
    expert_returns: np.ndarray         # (T, K) 各专家组合的持有期收益
    state_ids: list[int] = field(default_factory=list)
    pi: np.ndarray = field(default_factory=lambda: np.array([]))
    mean: np.ndarray = field(default_factory=lambda: np.array([]))      # (S, K)
    var: np.ndarray = field(default_factory=lambda: np.array([]))       # (S, K)
    corr: np.ndarray = field(default_factory=lambda: np.array([]))      # (S, K, K)
    n_obs: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def build(cls, states: np.ndarray, expert_returns: np.ndarray) -> "RegimeStats":
        ids = [int(s) for s in sorted(set(states.tolist())) if s >= 0]
        S, K = len(ids), expert_returns.shape[1]
        pi = np.zeros(S)
        mean = np.zeros((S, K))
        var = np.zeros((S, K))
        corr = np.zeros((S, K, K))
        n_obs = np.zeros(S, dtype=int)
        for i, s in enumerate(ids):
            m = states == s
            x = expert_returns[m]
            pi[i] = m.mean()
            n_obs[i] = int(m.sum())
            mean[i] = x.mean(axis=0)
            var[i] = x.var(axis=0, ddof=1) if x.shape[0] > 2 else 0.0
            if x.shape[0] > 2:
                sd = x.std(axis=0, ddof=1)
                sd = np.where(sd > 1e-12, sd, np.nan)
                c = np.corrcoef(x, rowvar=False)
                corr[i] = np.nan_to_num(c, nan=0.0)
                corr[i][np.arange(K), np.arange(K)] = 1.0
        return cls(states=states, expert_returns=expert_returns, state_ids=ids,
                   pi=pi, mean=mean, var=var, corr=corr, n_obs=n_obs)


# --------------------------------------------------------------------------
# 命题 5：门控价值的两渠道分解
# --------------------------------------------------------------------------
def channel_decomposition(
    stats: RegimeStats,
    *,
    risk_aversion: float = 5.0,
    n_boot: int = 500,
    block: int = 12,
    seed: int = 42,
    n_eff: float | None = None,
) -> dict:
    r"""把门控价值分解为两条渠道。

    记 $b_{k,s}=\E[r_k\mid s]$（专家 $k$ 在制度 $s$ 的条件期望收益）。
    以"按条件均值配置专家权重"为最优门控的一阶近似，主动权重偏离与
    $\bmu_s$ 成比例。于是

    .. math::
        \Gamma^{(1)} = \frac{1}{K}\sum_k \sum_s \pi_s (b_{k,s}-\bar b_k)^2
        \qquad\text{（一阶：收益择时）}

        \Gamma^{(2)} = \frac{1}{K}\sum_k \sum_s \pi_s
            \big(\sigma_{k,s}-\bar\sigma_k\big)^2
        \qquad\text{（二阶：波动择时）}

    **噪声底线的校准**：把制度标签在时间上做循环区块重抽样，破坏制度与收益的
    对应关系，重新计算 $\Gamma^{(1)},\Gamma^{(2)}$，取分布的 95\% 分位数作为
    $\Theta$。此处有两个必须处理的技术问题：

    1. $K$ 个专家的收益在横截面上高度相关（它们由同一批资产构成），
       重抽样时必须**按整期联合重抽样**（本实现天然如此，因按时间下标抽取），
       否则会低估噪声；
    2. 区块长度需覆盖制度内部的序列相关，默认取 12 个月（一年）。
       若区块过短，噪声分布会被压缩，导致 $p$ 值偏小。

    另外提供 ``n_eff`` 参数：当研究者认为专家的**有效独立个数**小于 $K$ 时，
    可显式指定有效样本量以对 $\Gamma$ 的抽样方差做保守校正。
    """
    base = _gammas(stats)

    # 用"专家收益横截面相关"估计有效独立个数（平均两两相关系数法）
    if n_eff is None:
        n_eff = _effective_experts(stats)

    T = stats.expert_returns.shape[0]
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    boot1 = np.empty(n_boot)
    boot2 = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        shuffled = RegimeStats.build(stats.states, stats.expert_returns[idx])
        g1, g2 = _gammas(shuffled)
        boot1[b] = g1
        boot2[b] = g2

    # 有效独立个数小于名义专家数时，按比例放大自助分布的离散度（保守校正）
    inflate = float(np.sqrt(stats.mean.shape[1] / max(n_eff, 1.0))) if n_eff else 1.0
    boot1 = boot1.mean() + (boot1 - boot1.mean()) * inflate
    boot2 = boot2.mean() + (boot2 - boot2.mean()) * inflate

    theta1 = float(np.quantile(boot1, 0.95))
    theta2 = float(np.quantile(boot2, 0.95))
    lam = risk_aversion
    return {
        "gamma1": base[0],
        "gamma2": base[1],
        "theta1_95": theta1,
        "theta2_95": theta2,
        "gamma1_minus_theta": base[0] - theta1,
        "gamma2_minus_theta": base[1] - theta2,
        "deltaU1": base[0] / (2 * lam),
        "deltaU2": base[1] / (2 * lam),
        "theta1_mean": float(boot1.mean()),
        "theta2_mean": float(boot2.mean()),
        "p1": float((boot1 >= base[0]).mean()),
        "p2": float((boot2 >= base[1]).mean()),
        "n_obs_by_state": stats.n_obs.tolist(),
        "state_ids": stats.state_ids,
        "pi": stats.pi.tolist(),
        "block": block,
        "n_eff_experts": float(n_eff) if n_eff else None,
        "variance_inflation": inflate,
    }


def _effective_experts(stats: RegimeStats) -> float:
    r"""由专家收益的平均两两相关估计"有效独立专家数"。

    $\hat K_{\text{eff}}=K/\big[1+(K-1)\bar\rho\big]$，其中 $\bar\rho$ 为
    制度内专家收益两两相关系数的平均。$\bar\rho\to1$ 时 $K_{\text{eff}}\to1$，
    表示各专家实质上是同一个信号。
    """
    if stats.corr.size == 0:
        return float(stats.mean.shape[1])
    K = stats.corr.shape[1]
    if K < 2:
        return float(K)
    rhos = []
    for s in range(stats.corr.shape[0]):
        C = stats.corr[s]
        off = C[~np.eye(K, dtype=bool)]
        rhos.append(np.nanmean(off))
    rho_bar = float(np.clip(np.nanmean(rhos), -0.99, 0.99))
    return float(K / (1.0 + (K - 1.0) * max(rho_bar, 0.0)))


def _gammas(stats: RegimeStats) -> tuple[float, float]:
    if stats.mean.size == 0:
        return np.nan, np.nan
    pi = stats.pi[:, None]
    mean_bar = (pi * stats.mean).sum(axis=0, keepdims=True)
    g1 = float((pi * (stats.mean - mean_bar) ** 2).sum() / stats.mean.shape[1])
    sd = np.sqrt(np.maximum(stats.var, 0.0))
    sd_bar = (pi * sd).sum(axis=0, keepdims=True)
    g2 = float((pi * (sd - sd_bar) ** 2).sum() / sd.shape[1])
    return g1, g2


# --------------------------------------------------------------------------
# 命题 7：制度持续性与动态乘数
# --------------------------------------------------------------------------
def transition_matrix(states: np.ndarray, state_ids: list[int] | None = None) -> np.ndarray:
    """估计一阶马尔可夫转移矩阵。"""
    ids = state_ids or [int(s) for s in sorted(set(states.tolist())) if s >= 0]
    S = len(ids)
    idx = {s: i for i, s in enumerate(ids)}
    P = np.zeros((S, S))
    for t in range(states.size - 1):
        a, b = int(states[t]), int(states[t + 1])
        if a in idx and b in idx:
            P[idx[a], idx[b]] += 1
    row = P.sum(axis=1, keepdims=True)
    return np.divide(P, np.where(row > 0, row, 1.0))


def persistence_multiplier(P: np.ndarray, discount: float = 0.98) -> dict:
    r"""制度持续性的动态乘数。

    在制度服从一阶马尔可夫链、投资者折现因子为 $\beta$ 的设定下，
    状态依赖的最优主动权重满足
    $\wtilde w_s^\star\propto\sum_{n\ge0}(\beta P)^n\bmu^{(n)}_s/\lambda$，
    即静态解被乘上"预期未来制度路径"的加权和。

    定义**持续性乘数** $\mathcal{M}_s=\big[\sum_{n\ge0}\beta^n P^n\big]_{ss}$
    的对角元素，以及**半衰期** $h_s=\ln 0.5/\ln P_{ss}$。
    $\mathcal{M}_s$ 越大，掌握当前制度的信息价值越高，门控的潜在收益越大。
    """
    S = P.shape[0]
    M = np.linalg.inv(np.eye(S) - discount * P)
    diag = np.diag(M)
    with np.errstate(divide="ignore", invalid="ignore"):
        half_life = np.where((P.diagonal() > 0) & (P.diagonal() < 1),
                             np.log(0.5) / np.log(np.maximum(P.diagonal(), 1e-12)), np.inf)
    return {
        "P": P,
        "stationary": _stationary(P),
        "multiplier": diag,
        "mean_multiplier": float(diag.mean()),
        "half_life": half_life,
        "mean_half_life": float(np.mean(half_life[np.isfinite(half_life)])) if np.isfinite(half_life).any() else np.nan,
        "diag": P.diagonal().copy(),
    }


def _stationary(P: np.ndarray) -> np.ndarray:
    S = P.shape[0]
    A = np.vstack([P.T - np.eye(S), np.ones(S)])
    b = np.concatenate([np.zeros(S), [1.0]])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol


def persistence_gain(
    stats: RegimeStats,
    P: np.ndarray,
    *,
    discount: float = 0.98,
    risk_aversion: float = 5.0,
) -> dict:
    r"""检验 H5：制度越持续，门控的潜在收益越大。

    对每个制度 $s$ 计算其持续性乘数 $\mathcal M_s$ 与该制度下的条件均值离散度
    $g^{(1)}_s=\frac1K\sum_k(b_{k,s}-\bar b_k)^2$，检验二者在制度间的正相关。
    按命题 7，$\mathcal M_s$ 放大该制度的条件信息价值。
    """
    pi = stats.pi[:, None]
    mean_bar = (pi * stats.mean).sum(axis=0, keepdims=True)
    g1_s = ((stats.mean - mean_bar) ** 2).mean(axis=1)          # (S,)
    pers = persistence_multiplier(P, discount=discount)
    M = pers["multiplier"]
    if len(M) != g1_s.size:
        return {"error": "状态数不匹配", "n_states_stats": int(g1_s.size), "n_states_P": int(len(M))}
    rho = float(np.corrcoef(M, g1_s)[0, 1]) if g1_s.size > 2 and g1_s.std() > 0 else np.nan
    # 持续性加权的潜在收益
    weighted = float((stats.pi * M * g1_s).sum() / max(stats.pi.sum(), 1e-12))
    unweighted = float((stats.pi * g1_s).sum() / max(stats.pi.sum(), 1e-12))
    return {
        "multiplier": M.tolist(),
        "gamma1_by_state": g1_s.tolist(),
        "corr_multiplier_gamma1": rho,
        "weighted_gamma1": weighted,
        "unweighted_gamma1": unweighted,
        "amplification": weighted / unweighted if unweighted > 0 else np.nan,
        "mean_half_life": pers["mean_half_life"],
        "diag_P": pers["diag"].tolist(),
        "stationary": pers["stationary"].tolist(),
        "discount": discount,
        "deltaU_weighted": weighted / (2 * risk_aversion),
    }


# --------------------------------------------------------------------------
# 命题 8：容量约束与最优状态数
# --------------------------------------------------------------------------
def optimal_states(
    stats_by_S: dict[int, RegimeStats],
    *,
    n_gate_params: int = 200,
    risk_aversion: float = 5.0,
    turnover_cost: float = 0.0,
) -> dict:
    r"""容量约束下的最优状态数。

    命题 8 表明，门控的净价值为

    .. math::
        \Delta(S)=\frac{1}{2\lambda}\Big[\Gamma^{(1)}(S)-\Theta^{(1)}(S)\Big]
        -\frac{\sigma^2}{2\lambda}\cdot\frac{d_{\text{gate}}}{T}
        -\kappa\,\mathcal C(S),

    其中第二项为**门控参数自身的估计误差**（$d_{\text{gate}}$ 为门控参数个数，
    $T$ 为训练样本量）。$\Gamma^{(1)}$ 随 $S$ 上升但边际递减，
    $\Theta^{(1)}\propto(S-1)/T$ 线性上升，容量项随 $S$ 上升（状态越多门控
    输入维度越高），故 $\Delta(S)$ 在内点取得最大值。

    本函数以实证估计的 $\Gamma^{(1)}(S)$、$\Theta^{(1)}(S)$ 为输入，
    数值求解 $\hat S^\ast=\arg\max_S\Delta(S)$。
    """
    rows = []
    for S, st in sorted(stats_by_S.items()):
        g1, g2 = _gammas(st)
        T = int(st.n_obs.sum())
        theta1 = (S - 1) / max(T, 1) * _ic_variance(st) * _vif(st)
        capacity = _ic_variance(st) * n_gate_params / max(T, 1)
        delta = (g1 - theta1) / (2 * risk_aversion) - capacity / (2 * risk_aversion)
        rows.append({
            "S": S, "gamma1": g1, "gamma2": g2, "theta1": theta1,
            "capacity": capacity, "delta": delta, "T": T,
            "n_params": n_gate_params + 3 * st.mean.shape[1],
        })
    best = max(rows, key=lambda r: r["delta"]) if rows else None
    return {"curve": rows, "S_star": best["S"] if best else None,
            "delta_star": best["delta"] if best else None,
            "n_gate_params": n_gate_params, "risk_aversion": risk_aversion,
            "turnover_cost": turnover_cost}


def _ic_variance(stats: RegimeStats) -> float:
    r"""条件均值的抽样方差 $\sigma_b^2/n$ 中的 $\sigma_b^2$ 部分。

    用专家收益在制度内的平均方差作为 $\sigma_\varepsilon^2$ 的代理。
    """
    return float(np.mean(stats.var)) if stats.var.size else np.nan


def _vif(stats: RegimeStats) -> float:
    r"""方差膨胀因子 $1/(1-\rho_k^2)$ 的平均值。"""
    if stats.corr.size == 0:
        return 1.0
    K = stats.corr.shape[1]
    if K < 2:
        return 1.0
    vifs = []
    for s in range(stats.corr.shape[0]):
        C = stats.corr[s]
        for k in range(K):
            others = [j for j in range(K) if j != k]
            if not others:
                continue
            R = C[np.ix_(others, others)]
            r = C[np.ix_(others, [k])]
            try:
                beta = np.linalg.solve(R + 1e-8 * np.eye(len(others)), r)
                r2 = float((r.T @ beta).item())
            except np.linalg.LinAlgError:
                r2 = 0.0
            vifs.append(1.0 / max(1.0 - min(max(r2, 0.0), 0.999), 1e-6))
    return float(np.mean(vifs)) if vifs else 1.0


# --------------------------------------------------------------------------
# 双重排序：区分两条渠道
# --------------------------------------------------------------------------
def channel_identification_test(
    stats: RegimeStats,
    *,
    n_perm: int = 2000,
    seed: int = 7,
    block: int = 12,
) -> dict:
    r"""渠道识别检验：一阶 vs 二阶。

    一阶渠道要求专家收益的**条件均值**随制度变化；二阶渠道只要求条件**波动/相关结构**
    随制度变化。两者的可检验含义不同，且噪声底线不同（命题 4）。

    检验构造：把制度标签在时间上做循环区块置换（保留制度的时序聚集结构，
    消除其与收益的真实对应），分别重新计算

    .. math::
        \mathrm{CV}^{(1)}=\frac{\operatorname{Var}_s(\bar b_s)}{\E_s[\bar b_s]^2},
        \qquad
        \mathrm{CV}^{(2)}=\frac{\operatorname{Var}_s(\bar\sigma_s)}{\E_s[\bar\sigma_s]^2},

    得到原假设下的分布。其中 $\bar b_s,\bar\sigma_s$ 为制度 $s$ 内专家收益的
    截面均值与截面波动均值。$p$ 值为置换分布中不小于观测值的比例。
    """
    T = stats.expert_returns.shape[0]
    obs1, obs2 = _cv_stats(stats)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    null1 = np.empty(n_perm)
    null2 = np.empty(n_perm)
    for b in range(n_perm):
        # 区块置换：对区块顺序做随机排列，保留块内序列结构
        blocks = [np.arange(s, s + block) % T for s in range(0, T, block)]
        order = rng.permutation(len(blocks))
        idx = np.concatenate([blocks[i] for i in order])[:T]
        sh = RegimeStats.build(stats.states, stats.expert_returns[idx])
        c1, c2 = _cv_stats(sh)
        null1[b] = c1
        null2[b] = c2
    return {
        "cv1_observed": obs1,
        "cv2_observed": obs2,
        "cv1_null_mean": float(np.nanmean(null1)),
        "cv2_null_mean": float(np.nanmean(null2)),
        "cv1_null_q95": float(np.nanquantile(null1, 0.95)),
        "cv2_null_q95": float(np.nanquantile(null2, 0.95)),
        "p1": float(np.nanmean(null1 >= obs1)),
        "p2": float(np.nanmean(null2 >= obs2)),
        "n_perm": n_perm,
        "block": block,
    }


def _cv_stats(stats: RegimeStats) -> tuple[float, float]:
    """制度间的一阶与二阶变异系数。"""
    if stats.mean.size == 0:
        return np.nan, np.nan
    b_s = stats.mean.mean(axis=1)                       # (S,) 各制度的截面平均专家收益
    sd_s = np.sqrt(np.maximum(stats.var, 0.0)).mean(axis=1)
    cv1 = float(np.var(b_s, ddof=1) / (np.mean(b_s) ** 2)) if np.mean(b_s) != 0 and b_s.size > 2 else np.nan
    cv2 = float(np.var(sd_s, ddof=1) / (np.mean(sd_s) ** 2)) if np.mean(sd_s) != 0 and sd_s.size > 2 else np.nan
    return cv1, cv2
