r"""信号—状态互补性统计量 $\hat\Gamma$ 的估计与噪声底线 $\Theta$ 的自助法校准。

理论对应
--------
命题 2 给出门控占优的充要条件 $\hat\Gamma>\Theta+2\lambda\kappa C(\hat g)$。
本模块负责把式 (4.2) 的 $\hat\Gamma$ 与式 (4.3) 的 $\Theta$ 变成可计算的统计量，
并用**平稳区块自助法**给出 $\hat\Gamma$ 在"无真实互补性"原假设下的分布，
从而得到与估计噪声等价的临界值。

估计步骤
--------
1. 在每个月 $t$ 内，把持有期收益对标准化特征做横截面回归：

   $$r_{i,t+1}=a_t+\sum_j c_{j,t}\,\tilde x_{j,i,t}+\varepsilon_{i,t+1}$$

   得到月度条件信息系数向量 $c_t$。

2. 按制度 $s$ 归并：$\hat b_{j,s}=\frac{1}{|T_s|}\sum_{t\in T_s}c_{j,t}$。

3. 计算
   $$\hat\Gamma=\frac1J\sum_j\sum_s\hat\pi_s\big(\hat b_{j,s}-\hat{\bar b}_j\big)^2 .$$

4. 原假设 $H_0$：$c_t$ 的真实条件均值在各制度间相同（无互补性）。用长度为 $L$ 的
   循环区块自助法重抽样月份下标，重新归并并计算 $\hat\Gamma^{*(b)}$；
   $\Theta$ 取 $\hat\Gamma^{*}$ 的 95% 分位数（即在无互补性下仅由抽样噪声即可达到的水平）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComplementarityResult:
    gamma: float
    theta_95: float
    theta_mean: float
    ic_by_state: dict
    ic_overall: dict
    p_value: float
    n_months: int
    state_counts: dict
    feature_names: list[str] = field(default_factory=list)
    bootstrap: np.ndarray = field(default_factory=lambda: np.array([]))
    identified: bool = True
    note: str = ""

    @property
    def exceeds_threshold(self) -> bool:
        if not self.identified or not np.isfinite(self.gamma) or not np.isfinite(self.theta_95):
            return False
        return self.gamma > self.theta_95

    def summary(self) -> dict:
        return {
            "identified": bool(self.identified),
            "note": self.note,
            "gamma": None if not np.isfinite(self.gamma) else self.gamma,
            "theta_95": None if not np.isfinite(self.theta_95) else self.theta_95,
            "theta_mean": None if not np.isfinite(self.theta_mean) else self.theta_mean,
            "gamma_minus_theta": (None if not (np.isfinite(self.gamma) and np.isfinite(self.theta_95))
                                  else self.gamma - self.theta_95),
            "exceeds": bool(self.exceeds_threshold),
            "p_value": None if not np.isfinite(self.p_value) else self.p_value,
            "n_months": self.n_months,
            "state_counts": self.state_counts,
            "ic_overall": self.ic_overall,
        }


def cross_sectional_ic(
    features: np.ndarray,
    forward_returns: np.ndarray,
    universe: np.ndarray,
    *,
    min_assets: int = 20,
) -> np.ndarray:
    """逐月横截面回归，返回 (T, J) 的月度信息系数。

    参数
    ----
    features : (T, N, J) 已标准化的特征
    forward_returns : (T, N) 持有期收益
    universe : (T, N) 1=进入当月股票池
    min_assets : 单月回归所需的最少证券数。横截面过窄时月度 IC 的抽样误差过大，
        回归结果不可用。窄横截面样本（如 10 只 ETF）因此**无法识别**互补性，
        这本身就是命题 2 比较静态 H2 的直接体现。
    """
    T, N, J = features.shape
    out = np.full((T, J), np.nan)
    for t in range(T):
        mask = (universe[t] > 0) & np.isfinite(forward_returns[t])
        if mask.sum() < min_assets:
            continue
        X = features[t][mask]
        y = forward_returns[t][mask]
        keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if keep.sum() < min_assets:
            continue
        Xk, yk = X[keep], y[keep]
        # 截面去均值后回归（无截距等价于含截距）
        Xc = Xk - Xk.mean(axis=0, keepdims=True)
        yc = yk - yk.mean()
        try:
            coef, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[t] = coef
    return out


def assign_states(regime_var: np.ndarray, n_states: int, *, labels: np.ndarray | None = None) -> np.ndarray:
    """按状态变量分位切分为离散状态；NaN 归入最近的有效状态。"""
    x = np.asarray(regime_var, dtype=float)
    states = np.full(x.shape, -1, dtype=int)
    valid = np.isfinite(x)
    if labels is not None:
        states[valid] = labels[valid]
        return states
    if valid.sum() < n_states * 3:
        states[valid] = 0
        return states
    qs = np.quantile(x[valid], np.linspace(0, 1, n_states + 1)[1:-1])
    states[valid] = np.searchsorted(qs, x[valid], side="right")
    return states


def _gamma_from_ic(ic: np.ndarray, states: np.ndarray) -> float:
    r"""由月度 IC 与状态标签计算 $\hat\Gamma$。"""
    valid = np.isfinite(ic).all(axis=1) & (states >= 0)
    if valid.sum() < 10:
        return np.nan
    ic_v, st_v = ic[valid], states[valid]
    uniq = np.unique(st_v)
    if uniq.size < 2:
        return np.nan
    pi = np.array([(st_v == s).mean() for s in uniq])
    b = np.stack([ic_v[st_v == s].mean(axis=0) for s in uniq])       # (S, J)
    bbar = pi @ b                                                     # (J,)
    gamma = float((pi[:, None] * (b - bbar[None, :]) ** 2).sum() / b.shape[1])
    return gamma


def block_bootstrap_gamma(
    ic: np.ndarray,
    states: np.ndarray,
    *,
    n_boot: int = 500,
    block: int = 6,
    seed: int = 42,
) -> np.ndarray:
    r"""平稳区块自助法下的 $\hat\Gamma$ 分布（原假设：无真实互补性）。

    重抽样以"月份区块"为单位，保留 IC 序列的序列相关性；每次重抽样后重新计算
    各制度均值与 $\hat\Gamma$。该分布在无互补性时仍会因抽样噪声产生正的
    $\hat\Gamma$，其高分位数即为噪声底线 $\Theta$。
    """
    valid = np.isfinite(ic).all(axis=1)
    ic_v, st_v = ic[valid], states[valid]
    T = ic_v.shape[0]
    if T < 2 * block:
        return np.array([])
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        out[b] = _gamma_from_ic(ic_v[idx], st_v[idx])
    return out[np.isfinite(out)]


def estimate_complementarity(
    features: np.ndarray,
    forward_returns: np.ndarray,
    universe: np.ndarray,
    states: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    n_boot: int = 500,
    block: int = 6,
    seed: int = 42,
    min_assets: int = 20,
    min_months: int = 12,
) -> ComplementarityResult:
    r"""完整估计：先算月度 IC，再计算 $\hat\Gamma$ 与噪声底线 $\Theta$。

    当有效月份不足（横截面过窄导致月度 IC 无法估计）时，返回 ``identified=False``
    且各统计量为 NaN。此时**互补性在数据中不可识别**，按命题 2 比较静态 H2，
    门控在该样本上不具备可检验的基础。
    """
    ic = cross_sectional_ic(features, forward_returns, universe, min_assets=min_assets)
    valid = np.isfinite(ic).all(axis=1) & (states >= 0)
    n_valid = int(valid.sum())

    if n_valid < min_months:
        return ComplementarityResult(
            gamma=np.nan, theta_95=np.nan, theta_mean=np.nan,
            ic_by_state={}, ic_overall=[], p_value=np.nan,
            n_months=n_valid, state_counts={}, feature_names=feature_names or [],
            bootstrap=np.array([]), identified=False,
            note=f"有效月份 {n_valid} < {min_months}：横截面过窄，互补性不可识别",
        )

    ic_v, st_v = ic[valid], states[valid]
    uniq = np.unique(st_v)
    ic_by_state = {int(s): ic_v[st_v == s].mean(axis=0).tolist() for s in uniq}
    ic_overall = ic_v.mean(axis=0).tolist()

    gamma = _gamma_from_ic(ic_v, st_v)
    if not np.isfinite(gamma) or uniq.size < 2:
        return ComplementarityResult(
            gamma=np.nan, theta_95=np.nan, theta_mean=np.nan,
            ic_by_state=ic_by_state, ic_overall=ic_overall, p_value=np.nan,
            n_months=n_valid, state_counts={int(s): int((st_v == s).sum()) for s in uniq},
            feature_names=feature_names or [], bootstrap=np.array([]), identified=False,
            note=f"有效状态数 {uniq.size} < 2，无法计算状态间差异",
        )

    boot = block_bootstrap_gamma(ic_v, st_v, n_boot=n_boot, block=block, seed=seed)
    theta_95 = float(np.quantile(boot, 0.95)) if boot.size else np.nan
    theta_mean = float(boot.mean()) if boot.size else np.nan
    p_value = float((boot >= gamma).mean()) if boot.size else np.nan

    return ComplementarityResult(
        gamma=float(gamma),
        theta_95=theta_95,
        theta_mean=theta_mean,
        ic_by_state=ic_by_state,
        ic_overall=ic_overall,
        p_value=p_value,
        n_months=n_valid,
        state_counts={int(s): int((st_v == s).sum()) for s in uniq},
        feature_names=feature_names or [],
        bootstrap=boot,
    )


def state_conditional_sharpe(
    strategy_returns: np.ndarray,
    states: np.ndarray,
    *,
    periods_per_year: int = 12,
) -> dict:
    """按状态分组计算策略夏普，用于检验"门控优势的状态依赖性"。"""
    out = {}
    for s in np.unique(states[states >= 0]):
        x = strategy_returns[(states == s) & np.isfinite(strategy_returns)]
        if x.size < 6 or x.std(ddof=1) == 0:
            out[int(s)] = {"n": int(x.size), "sharpe": np.nan, "mean": float(x.mean()) if x.size else np.nan}
            continue
        out[int(s)] = {
            "n": int(x.size),
            "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(periods_per_year)),
            "mean": float(x.mean()),
        }
    return out
