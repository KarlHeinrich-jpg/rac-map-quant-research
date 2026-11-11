r"""回测引擎与绩效度量。

执行规则
--------
- 每个形成期 $t$ 得到目标权重 $w_t$（对股票池内证券，权重和为 1）；
- $t+1$ 月首个交易日按收盘价调仓，持有至 $t+2$ 月首个交易日收盘；
- 换手按**漂移后权重**与目标权重之差计算（先让旧权重随期间收益漂移，再与目标比较）；
- 交易成本 $=\kappa\times\frac12\|w_t^{\text{drift}}-w_t^{\text{target}}\|_1$，从当期收益中扣除；
- 执行月首三个交易日涨停的证券不可买入，其权重按剩余可交易证券的比例再分配。

绩效指标
--------
CAGR、年化波动、Sharpe、Sortino、最大回撤、日频/月频 CVaR、平均换手、主动度。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PERIODS_PER_YEAR = 12


@dataclass
class BacktestResult:
    dates: np.ndarray
    gross: np.ndarray
    net: np.ndarray
    turnover: np.ndarray
    weights: np.ndarray
    benchmark: np.ndarray
    active: np.ndarray = field(default_factory=lambda: np.array([]))
    diagnostics: dict = field(default_factory=dict)


def _reallocate_unbuyable(target: np.ndarray, tradable: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """把不可买入证券的权重按比例分配给可买入证券。"""
    w = np.where(universe > 0, target, 0.0)
    total = w.sum()
    if total <= 0:
        return w
    w = w / total
    if tradable.min() >= 1.0:
        return w
    ok = (tradable > 0) & (w > 0)
    if ok.sum() == 0:
        return w
    blocked_mass = w[~ok].sum()
    w[~ok] = 0.0
    w[ok] += blocked_mass * w[ok] / w[ok].sum()
    return w


def run_backtest(
    weights: np.ndarray,
    ret_next: np.ndarray,
    tradable: np.ndarray,
    universe: np.ndarray,
    *,
    cost_bps: float = 10.0,
) -> BacktestResult:
    """执行回测。

    参数
    ----
    weights : (T, N) 每个形成期的目标权重（未归一化亦可）
    ret_next : (T, N) 从执行日到退出日的持有期收益
    tradable : (T, N) 1 = 执行月首三日未涨停、可买入
    universe : (T, N) 1 = 当期进入股票池
    cost_bps : 单边交易成本（基点）
    """
    T, N = weights.shape
    kappa = cost_bps / 1e4

    applied = np.zeros((T, N))
    for t in range(T):
        applied[t] = _reallocate_unbuyable(weights[t], tradable[t], universe[t])

    gross = np.full(T, np.nan)
    net = np.full(T, np.nan)
    turnover = np.zeros(T)
    drift_prev = None

    for t in range(T):
        r = ret_next[t]
        valid = np.isfinite(r)
        if not valid.any() or applied[t].sum() <= 0:
            continue
        w = applied[t].copy()
        # 漂移：上一期权重按上一期实际收益增长
        if drift_prev is not None:
            base = drift_prev * (1.0 + np.where(valid, r, 0.0))
            base = np.where(base > 0, base, 0.0)
            s = base.sum()
            base = base / s if s > 0 else applied[t]
        else:
            base = np.zeros(N)
        to = 0.5 * np.abs(base - w).sum() if t > 0 else 0.0
        turnover[t] = to
        g = float(np.nansum(w * np.where(valid, r, 0.0)))
        gross[t] = g
        net[t] = g - kappa * to
        drift_prev = w

    ew = np.where(universe > 0, 1.0, 0.0)
    ew = ew / np.maximum(ew.sum(axis=1, keepdims=True), 1e-12)
    bench = np.array([float(np.nansum(ew[t] * np.where(np.isfinite(ret_next[t]), ret_next[t], 0.0))) for t in range(T)])

    mask = np.isfinite(net)
    active = net - bench
    active[~mask] = np.nan
    return BacktestResult(
        dates=np.arange(T),
        gross=gross,
        net=net,
        turnover=turnover,
        weights=applied,
        benchmark=bench,
        active=active,
        diagnostics={
            "mean_active_exposure": float(np.nanmean(0.5 * np.abs(applied - ew).sum(axis=1))),
            "mean_n_holdings": float(np.nanmean((applied > 1e-6).sum(axis=1))),
        },
    )


def performance(net: np.ndarray, benchmark: np.ndarray | None = None, turnover: np.ndarray | None = None) -> dict:
    x = np.asarray(net, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {k: np.nan for k in
                ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "cvar_5", "skew", "mean_active", "sharpe_active", "avg_turnover"]}
    cum = np.cumprod(1.0 + x)
    years = x.size / PERIODS_PER_YEAR
    cagr = float(cum[-1] ** (1.0 / years) - 1.0) if cum[-1] > 0 else -1.0
    vol = float(np.std(x, ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    sharpe = float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if np.std(x, ddof=1) > 0 else np.nan
    downside = x[x < 0]
    sortino = (
        float(np.mean(x) / np.std(downside, ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        if downside.size > 2 and np.std(downside, ddof=1) > 0
        else np.nan
    )
    peak = np.maximum.accumulate(cum)
    mdd = float((cum / peak - 1.0).min())
    cvar5 = float(np.mean(np.sort(x)[: max(1, int(0.05 * x.size))]))

    out = {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "cvar_5": cvar5,
        "skew": float(np.mean(((x - x.mean()) / (x.std(ddof=1) + 1e-12)) ** 3)),
        "avg_turnover": float(np.mean(turnover)) if turnover is not None else np.nan,
    }
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=float)
        b = b[np.isfinite(b)]
        if b.size == x.size:
            a = x - b
            out["mean_active"] = float(np.mean(a) * PERIODS_PER_YEAR)
            sd = np.std(a, ddof=1)
            out["sharpe_active"] = float(np.mean(a) / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else np.nan
            out["tracking_error"] = float(sd * np.sqrt(PERIODS_PER_YEAR))
    return out


def sharpe_from_series(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3 or np.std(x, ddof=1) == 0:
        return np.nan
    return float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(PERIODS_PER_YEAR))
