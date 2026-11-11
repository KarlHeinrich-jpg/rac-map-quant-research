r"""理论与假设检验主驱动。

把 ``paper/theory.md`` 中的命题转化为可计算的检验，输出 JSON 与 CSV 供论文使用。

检验清单
--------
E2  互补性阈值检验：$\hat\Gamma$ 对噪声底线 $\Theta$（区块自助法 95% 分位），
    在 A 股宽横截面与美股 ETF 窄横截面上分别估计（对应 H2）。
E3  Oracle 门控：用真实条件期望收益构造的组合作为可实现上界，检验
    "潜在互补性 -> 可利用互补性 -> 实际实现"三段落差。
E4a 状态数比较静态：$S=2,3,4$，检验 H1（$\partial/\partial S<0$）。
E5  安慰剂检验：打乱状态标签（破坏 $b_{k,s}$ 的状态依赖性），
    按推论 1.1，门控收益应归零。
E7  目标函数对照：绝对收益目标（原稿）vs 主动收益目标（本文修正），检验命题 3。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt  # noqa: E402
import complementarity as cp  # noqa: E402
import panel_data as pdmod  # noqa: E402
from model import (  # noqa: E402
    EXPERT_FEATURES,
    EXPERT_NAMES,
    GateConfig,
    expert_confidence,
    expert_risk,
    expert_scores,
    expert_weights,
    predict_gate,
    recent_performance,
    train_gate,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_panel() -> pdmod.PanelData:
    return pdmod.build_panel_data()


def feature_tensor(panel: pdmod.PanelData) -> tuple[np.ndarray, list[str]]:
    names = [n for n in EXPERT_FEATURES if n in panel.features]
    return np.nan_to_num(np.stack([panel.features[n] for n in names], axis=-1), nan=0.0), names


def prepare(panel: pdmod.PanelData) -> dict:
    """预计算专家权重、门控输入与专家收益。"""
    _, names = feature_tensor(panel)
    scores = expert_scores({n: panel.features[n] for n in names})
    ew = expert_weights(scores, panel.in_universe)
    ret = np.nan_to_num(panel.ret_next)
    expert_ret = np.einsum("ktn,tn->tk", ew, ret)               # (T, K)
    vol = panel.features.get("vol60", np.zeros_like(panel.in_universe))
    return {
        "scores": scores,
        "expert_w": ew,
        "expert_ret": expert_ret,
        "conf": expert_confidence(scores, panel.in_universe).T,   # (T, K)
        "risk": expert_risk(ew, vol).T,                          # (T, K)
        "perf": recent_performance(expert_ret.T),                # (K, T)
        "asset_ret": panel.ret_next,
        "universe": panel.in_universe,
        "tradable": panel.tradable,
    }


# --------------------------------------------------------------------------
# E2：互补性阈值检验
# --------------------------------------------------------------------------
def e2_complementarity(
    panel: pdmod.PanelData,
    prep: dict,
    *,
    state_var: str = "mkt_ret_12m",
    n_states: int = 3,
    n_boot: int = 500,
    block: int = 6,
    seed: int = 42,
) -> dict:
    names = [n for n in EXPERT_FEATURES if n in panel.features]
    feats = np.nan_to_num(np.stack([panel.features[n] for n in names], axis=-1), nan=0.0)
    idx = panel.regime_names.index(state_var) if state_var in panel.regime_names else 0
    states = cp.assign_states(panel.regime[:, idx], n_states)
    res = cp.estimate_complementarity(
        feats, panel.ret_next, panel.in_universe, states,
        feature_names=names, n_boot=n_boot, block=block, seed=seed,
    )
    out = res.summary()
    out["state_var"] = state_var
    out["n_states"] = n_states
    out["ic_by_state"] = {str(k): v for k, v in res.ic_by_state.items()}
    return out


# --------------------------------------------------------------------------
# E3：Oracle 门控上界
# --------------------------------------------------------------------------
def oracle_gate_weights(
    panel: pdmod.PanelData,
    prep: dict,
    window: dict,
    *,
    eta: float = 50.0,
) -> np.ndarray:
    r"""Oracle 门控：按状态内**真实**条件期望收益给专家打分。

    $g^{\text{oracle}}_{k,s}\propto\exp\!\big(\eta\,\bar a_k^\top\hat\mu_s\big)$，
    其中 $\hat\mu_s$ 为该状态内资产的平均持有期收益（仅用于构造上界基准，
    使用了测试期信息，因此不是可实现策略）。
    """
    te_lo, te_hi = window["test"]
    tr_lo, tr_hi = window["train"]
    expert_w = prep["expert_w"]
    ret = np.nan_to_num(prep["asset_ret"])
    uni = prep["universe"]
    idx = panel.regime_names.index("mkt_ret_12m") if "mkt_ret_12m" in panel.regime_names else 0
    states = cp.assign_states(panel.regime[:, idx], 3)

    # 用训练窗估计各状态的资产均值（严格样本外：只用训练窗）
    mu_s = {}
    for s in np.unique(states[tr_lo:tr_hi]):
        if s < 0:
            continue
        mask = (states[tr_lo:tr_hi] == s)
        sub = ret[tr_lo:tr_hi][mask]
        mu_s[int(s)] = np.nanmean(sub, axis=0)
    fallback = np.nanmean(ret[tr_lo:tr_hi], axis=0)
    mu_all = np.stack([mu_s.get(s, fallback) for s in range(3)])        # (S, N)

    w = np.zeros_like(uni)
    for t in range(te_lo, te_hi):
        s = int(states[t])
        if s < 0:
            s = 0
        # expert_w[:, t, :] 形状为 (K, N)，mu_all[s] 为 (N,)
        scores_k = expert_w[:, t, :] @ mu_all[s]                          # (K,)
        g = np.exp(np.clip(eta * scores_k, -30, 30))
        g = g / max(g.sum(), 1e-12)
        w[t] = np.einsum("k,kn->n", g, expert_w[:, t, :])
    return w


def e3_oracle(panel: pdmod.PanelData, prep: dict, windows: list[dict], *, cost_bps: float = 10.0) -> pd.DataFrame:
    rows = []
    for w in windows:
        te_lo, te_hi = w["test"]
        for name, weights in (
            ("equal_weight", None),
            ("static_experts", prep["expert_w"].mean(axis=0)),
            ("oracle_gate", oracle_gate_weights(panel, prep, w)),
        ):
            if weights is None:
                ww = np.where(prep["universe"][te_lo:te_hi] > 0, 1.0, 0.0)
                ww = ww / np.maximum(ww.sum(axis=1, keepdims=True), 1e-12)
            else:
                ww = weights[te_lo:te_hi]
            res = bt.run_backtest(ww, prep["asset_ret"][te_lo:te_hi], prep["tradable"][te_lo:te_hi],
                                  prep["universe"][te_lo:te_hi], cost_bps=cost_bps)
            m = bt.performance(res.net, res.benchmark, res.turnover)
            rows.append({"model": name, "test_year": w["test_year"], "cost_bps": cost_bps, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E4a：状态数比较静态
# --------------------------------------------------------------------------
def e4a_state_count(
    panel: pdmod.PanelData,
    prep: dict,
    windows: list[dict],
    *,
    state_counts: tuple[int, ...] = (2, 3, 4),
    cost_bps: float = 10.0,
    seeds: tuple[int, ...] = (7, 19),
    epochs: int = 150,
) -> pd.DataFrame:
    rows = []
    for n_states in state_counts:
        for w in windows:
            tr_lo, tr_hi = w["train"]
            te_lo, te_hi = w["test"]
            idx = panel.regime_names.index("mkt_ret_12m") if "mkt_ret_12m" in panel.regime_names else 0
            states = cp.assign_states(panel.regime[:, idx], n_states)
            # 把状态离散标签作为额外门控输入（one-hot）
            onehot = np.eye(n_states)[np.clip(states, 0, n_states - 1)]
            reg_tr = np.concatenate([panel.regime[tr_lo:tr_hi], onehot[tr_lo:tr_hi]], axis=1)
            reg_all = np.concatenate([panel.regime, onehot], axis=1)
            for seed in seeds:
                cfg = GateConfig(seed=seed, epochs=epochs, objective="active")
                model, _ = train_gate(reg_tr, prep["conf"][tr_lo:tr_hi], prep["risk"][tr_lo:tr_hi],
                                      prep["perf"][:, tr_lo:tr_hi].T, prep["expert_w"][:, tr_lo:tr_hi],
                                      prep["asset_ret"][tr_lo:tr_hi], cfg)
                g = predict_gate(model, reg_all, prep["conf"], prep["risk"], prep["perf"].T)
                ww = np.einsum("tk,ktn->tn", g, prep["expert_w"])[te_lo:te_hi]
                res = bt.run_backtest(ww, prep["asset_ret"][te_lo:te_hi], prep["tradable"][te_lo:te_hi],
                                      prep["universe"][te_lo:te_hi], cost_bps=cost_bps)
                m = bt.performance(res.net, res.benchmark, res.turnover)
                rows.append({"model": f"gate_S{n_states}", "n_states": n_states, "seed": seed,
                             "test_year": w["test_year"], "cost_bps": cost_bps, **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# E5：安慰剂检验（打乱状态标签）
# --------------------------------------------------------------------------
def e5_placebo(
    panel: pdmod.PanelData,
    prep: dict,
    windows: list[dict],
    *,
    n_perm: int = 200,
    cost_bps: float = 10.0,
) -> dict:
    r"""安慰剂：按推论 1.1，破坏状态与信号有效性的联系后门控收益应归零。

    实现方式：保持门控输入与专家权重不变，仅把**资产持有期收益矩阵**在时间维度上
    做循环区块置换，从而消除状态—收益的真实关联，再重新训练门控并观察其主动收益。
    """
    rng = np.random.default_rng(12345)
    T = prep["asset_ret"].shape[0]
    stats = []
    for _ in range(n_perm):
        shift = int(rng.integers(1, T))
        perm = np.roll(np.arange(T), shift)
        ret_perm = prep["asset_ret"][perm]
        # 用置换后的收益重算专家收益与近期表现，保持其余输入不变
        r = np.nan_to_num(ret_perm)
        expert_ret = np.einsum("ktn,tn->tk", prep["expert_w"], r)
        perf = recent_performance(expert_ret.T)
        w = windows[-1]
        tr_lo, tr_hi = w["train"]
        te_lo, te_hi = w["test"]
        cfg = GateConfig(seed=7, epochs=80, objective="active")
        model, _ = train_gate(panel.regime[tr_lo:tr_hi], prep["conf"][tr_lo:tr_hi], prep["risk"][tr_lo:tr_hi],
                              perf[:, tr_lo:tr_hi].T, prep["expert_w"][:, tr_lo:tr_hi], ret_perm[tr_lo:tr_hi], cfg)
        g = predict_gate(model, panel.regime[te_lo:te_hi], prep["conf"][te_lo:te_hi],
                         prep["risk"][te_lo:te_hi], perf[:, te_lo:te_hi].T)
        ww = np.einsum("tk,ktn->tn", g, prep["expert_w"][:, te_lo:te_hi])
        res = bt.run_backtest(ww, ret_perm[te_lo:te_hi], prep["tradable"][te_lo:te_hi],
                              prep["universe"][te_lo:te_hi], cost_bps=cost_bps)
        stats.append(float(np.nanmean(res.active)))
    arr = np.array(stats)
    return {
        "n_perm": int(arr.size),
        "mean_active_placebo": float(np.nanmean(arr)),
        "std_active_placebo": float(np.nanstd(arr)),
        "q05": float(np.nanquantile(arr, 0.05)),
        "q95": float(np.nanquantile(arr, 0.95)),
        "fraction_positive": float(np.nanmean(arr > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test", type=int, default=2016)
    parser.add_argument("--last-test", type=int, default=2025)
    parser.add_argument("--state-var", default="mkt_ret_12m")
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--out", default="hypothesis_tests")
    parser.add_argument("--skip-e5", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    prep = prepare(panel)
    windows = make_windows(panel, args.first_test, args.last_test)
    print(f"panel T={panel.n_periods} N={panel.n_assets}; windows={len(windows)}")

    report: dict = {"panel": panel.meta, "n_windows": len(windows)}

    print("\n[E2] 互补性阈值检验 ...", flush=True)
    report["E2_complementarity"] = e2_complementarity(
        panel, prep, state_var=args.state_var, n_states=args.n_states, n_boot=args.n_boot
    )
    e2 = report["E2_complementarity"]
    print(f"  gamma={e2['gamma']:.6f}  theta95={e2['theta_95']:.6f}  "
          f"exceeds={e2['exceeds']}  p={e2['p_value']:.3f}  months={e2['n_months']}")

    print("\n[E3] Oracle 门控上界 ...", flush=True)
    e3 = e3_oracle(panel, prep, windows, cost_bps=args.cost_bps)
    e3.to_csv(out_dir / "e3_oracle.csv", index=False)
    report["E3_oracle"] = e3.groupby("model")[["cagr", "sharpe", "avg_turnover"]].mean().round(4).to_dict()
    print(e3.groupby("model")[["sharpe", "cagr", "avg_turnover"]].mean().round(3).to_string())

    print("\n[E4a] 状态数比较静态 ...", flush=True)
    e4 = e4a_state_count(panel, prep, windows, cost_bps=args.cost_bps)
    e4.to_csv(out_dir / "e4a_state_count.csv", index=False)
    report["E4a_state_count"] = (
        e4.groupby(["n_states", "test_year"])[["sharpe", "mean_active"]].mean()
        .groupby("n_states").mean().round(4).to_dict()
    )
    print(e4.groupby("n_states")[["sharpe", "mean_active"]].mean().round(4).to_string())

    if not args.skip_e5:
        print("\n[E5] 安慰剂检验（打乱收益-状态关联） ...", flush=True)
        report["E5_placebo"] = e5_placebo(panel, prep, windows, cost_bps=args.cost_bps)
        print(f"  placebo mean active={report['E5_placebo']['mean_active_placebo']:.5f} "
              f"[{report['E5_placebo']['q05']:.5f}, {report['E5_placebo']['q95']:.5f}]")

    (out_dir / "hypothesis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n结果已写入 {out_dir}")


def make_windows(panel: pdmod.PanelData, first_test: int, last_test: int,
                 train_years: int = 5, valid_years: int = 1) -> list[dict]:
    years = pd.DatetimeIndex(panel.dates).year.to_numpy()
    windows = []
    for ty in range(first_test, last_test + 1):
        if ty - valid_years - train_years < years.min():
            continue
        f = lambda y, side: int(np.searchsorted(years, y, side=side))  # noqa: E731
        tr_lo, tr_hi = f(ty - valid_years - train_years, "left"), f(ty - valid_years - 1, "right")
        va_lo, va_hi = f(ty - 1, "left"), f(ty - 1, "right")
        te_lo, te_hi = f(ty, "left"), f(ty, "right")
        if tr_hi - tr_lo < 12 or te_hi - te_lo < 1:
            continue
        windows.append({"test_year": ty, "train": (tr_lo, tr_hi), "valid": (va_lo, va_hi),
                        "test": (te_lo, te_hi), "train_years": train_years})
    return windows


if __name__ == "__main__":
    main()
