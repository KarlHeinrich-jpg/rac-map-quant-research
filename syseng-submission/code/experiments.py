r"""主实验：滚动样本外门控组合、基线比较、互补性阈值检验。

实验编号与理论命题的对应见 ``paper/theory.md`` 第 6 节。

用法::

    python src/experiments.py --sample ashare --first-test 2016 --last-test 2025
    python src/experiments.py --sample intl --first-test 2016 --last-test 2025
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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

SEEDS = (7, 19, 42, 73, 101)


# --------------------------------------------------------------------------
# 基线组合
# --------------------------------------------------------------------------
def weight_equal(universe: np.ndarray) -> np.ndarray:
    w = np.where(universe > 0, 1.0, 0.0)
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def weight_risk_parity(universe: np.ndarray, features: dict) -> np.ndarray:
    """风险平价：权重与波动率成反比。

    注意：``vol60`` 为年化波动率的水平值（约 0.3），若直接取倒数并按行 softmax
    会因量纲过大而退化为极端集中组合，故先做横截面标准化再取倒数。
    """
    vol = features.get("vol60")
    if vol is None:
        return weight_equal(universe)
    v = np.where(universe > 0, vol, np.nan)
    mu = np.nanmean(v, axis=1, keepdims=True)
    sd = np.nanstd(v, axis=1, keepdims=True)
    z = (v - mu) / np.where(sd > 1e-12, sd, np.nan)
    z = np.clip(np.nan_to_num(z, nan=0.0), -3.0, 3.0)
    inv = np.where(universe > 0, np.exp(-z), 0.0)   # 低波动 -> 高权重
    return inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)


def weight_expert(scores: np.ndarray, universe: np.ndarray, k: int, temperature: float = 0.6) -> np.ndarray:
    masked = np.where(universe > 0, scores[k], np.nan)
    filled = np.where(np.isfinite(masked), masked, -1e6)
    w = pdmod_softmax(filled, temperature)
    return w


def pdmod_softmax(z: np.ndarray, temperature: float) -> np.ndarray:
    zz = z / max(temperature, 1e-6)
    zmax = np.max(zz, axis=1, keepdims=True)
    e = np.exp(np.clip(zz - zmax, -60, 0))
    s = e.sum(axis=1, keepdims=True)
    return np.divide(e, s, out=np.zeros_like(e), where=s > 1e-12)


def weight_ml(
    panel: pdmod.PanelData,
    tr_lo: int,
    tr_hi: int,
    *,
    method: str = "ridge",
    temperature: float = 0.6,
) -> np.ndarray:
    """监督学习基线：在训练窗内拟合，对**全样本**逐期预测并转成组合权重。

    返回长度与面板一致的权重矩阵，便于与其它基线共用同一回测切片逻辑。
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    feats, _ = feature_tensor(panel)
    ret, uni = panel.ret_next, panel.in_universe
    xt, yt = _stack(feats[tr_lo:tr_hi], ret[tr_lo:tr_hi], uni[tr_lo:tr_hi])
    if xt.shape[0] < 50:
        return weight_equal(uni)
    model = Ridge(alpha=1.0) if method == "ridge" else GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0
    )
    model.fit(xt, yt)
    scores = np.full(uni.shape, np.nan)
    for t in range(uni.shape[0]):
        m = uni[t] > 0
        if m.sum() == 0:
            continue
        X = feats[t][m]
        ok = np.isfinite(X).all(axis=1)
        pred = np.full(int(m.sum()), np.nan)
        if ok.sum():
            pred[ok] = model.predict(X[ok])
        scores[t][m] = pred
    filled = np.where(np.isfinite(scores), scores, -1e6)
    return pdmod_softmax(filled, temperature)


def _stack(features: np.ndarray, ret: np.ndarray, universe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    T = features.shape[0]
    for t in range(T):
        m = (universe[t] > 0) & np.isfinite(ret[t])
        if m.sum() < 10:
            continue
        X = features[t][m]
        ok = np.isfinite(X).all(axis=1)
        if ok.sum() < 10:
            continue
        xs.append(X[ok])
        ys.append(ret[t][m][ok])
    if not xs:
        return np.empty((0, features.shape[-1])), np.empty(0)
    return np.vstack(xs), np.concatenate(ys)


def feature_tensor(panel: pdmod.PanelData) -> tuple[np.ndarray, list[str]]:
    names = [n for n in EXPERT_FEATURES if n in panel.features]
    arr = np.stack([panel.features[n] for n in names], axis=-1)
    return np.nan_to_num(arr, nan=0.0), names


# --------------------------------------------------------------------------
# 滚动窗
# --------------------------------------------------------------------------
def make_windows(
    panel: pdmod.PanelData, first_test: int, last_test: int, train_years: int = 5, valid_years: int = 1
) -> list[dict]:
    windows = []
    years = pd.DatetimeIndex(panel.dates).year.to_numpy()
    for ty in range(first_test, last_test + 1):
        tr_lo_year = ty - valid_years - train_years
        tr_hi_year = ty - valid_years - 1
        va_year = ty - 1
        if tr_lo_year < years.min():
            continue
        idx = lambda y, side: (  # noqa: E731
            int(np.searchsorted(years, y, side=side))
        )
        tr_lo, tr_hi = idx(tr_lo_year, "left"), idx(tr_hi_year, "right")
        va_lo, va_hi = idx(va_year, "left"), idx(va_year, "right")
        te_lo, te_hi = idx(ty, "left"), idx(ty, "right")
        if tr_hi - tr_lo < 12 or te_hi - te_lo < 1:
            continue
        windows.append(
            {
                "test_year": ty,
                "train": (tr_lo, tr_hi),
                "valid": (va_lo, va_hi),
                "test": (te_lo, te_hi),
                "train_years": train_years,
            }
        )
    return windows


# --------------------------------------------------------------------------
# 单窗口评估
# --------------------------------------------------------------------------
def evaluate_window(
    panel: pdmod.PanelData,
    scores: np.ndarray,
    conf: np.ndarray,
    risk: np.ndarray,
    perf: np.ndarray,
    window: dict,
    *,
    seeds: tuple[int, ...] = SEEDS,
    cost_bps: float = 10.0,
    train_objective: str = "active",
    candidate_grid: tuple[tuple[float, float], ...] = ((0.1, 0.05), (0.3, 0.2), (0.6, 0.4)),
    entropy_penalty: float = 0.01,
    epochs: int = 200,
) -> list[dict]:
    tr_lo, tr_hi = window["train"]
    va_lo, va_hi = window["valid"]
    te_lo, te_hi = window["test"]
    uni, ret = panel.in_universe, panel.ret_next
    expert_w = expert_weights(scores, uni)
    static_w = expert_w.mean(axis=0)
    rows: list[dict] = []

    def run(name: str, w: np.ndarray, seed: int = -1, extra: dict | None = None) -> None:
        if w.shape[0] != panel.n_periods:
            raise ValueError(f"{name}: 权重矩阵期数 {w.shape[0]} 与面板 {panel.n_periods} 不一致")
        res = bt.run_backtest(w[te_lo:te_hi], ret[te_lo:te_hi], panel.tradable[te_lo:te_hi], uni[te_lo:te_hi], cost_bps=cost_bps)
        bench = res.benchmark
        metrics = bt.performance(res.net, bench, res.turnover)
        metrics.update(res.diagnostics)
        row = {
            "model": name,
            "seed": seed,
            "test_year": window["test_year"],
            "n_periods": int(te_hi - te_lo),
            "cost_bps": cost_bps,
            "train_years": window["train_years"],
            **metrics,
        }
        if extra:
            row.update(extra)
        rows.append(row)

    # ---- 基线 ----
    run("equal_weight", weight_equal(uni))
    run("risk_parity", weight_risk_parity(uni, panel.features))
    run("static_experts", static_w)
    for k, nm in enumerate(EXPERT_NAMES):
        run(f"expert_{nm}", expert_w[k])
    run("ridge", weight_ml(panel, tr_lo, tr_hi, method="ridge"))
    run("gbrt", weight_ml(panel, tr_lo, tr_hi, method="gbrt"))

    # ---- 门控：验证集选择超参数，测试集只用于报告 ----
    for seed in seeds:
        best = None
        for risk_pen, turn_pen in candidate_grid:
            cfg = GateConfig(seed=seed, risk_penalty=risk_pen, turnover_penalty=turn_pen,
                             entropy_penalty=entropy_penalty, objective=train_objective, epochs=epochs)
            model, diag = train_gate(
                panel.regime[tr_lo:tr_hi], conf[tr_lo:tr_hi], risk[tr_lo:tr_hi], perf[:, tr_lo:tr_hi].T,
                expert_w[:, tr_lo:tr_hi], ret[tr_lo:tr_hi], cfg,
            )
            g_va = predict_gate(model, panel.regime[va_lo:va_hi], conf[va_lo:va_hi], risk[va_lo:va_hi],
                                perf[:, va_lo:va_hi].T)
            w_va = np.einsum("tk,ktn->tn", g_va, expert_w[:, va_lo:va_hi])
            res_va = bt.run_backtest(w_va, ret[va_lo:va_hi], panel.tradable[va_lo:va_hi], uni[va_lo:va_hi], cost_bps=cost_bps)
            score = bt.sharpe_from_series(res_va.net)
            if best is None or (np.isfinite(score) and score > best[0]):
                best = (score, risk_pen, turn_pen, model, diag)
        _, risk_pen, turn_pen, model, diag = best
        # 对全样本逐期给出门控权重，再由 run() 切到测试窗（训练窗之外的期不参与评估）
        g_all = predict_gate(model, panel.regime, conf, risk, perf.T)
        w_all = np.einsum("tk,ktn->tn", g_all, expert_w)
        run(f"gate_{train_objective}", w_all, seed=seed,
            extra={"risk_penalty": risk_pen, "turnover_penalty": turn_pen,
                   "valid_sharpe": best[0], "gate_dispersion": diag["gate_dispersion"],
                   "objective": train_objective,
                   "gate_mean": json.dumps(diag["gate_mean"])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="ashare", choices=["ashare", "intl"])
    parser.add_argument("--first-test", type=int, default=2016)
    parser.add_argument("--last-test", type=int, default=2025)
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--objective", default="active", choices=["active", "absolute"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="逗号分隔，默认 7,19,42,73,101")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else SEEDS

    if args.sample == "ashare":
        panel = pdmod.build_panel_data()
    else:
        raise SystemExit("intl 样本请使用 src/experiments_intl.py")

    print(f"panel: T={panel.n_periods} N={panel.n_assets} mean universe={panel.meta['mean_universe']:.1f}")

    # 专家打分与门控输入
    x_all, feat_names = feature_tensor(panel)
    scores = expert_scores({n: panel.features[n] for n in feat_names})
    ew_all = expert_weights(scores, panel.in_universe)
    conf = expert_confidence(scores, panel.in_universe).T                    # (T, K)
    vol_feat = panel.features.get("vol60", np.zeros_like(panel.in_universe))
    risk = expert_risk(ew_all, vol_feat).T                                   # (T, K)
    expert_port_ret = np.einsum("ktn,tn->tk", ew_all, np.nan_to_num(panel.ret_next))   # (T, K)
    perf = recent_performance(expert_port_ret.T)                             # (K, T)
    print(f"gate inputs: conf{conf.shape} risk{risk.shape} perf(transposed){perf.T.shape}")

    windows = make_windows(panel, args.first_test, args.last_test, train_years=args.train_years)
    if args.max_windows:
        windows = windows[: args.max_windows]
    print(f"rolling windows: {len(windows)} ({[w['test_year'] for w in windows]})")

    all_rows: list[dict] = []
    for w in windows:
        rows = evaluate_window(panel, scores, conf, risk, perf, w, cost_bps=args.cost_bps,
                               train_objective=args.objective, seeds=seeds, epochs=args.epochs)
        all_rows.extend(rows)
        print(f"  {w['test_year']}: " + ", ".join(
            f"{r['model']}={r['sharpe']:.2f}" for r in rows if r["model"] in {"equal_weight", "static_experts", f"gate_{args.objective}"} and r["seed"] in (-1, seeds[0])
        ), flush=True)

    frame = pd.DataFrame(all_rows)
    out_dir = Path(args.out) if args.out else RESULTS / f"exp_{args.sample}_{args.objective}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "window_metrics.csv", index=False)

    # 年度内先对 seed 取均值，再跨年汇总
    metric_cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "cvar_5", "avg_turnover",
                   "mean_active", "sharpe_active", "mean_active_exposure"]
    model_year = frame.groupby(["model", "test_year"])[metric_cols].mean().reset_index()
    summary = model_year.groupby("model")[metric_cols].mean().sort_values("sharpe", ascending=False)
    summary.to_csv(out_dir / "summary.csv")

    manifest = {
        "sample": args.sample,
        "first_test": args.first_test,
        "last_test": args.last_test,
        "train_years": args.train_years,
        "cost_bps": args.cost_bps,
        "objective": args.objective,
        "seeds": list(SEEDS),
        "panel": panel.meta,
        "n_windows": len(windows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 汇总（年度内先平均 seed） ===")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
