r"""国际对照实验：窄横截面样本（10 只美国 ETF / 45 只全球多资产）。

用途
----
命题 2 的比较静态 H2 预测：门控价值随横截面宽度 $N$ 上升。本模块提供窄横截面
样本（$N=10$ 与 $N=45$，月度频率），与 A 股全市场（$N\approx 350$）形成对照。
同时，原稿的零结果正是在该 ETF 样本上得到的，因此这里也是"复制原结果并给出
理论解释"的场所。

数据
----
``data/intl_daily.parquet``：日频复权价（Yahoo chart API，adjclose）

面板构造（与 A 股口径一致）
--------------------------
- 月末形成权重，下一月首个交易日建仓，持有至再下一月首个交易日；
- 特征：1/3/6/12 月动量、20/60 日波动、60 日相对均线偏离、120 日 beta；
- 制度变量：组合等权指数的 3/12 月收益、6 月波动、1 年回撤、横截面离散度。
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
from model import (  # noqa: E402
    GateConfig,
    expert_confidence,
    expert_risk,
    expert_scores,
    expert_weights,
    predict_gate,
    recent_performance,
    softmax_rows,
    train_gate,
)

DATA = ROOT / "data"
RESULTS = ROOT / "results"

ETF10 = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VNQ", "LQD", "DBC"]
GLOBAL = [
    "MDY", "EWJ", "EWG", "EWU", "EWZ", "EWW", "FXI", "INDA", "IEF", "SHY",
    "TIP", "HYG", "EMB", "BND", "SLV", "USO", "RWX", "IYR", "XLE", "XLF",
    "XLK", "XLV", "XLP", "XLY", "XLI", "XLB", "XLU", "XLC", "SMH", "IBB",
]
SEEDS = (7, 19, 42, 73, 101)


# --------------------------------------------------------------------------
# 面板
# --------------------------------------------------------------------------
def build_intl_panel(tickers: list[str], start: str = "2007-01-01") -> dict:
    daily = pd.read_parquet(DATA / "intl_daily.parquet")
    daily = daily[daily["ticker"].isin(tickers)].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["ticker", "date"]).reset_index(drop=True)

    wide = daily.pivot(index="date", columns="ticker", values="close").sort_index()
    wide = wide.dropna(how="all")
    # 上市较晚的品种用其后第一个有效值之前置空，避免前视填充
    valid_from = wide.notna().idxmax()
    for col in wide.columns:
        wide.loc[wide.index < valid_from[col], col] = np.nan
    wide = wide.ffill()

    ret_d = wide.pct_change()
    # 月末快照
    ym = wide.index.to_period("M")
    first_idx, last_idx = {}, {}
    for i, p in enumerate(ym):
        first_idx.setdefault(p, i)
        last_idx[p] = i
    months = sorted(last_idx)
    rows = []
    for p in months:
        li = last_idx[p]
        rows.append({"ym": str(p), "last_i": li, "first_i": first_idx[p]})
    meta = pd.DataFrame(rows)
    # 次月首日索引（执行价）与其后一月首日（退出价）
    meta["exec_i"] = meta["first_i"].shift(-1)
    meta["exit_i"] = meta["first_i"].shift(-2)

    T, N = len(meta), wide.shape[1]
    px = wide.to_numpy(dtype=float)
    ret_next = np.full((T, N), np.nan)
    for t in range(T):
        ei, xi = meta.loc[t, "exec_i"], meta.loc[t, "exit_i"]
        if pd.notna(ei) and pd.notna(xi) and int(xi) < len(px):
            ret_next[t] = px[int(xi)] / px[int(ei)] - 1.0

    # 特征（月末可得）
    feats: dict[str, np.ndarray] = {}
    for name, window in {"mom20": 20, "mom60": 60, "mom120": 120, "mom250": 250}.items():
        m = wide.pct_change(window)
        feats[name] = m.to_numpy(dtype=float)[[int(i) for i in meta["last_i"]]]
    vol60 = ret_d.rolling(60, min_periods=40).std() * np.sqrt(252)
    vol20 = ret_d.rolling(20, min_periods=15).std() * np.sqrt(252)
    feats["vol60"] = vol60.to_numpy(dtype=float)[[int(i) for i in meta["last_i"]]]
    feats["vol20"] = vol20.to_numpy(dtype=float)[[int(i) for i in meta["last_i"]]]
    feats["reversal_1m"] = -feats["mom20"]
    feats["turnover20"] = np.zeros((T, N))          # ETF 无换手数据，置零
    mkt_ret = ret_d.mean(axis=1)
    cov = ret_d.mul(mkt_ret, axis=0).rolling(120, min_periods=80).mean() - ret_d.rolling(120, min_periods=80).mean().mul(
        mkt_ret.rolling(120, min_periods=80).mean(), axis=0
    )
    var = (mkt_ret**2).rolling(120, min_periods=80).mean() - mkt_ret.rolling(120, min_periods=80).mean() ** 2
    beta = cov.div(var, axis=0)
    feats["beta120"] = beta.to_numpy(dtype=float)[[int(i) for i in meta["last_i"]]]

    # 横截面标准化
    for k in list(feats):
        arr = feats[k]
        mu = np.nanmean(arr, axis=1, keepdims=True)
        sd = np.nanstd(arr, axis=1, keepdims=True)
        feats[k] = np.nan_to_num((arr - mu) / np.where(sd > 1e-12, sd, np.nan), nan=0.0)

    # 制度变量
    cum = (1.0 + mkt_ret.fillna(0.0)).cumprod()
    regime_df = pd.DataFrame(
        {
            "mkt_ret_3m": cum.pct_change(63),
            "mkt_ret_12m": cum.pct_change(252),
            "mkt_vol_6m": mkt_ret.rolling(126, min_periods=80).std() * np.sqrt(252),
            "mkt_trend_6m": cum / cum.rolling(126, min_periods=80).mean() - 1.0,
            "mkt_drawdown_1y": cum / cum.rolling(252, min_periods=120).max() - 1.0,
            "cs_disp": ret_d.std(axis=1).rolling(21, min_periods=10).mean(),
        }
    ).loc[[wide.index[int(i)] for i in meta["last_i"]]]
    regime_names = list(regime_df.columns)
    reg = regime_df.to_numpy(dtype=float)
    reg = np.nan_to_num((reg - np.nanmean(reg, axis=0)) / (np.nanstd(reg, axis=0) + 1e-12), nan=0.0)

    dates = pd.DatetimeIndex([wide.index[int(i)] for i in meta["last_i"]])
    return {
        "dates": dates,
        "codes": list(wide.columns),
        "features": feats,
        "ret_next": ret_next,
        "universe": np.isfinite(ret_next).astype(float),
        "tradable": np.isfinite(ret_next).astype(float),
        "regime": reg,
        "regime_names": regime_names,
    }


# --------------------------------------------------------------------------
# 实验
# --------------------------------------------------------------------------
def prepare(panel: dict) -> dict:
    from model import EXPERT_FEATURES

    names = [n for n in EXPERT_FEATURES if n in panel["features"]]
    scores = expert_scores({n: panel["features"][n] for n in names})
    uni = panel["universe"]
    ew = expert_weights(scores, uni)
    ret = np.nan_to_num(panel["ret_next"])
    expert_ret = np.einsum("ktn,tn->tk", ew, ret)
    vol = panel["features"].get("vol60", np.zeros_like(uni))
    return {
        "expert_w": ew,
        "conf": expert_confidence(scores, uni).T,
        "risk": expert_risk(ew, vol).T,
        "perf": recent_performance(expert_ret.T),
        "expert_ret": expert_ret,
        "asset_ret": panel["ret_next"],
    }


def make_windows(dates, first_test: int, last_test: int, train_years: int = 5, valid_years: int = 1) -> list[dict]:
    years = pd.DatetimeIndex(dates).year.to_numpy()
    out = []
    for ty in range(first_test, last_test + 1):
        if ty - valid_years - train_years < years.min():
            continue
        f = lambda y, s: int(np.searchsorted(years, y, side=s))  # noqa: E731
        tr_lo, tr_hi = f(ty - valid_years - train_years, "left"), f(ty - valid_years - 1, "right")
        va_lo, va_hi = f(ty - 1, "left"), f(ty - 1, "right")
        te_lo, te_hi = f(ty, "left"), f(ty, "right")
        if tr_hi - tr_lo < 12 or te_hi - te_lo < 1:
            continue
        out.append({"test_year": ty, "train": (tr_lo, tr_hi), "valid": (va_lo, va_hi), "test": (te_lo, te_hi)})
    return out


def run_sample(panel: dict, prep: dict, windows: list[dict], *, cost_bps: float,
               seeds: tuple[int, ...], epochs: int = 200) -> pd.DataFrame:
    rows = []
    uni, ret = panel["universe"], panel["ret_next"]
    ew_static = prep["expert_w"].mean(axis=0)

    def add(name, w_full, seed, extra=None):
        for w in windows:
            te_lo, te_hi = w["test"]
            res = bt.run_backtest(w_full[te_lo:te_hi], ret[te_lo:te_hi], panel["tradable"][te_lo:te_hi],
                                  uni[te_lo:te_hi], cost_bps=cost_bps)
            m = bt.performance(res.net, res.benchmark, res.turnover)
            m.update(res.diagnostics)
            row = {"model": name, "seed": seed, "test_year": w["test_year"], "cost_bps": cost_bps, **m}
            if extra:
                row.update(extra)
            rows.append(row)

    add("equal_weight", np.where(uni > 0, 1.0, 0.0) / np.maximum((uni > 0).sum(axis=1, keepdims=True), 1), -1)
    add("static_experts", ew_static, -1)
    for k in range(prep["expert_w"].shape[0]):
        add(f"expert{k}", prep["expert_w"][k], -1)

    for seed in seeds:
        best = None
        for risk_pen, turn_pen in ((0.1, 0.05), (0.3, 0.2), (0.6, 0.4)):
            cfg = GateConfig(seed=seed, epochs=epochs, objective="active",
                             risk_penalty=risk_pen, turnover_penalty=turn_pen)
            model, diag = train_gate(panel["regime"], prep["conf"], prep["risk"], prep["perf"].T,
                                     prep["expert_w"], ret, cfg)
            w_all = np.einsum("tk,ktn->tn", predict_gate(model, panel["regime"], prep["conf"], prep["risk"], prep["perf"].T),
                              prep["expert_w"])
            for w in windows:
                va_lo, va_hi = w["valid"]
                rv = bt.run_backtest(w_all[va_lo:va_hi], ret[va_lo:va_hi], panel["tradable"][va_lo:va_hi],
                                     uni[va_lo:va_hi], cost_bps=cost_bps)
                s = bt.sharpe_from_series(rv.net)
                if best is None or (np.isfinite(s) and s > best[0]):
                    best = (s, risk_pen, turn_pen, w_all, diag)
        _, rp, tp, w_all, diag = best
        add("gate_active", w_all, seed, {"risk_penalty": rp, "turnover_penalty": tp,
                                        "gate_dispersion": diag["gate_dispersion"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="etf10", choices=["etf10", "global45"])
    parser.add_argument("--first-test", type=int, default=2014)
    parser.add_argument("--last-test", type=int, default=2025)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--seeds", default="7,19,42")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tickers = ETF10 if args.sample == "etf10" else sorted(set(ETF10 + GLOBAL))
    panel = build_intl_panel(tickers)
    prep = prepare(panel)
    windows = make_windows(panel["dates"], args.first_test, args.last_test)
    print(f"[{args.sample}] N={len(panel['codes'])} T={len(panel['dates'])} windows={len(windows)}")

    frame = run_sample(panel, prep, windows, cost_bps=args.cost_bps,
                       seeds=tuple(int(s) for s in args.seeds.split(",")), epochs=args.epochs)
    out_dir = RESULTS / (args.out or f"exp_intl_{args.sample}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "window_metrics.csv", index=False)

    cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "cvar_5", "avg_turnover",
            "mean_active", "sharpe_active", "mean_active_exposure"]
    my = frame.groupby(["model", "test_year"])[cols].mean().reset_index()
    summary = my.groupby("model")[cols].mean().sort_values("sharpe", ascending=False)
    summary.to_csv(out_dir / "summary.csv")
    print(summary.round(4).to_string())

    # 互补性阈值检验（E2）在窄横截面上的结果
    feats = np.stack([panel["features"][k] for k in ("mom120", "mom20", "mom250", "vol60", "turnover20", "beta120")], axis=-1)
    idx = panel["regime_names"].index("mkt_ret_12m")
    states = cp.assign_states(panel["regime"][:, idx], 3)
    comp = cp.estimate_complementarity(feats, panel["ret_next"], panel["universe"], states,
                                       feature_names=["mom120", "mom20", "mom250", "vol60", "turnover20", "beta120"],
                                       n_boot=500, block=6)
    (out_dir / "complementarity.json").write_text(
        json.dumps(comp.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[E2/{args.sample}] gamma={comp.gamma:.6f} theta95={comp.theta_95:.6f} "
          f"exceeds={comp.exceeds_threshold} p={comp.p_value:.3f} months={comp.n_months}")

    (out_dir / "manifest.json").write_text(json.dumps({
        "sample": args.sample, "tickers": panel["codes"], "n_assets": len(panel["codes"]),
        "n_periods": len(panel["dates"]), "first_test": args.first_test, "last_test": args.last_test,
        "cost_bps": args.cost_bps, "windows": [w["test_year"] for w in windows],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
