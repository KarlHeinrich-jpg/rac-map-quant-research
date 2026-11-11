r"""稳健性扩展实验。

E6  子样本稳定性：把测试年份分成两段，检验门控优势是否稳定；
E7  交易成本敏感性（H3）：$\kappa\in\{0,5,10,20,30,50\}$ 基点；
E8  容量曲线：训练窗长度 $T$ 与横截面宽度 $N$ 对门控优势的影响（H2、H4）；
E9  状态变量稳健性：换用不同状态定义（市场波动、回撤、横截面离散度）。

输出到 ``results/robustness/``。
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
import theory_core as tc  # noqa: E402
from model import (  # noqa: E402
    EXPERT_FEATURES,
    GateConfig,
    expert_confidence,
    expert_risk,
    expert_scores,
    expert_weights,
    predict_gate,
    recent_performance,
    train_gate,
)

RESULTS = ROOT / "results"


def prepare(panel: pdmod.PanelData) -> dict:
    names = [n for n in EXPERT_FEATURES if n in panel.features]
    scores = expert_scores({n: panel.features[n] for n in names})
    uni = panel.in_universe
    ew = expert_weights(scores, uni)
    ret = np.nan_to_num(panel.ret_next)
    expert_ret = np.einsum("ktn,tn->tk", ew, ret)
    vol = panel.features.get("vol60", np.zeros_like(uni))
    return {
        "expert_w": ew,
        "conf": expert_confidence(scores, uni).T,
        "risk": expert_risk(ew, vol).T,
        "perf": recent_performance(expert_ret.T),
        "expert_ret": expert_ret,
    }


def windows(dates, first_test: int, last_test: int, train_years: int = 5, valid_years: int = 1) -> list[dict]:
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
        out.append({"test_year": ty, "train": (tr_lo, tr_hi), "valid": (va_lo, va_hi),
                    "test": (te_lo, te_hi), "train_years": train_years})
    return out


def run_gate_windows(panel, prep, wins, *, seeds, cost_bps, epochs, objective="active") -> pd.DataFrame:
    """与 ``experiments.py`` 完全一致的滚动协议。

    关键纪律：门控**只在训练窗内拟合**，验证窗用于选择超参数，
    测试窗仅用于评估。不得用全样本拟合后再切到测试窗——那会造成信息泄漏。
    """
    uni, ret = panel.in_universe, panel.ret_next
    rows = []
    for w in wins:
        tr_lo, tr_hi = w["train"]
        va_lo, va_hi = w["valid"]
        te_lo, te_hi = w["test"]
        for seed in seeds:
            best = None
            for rp, tp in ((0.1, 0.05), (0.3, 0.2), (0.6, 0.4)):
                cfg = GateConfig(seed=seed, epochs=epochs, objective=objective,
                                 risk_penalty=rp, turnover_penalty=tp)
                model, _ = train_gate(panel.regime[tr_lo:tr_hi], prep["conf"][tr_lo:tr_hi],
                                      prep["risk"][tr_lo:tr_hi], prep["perf"][:, tr_lo:tr_hi].T,
                                      prep["expert_w"][:, tr_lo:tr_hi], ret[tr_lo:tr_hi], cfg)
                g_va = predict_gate(model, panel.regime[va_lo:va_hi], prep["conf"][va_lo:va_hi],
                                    prep["risk"][va_lo:va_hi], prep["perf"][:, va_lo:va_hi].T)
                wva = np.einsum("tk,ktn->tn", g_va, prep["expert_w"][:, va_lo:va_hi])
                res = bt.run_backtest(wva, ret[va_lo:va_hi], panel.tradable[va_lo:va_hi],
                                      uni[va_lo:va_hi], cost_bps=cost_bps)
                s = bt.sharpe_from_series(res.net)
                if best is None or (np.isfinite(s) and s > best[0]):
                    best = (s, model)
            model = best[1]
            # 只对测试窗做预测，避免全样本预测带来的任何泄漏
            g = predict_gate(model, panel.regime[te_lo:te_hi], prep["conf"][te_lo:te_hi],
                             prep["risk"][te_lo:te_hi], prep["perf"][:, te_lo:te_hi].T)
            wg = np.einsum("tk,ktn->tn", g, prep["expert_w"][:, te_lo:te_hi])
            rg = bt.run_backtest(wg, ret[te_lo:te_hi], panel.tradable[te_lo:te_hi],
                                 uni[te_lo:te_hi], cost_bps=cost_bps)
            we = np.where(uni[te_lo:te_hi] > 0, 1.0, 0.0)
            we = we / np.maximum(we.sum(axis=1, keepdims=True), 1e-12)
            re = bt.run_backtest(we, ret[te_lo:te_hi], panel.tradable[te_lo:te_hi],
                                 uni[te_lo:te_hi], cost_bps=cost_bps)
            mg = bt.performance(rg.net, rg.benchmark, rg.turnover)
            me = bt.performance(re.net, re.benchmark, re.turnover)
            rows.append({
                "test_year": w["test_year"], "seed": seed, "train_years": w["train_years"],
                "cost_bps": cost_bps,
                "sharpe_gate": mg["sharpe"], "sharpe_ew": me["sharpe"],
                "sharpe_diff": mg["sharpe"] - me["sharpe"],
                "active_gate": mg.get("mean_active", np.nan),
                "turnover_gate": mg["avg_turnover"],
                "exposure_gate": rg.diagnostics["mean_active_exposure"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test", type=int, default=2014)
    parser.add_argument("--last-test", type=int, default=2025)
    parser.add_argument("--seeds", default="7,19")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--out", default="robustness")
    args = parser.parse_args()

    out_dir = RESULTS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))

    panel = pdmod.build_panel_data()
    prep = prepare(panel)
    base_wins = windows(panel.dates, args.first_test, args.last_test)
    print(f"panel T={panel.n_periods} N={panel.n_assets}; windows={len(base_wins)}")

    report: dict = {}

    # ---------------- E7：交易成本敏感性（H3） ----------------
    print("\n[E7] 交易成本敏感性 ...", flush=True)
    cost_rows = []
    for kappa in (0.0, 5.0, 10.0, 20.0, 30.0, 50.0):
        df = run_gate_windows(panel, prep, base_wins, seeds=seeds, cost_bps=kappa, epochs=args.epochs)
        cost_rows.append(df)
        print(f"  kappa={kappa:5.1f}bp: 门控 {df.sharpe_gate.mean():.3f} vs 等权 {df.sharpe_ew.mean():.3f} "
              f"(差 {df.sharpe_diff.mean():+.4f})", flush=True)
    cost_df = pd.concat(cost_rows, ignore_index=True)
    cost_df.to_csv(out_dir / "e7_cost_sensitivity.csv", index=False)
    report["E7_cost"] = (cost_df.groupby("cost_bps")[["sharpe_gate", "sharpe_ew", "sharpe_diff",
                                                     "turnover_gate"]].mean().round(4).to_dict())

    # ---------------- E6：子样本稳定性 ----------------
    print("\n[E6] 子样本稳定性 ...", flush=True)
    base_df = run_gate_windows(panel, prep, base_wins, seeds=seeds, cost_bps=10.0, epochs=args.epochs)
    base_df.to_csv(out_dir / "e6_subsample.csv", index=False)
    my = base_df.groupby("test_year")[["sharpe_gate", "sharpe_ew", "sharpe_diff"]].mean()
    half = len(my) // 2
    seg = {"前半段": my.iloc[:half], "后半段": my.iloc[half:]}
    report["E6_subsample"] = {
        k: {"years": f"{v.index.min()}-{v.index.max()}", "n": int(len(v)),
            "sharpe_gate": float(v.sharpe_gate.mean()), "sharpe_ew": float(v.sharpe_ew.mean()),
            "sharpe_diff": float(v.sharpe_diff.mean())}
        for k, v in seg.items()
    }
    report["E6_subsample"]["逐年差值"] = my.sharpe_diff.round(4).to_dict()
    for k, v in seg.items():
        print(f"  {k} ({v.index.min()}-{v.index.max()}): 门控 {v.sharpe_gate.mean():.3f} "
              f"vs 等权 {v.sharpe_ew.mean():.3f} (差 {v.sharpe_diff.mean():+.4f})", flush=True)

    # ---------------- E8：训练窗长度（H4） ----------------
    print("\n[E8] 训练窗长度（H4） ...", flush=True)
    len_rows = []
    for ty_len in (3, 5, 8):
        wins = windows(panel.dates, args.first_test, args.last_test, train_years=ty_len)
        if not wins:
            continue
        df = run_gate_windows(panel, prep, wins, seeds=seeds, cost_bps=10.0, epochs=args.epochs)
        len_rows.append(df)
        print(f"  训练窗 {ty_len} 年: {len(df)} 窗口, 差 {df.sharpe_diff.mean():+.4f}", flush=True)
    len_df = pd.concat(len_rows, ignore_index=True)
    len_df.to_csv(out_dir / "e8_train_length.csv", index=False)
    report["E8_train_length"] = (len_df.groupby("train_years")[["sharpe_gate", "sharpe_ew", "sharpe_diff"]]
                                 .mean().round(4).to_dict())

    # ---------------- E9：状态定义稳健性 ----------------
    print("\n[E9] 状态定义稳健性 ...", flush=True)
    state_rows = []
    for var in ("mkt_ret_12m", "mkt_vol_6m", "mkt_drawdown_1y", "cs_disp"):
        if var not in panel.regime_names:
            continue
        idx = panel.regime_names.index(var)
        for S in (2, 3):
            states = cp.assign_states(panel.regime[:, idx], S)
            onehot = np.eye(S)[np.clip(states, 0, S - 1)]
            reg = np.concatenate([panel.regime, onehot], axis=1)
            p2 = dict(prep)
            for w in base_wins:
                tr_lo, tr_hi = w["train"]
                te_lo, te_hi = w["test"]
                cfg = GateConfig(seed=seeds[0], epochs=args.epochs, objective="active")
                model, _ = train_gate(reg[tr_lo:tr_hi], prep["conf"][tr_lo:tr_hi], prep["risk"][tr_lo:tr_hi],
                                      prep["perf"][:, tr_lo:tr_hi].T, prep["expert_w"][:, tr_lo:tr_hi],
                                      np.nan_to_num(panel.ret_next[tr_lo:tr_hi]), cfg)
                g = predict_gate(model, reg[te_lo:te_hi], prep["conf"][te_lo:te_hi],
                                 prep["risk"][te_lo:te_hi], prep["perf"][:, te_lo:te_hi].T)
                wg = np.einsum("tk,ktn->tn", g, prep["expert_w"][:, te_lo:te_hi])
                rg = bt.run_backtest(wg, panel.ret_next[te_lo:te_hi], panel.tradable[te_lo:te_hi],
                                     panel.in_universe[te_lo:te_hi], cost_bps=10.0)
                state_rows.append({"state_var": var, "n_states": S, "test_year": w["test_year"],
                                   "sharpe": bt.sharpe_from_series(rg.net),
                                   "active": float(np.nanmean(rg.active))})
    st_df = pd.DataFrame(state_rows)
    st_df.to_csv(out_dir / "e9_state_definition.csv", index=False)
    e9 = st_df.groupby(["state_var", "n_states"])[["sharpe", "active"]].mean().round(4)
    # 多级索引的键为元组，需转成字符串才能序列化为 JSON
    report["E9_state_definition"] = {
        f"{var}|S={int(S)}": {"sharpe": float(r["sharpe"]), "active": float(r["active"])}
        for (var, S), r in e9.iterrows()
    }
    print(e9.to_string(), flush=True)

    # ---------------- E10：区块长度敏感性（推断稳健性） ----------------
    print("\n[E10] 自助区块长度敏感性 ...", flush=True)
    names = [n for n in EXPERT_FEATURES if n in panel.features]
    er = prep["expert_ret"]
    st3 = cp.assign_states(panel.regime[:, panel.regime_names.index("mkt_ret_12m")], 3)
    stats_for_block = tc.RegimeStats.build(st3, er)
    block_rows = []
    for blk in (3, 6, 12, 24):
        d = tc.channel_decomposition(stats_for_block, n_boot=400, block=blk)
        block_rows.append({"block_months": blk, "gamma1": d["gamma1"], "theta1_95": d["theta1_95"],
                           "p1": d["p1"], "gamma2": d["gamma2"], "theta2_95": d["theta2_95"], "p2": d["p2"]})
        print(f"  块长 {blk:2d} 月: 一阶 p={d['p1']:.3f}  二阶 p={d['p2']:.3f}", flush=True)
    pd.DataFrame(block_rows).to_csv(out_dir / "e10_block_length.csv", index=False)
    report["E10_block_length"] = block_rows

    (out_dir / "robustness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果写入 {out_dir}")


if __name__ == "__main__":
    main()
