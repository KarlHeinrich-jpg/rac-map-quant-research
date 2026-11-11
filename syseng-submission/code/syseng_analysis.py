# -*- coding: utf-8 -*-
"""《系统工程》专刊投稿版：影响因素与边际效应分析。

输入：results/ 下已有的滚动实验输出（不重跑模型，只做二次分析）。
输出：syseng/results/*.csv|json 与 syseng/figures/*.pdf|png（黑白图）。

设计：
- 影响因素：年度市场环境（波动、收益、制度切换、横截面离散度）与门控相对
  等权的夏普差值的关联（N=12，描述性关联，年度层面 bootstrap）。
- 边际效应：交易成本（e7 面板）、制度数（e4a）、训练窗长度（e8）、
  制度定义（e9）四个设计维度的边际效应，均以测试年份为聚类单位做 bootstrap。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "syseng" / "results"
FIG = ROOT / "syseng" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260930)
N_BOOT = 2000


# ---------------------------------------------------------------- 基础数据
def load_year_env() -> pd.DataFrame:
    """年度市场环境变量（由 data/index_daily.parquet 的月度序列聚合）。"""
    idx = pd.read_parquet(ROOT / "data" / "index_daily.parquet")
    idx["ym"] = pd.to_datetime(idx["ym_period_str"] + "-01")
    idx["year"] = idx["ym"].dt.year

    # 制度划分与主结果一致：12 个月市场收益的三分位（扩展窗标准化 -> 分位）
    v = idx["mkt_ret_12m"].astype(float)
    tercile = pd.qcut(v.rank(method="first"), 3, labels=False)
    idx["state"] = tercile
    idx["switch"] = (idx["state"] != idx["state"].shift(1)).astype(float)
    idx.loc[idx["state"].isna(), "switch"] = np.nan

    g = idx.groupby("year")
    out = pd.DataFrame(
        {
            "mkt_ret_yr": g["mkt_ret"].apply(lambda s: float(np.prod(1 + s) - 1)),
            "mkt_vol_yr": g["mkt_ret"].std(ddof=1) * np.sqrt(12),
            "cs_disp_yr": g["cs_disp"].mean(),
            "n_switch": g["switch"].sum(),
            "drawdown_yr": g["mkt_drawdown_1y"].min(),
            "n_months": g["mkt_ret"].size,
        }
    )
    out = out.reset_index()
    return out


def load_main() -> pd.DataFrame:
    d = pd.read_csv(RES / "exp_ashare_active" / "window_metrics.csv")
    d = d[d["test_year"].between(2014, 2025)]
    gate = d[d["model"] == "gate_active"][["test_year", "seed", "sharpe", "avg_turnover", "mean_active_exposure"]]
    gate = gate.rename(columns={"sharpe": "sharpe_gate"})
    ew = d[d["model"] == "equal_weight"][["test_year", "sharpe"]].rename(columns={"sharpe": "sharpe_ew"})
    m = gate.merge(ew, on="test_year", how="left")
    m["sharpe_diff"] = m["sharpe_gate"] - m["sharpe_ew"]
    return m


def load_ew_by_year() -> pd.Series:
    d = pd.read_csv(RES / "exp_ashare_active" / "window_metrics.csv")
    d = d[d["model"] == "equal_weight"]
    return d.set_index("test_year")["sharpe"]


# ---------------------------------------------------------------- 统计工具
def year_bootstrap_slope(x: np.ndarray, y: np.ndarray, years: np.ndarray,
                         n_boot: int = N_BOOT) -> dict:
    """以测试年份为重抽样单位的单变量回归斜率 bootstrap。"""
    def ols(xx, yy):
        xx = np.asarray(xx, float)
        yy = np.asarray(yy, float)
        xx_c = xx - xx.mean()
        denom = float((xx_c ** 2).sum())
        if denom <= 0:
            return np.nan, np.nan
        beta = float((xx_c * (yy - yy.mean())).sum() / denom)
        alpha = float(yy.mean() - beta * xx.mean())
        resid = yy - (alpha + beta * xx)
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        return beta, r2

    beta, r2 = ols(x, y)
    uniq = np.unique(years)
    by_year = {u: np.where(years == u)[0] for u in uniq}
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_year[u] for u in pick])
        b, _ = ols(np.asarray(x)[idx], np.asarray(y)[idx])
        if np.isfinite(b):
            boots.append(b)
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan)
    # 双侧 bootstrap p（对零的偏离）
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean())) if boots.size else np.nan
    return {"beta": beta, "r2": r2, "ci_lo": float(lo), "ci_hi": float(hi), "p": p,
            "n_obs": int(len(x)), "n_years": int(len(uniq))}


def year_cluster_slope(df: pd.DataFrame, y: str, x: str, year: str = "test_year",
                       n_boot: int = N_BOOT) -> dict:
    """面板（含年份固定效应）中 x 对 y 的斜率，年份聚类 bootstrap。"""
    def fit(d):
        d = d.copy()
        d["y_dm"] = d[y] - d.groupby(year)[y].transform("mean")
        d["x_dm"] = d[x] - d.groupby(year)[x].transform("mean")
        xx = d["x_dm"].to_numpy(float)
        yy = d["y_dm"].to_numpy(float)
        denom = float((xx ** 2).sum())
        return float((xx * yy).sum() / denom) if denom > 0 else np.nan

    beta = fit(df)
    uniq = np.unique(df[year].to_numpy())
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        parts = []
        for i, u in enumerate(pick):
            sub = df[df[year] == u].copy()
            sub["_by"] = i  # 重抽后的年份标签，保证同年观测同组
            parts.append(sub)
        bb = pd.concat(parts, ignore_index=True)
        b = fit(bb)
        if np.isfinite(b):
            boots.append(b)
    boots = np.asarray(boots)
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots.size else (np.nan, np.nan))
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean())) if boots.size else np.nan
    return {"beta": beta, "ci_lo": float(lo), "ci_hi": float(hi), "p": p,
            "n_obs": int(len(df)), "n_years": int(len(uniq))}


def fmt(x, nd=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


# 外生冲击年（事件时间由外部事件日历决定，与模型和策略无关）
SHOCK_YEARS = {
    2015: "股市异常波动",
    2016: "熔断机制实施",
    2018: "中美贸易摩擦",
    2020: "新冠疫情冲击",
    2022: "俄乌冲突与疫情封控",
}


def shock_analysis(yr: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """外生冲击年与对照年的门控价值对比（年份聚类 bootstrap）。"""
    d = yr.copy()
    d["shock"] = d["year"].isin(SHOCK_YEARS).astype(int)
    shock = d.loc[d["shock"] == 1, "sharpe_diff"].to_numpy(float)
    ctrl = d.loc[d["shock"] == 0, "sharpe_diff"].to_numpy(float)
    diff = float(shock.mean() - ctrl.mean())
    boots = []
    for _ in range(n_boot):
        s = RNG.choice(shock, size=len(shock), replace=True)
        c = RNG.choice(ctrl, size=len(ctrl), replace=True)
        boots.append(s.mean() - c.mean())
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))

    def grp(col):
        return (float(d.loc[d["shock"] == 1, col].mean()),
                float(d.loc[d["shock"] == 0, col].mean()))

    vol_s, vol_c = grp("mkt_vol_yr")
    sw_s, sw_c = grp("n_switch")
    disp_s, disp_c = grp("cs_disp_yr")
    out = {
        "shock_years": {str(k): v for k, v in SHOCK_YEARS.items()},
        "shock_mean": float(shock.mean()),
        "control_mean": float(ctrl.mean()),
        "diff": diff, "ci_lo": float(lo), "ci_hi": float(hi), "p": p,
        "n_shock": int(len(shock)), "n_control": int(len(ctrl)),
        "vol_shock": vol_s, "vol_control": vol_c,
        "switch_shock": sw_s, "switch_control": sw_c,
        "disp_shock": disp_s, "disp_control": disp_c,
    }
    return out


# ---------------------------------------------------------------- 分析主体
def main() -> None:
    env = load_year_env()
    env = env[env["year"].between(2014, 2025)].reset_index(drop=True)
    main_df = load_main()
    by_year = (
        main_df.groupby("test_year")
        .agg(sharpe_gate=("sharpe_gate", "mean"),
             sharpe_ew=("sharpe_ew", "first"),
             sharpe_diff=("sharpe_diff", "mean"),
             turnover=("avg_turnover", "mean"),
             exposure=("mean_active_exposure", "mean"))
        .reset_index()
        .rename(columns={"test_year": "year"})
    )
    yr = by_year.merge(env, on="year", how="left")
    yr.to_csv(OUT / "year_level.csv", index=False, encoding="utf-8-sig")

    # ---- 影响因素：年度环境的单变量回归 ----
    rows = []
    covs = [("mkt_vol_yr", "市场波动率（年化）"),
            ("n_switch", "制度切换次数"),
            ("mkt_ret_yr", "市场年度收益"),
            ("cs_disp_yr", "横截面离散度"),
            ("drawdown_yr", "年度最大回撤")]
    for c, label in covs:
        r = year_bootstrap_slope(yr[c].to_numpy(float), yr["sharpe_diff"].to_numpy(float),
                                 yr["year"].to_numpy())
        rows.append({"变量": label, "系数": r["beta"], "CI下": r["ci_lo"], "CI上": r["ci_hi"],
                     "p": r["p"], "R2": r["r2"], "N": r["n_obs"]})
    det = pd.DataFrame(rows)
    sp = {c: float(pd.Series(yr[c]).corr(pd.Series(yr["sharpe_diff"]), method="spearman"))
          for c, _ in covs}
    det["秩相关"] = [sp[c] for c, _ in covs]
    det.to_csv(OUT / "determinants_year.csv", index=False, encoding="utf-8-sig")

    # 三分组对照：波动率/制度切换高低的门控差值
    terc = {}
    for c in ["mkt_vol_yr", "n_switch", "cs_disp_yr"]:
        q = pd.qcut(yr[c], 3, labels=["低", "中", "高"])
        g = yr.groupby(q, observed=False)["sharpe_diff"].mean()
        terc[c] = {str(k): float(v) for k, v in g.items()}
    (OUT / "determinants_terciles.json").write_text(
        json.dumps(terc, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 外生冲击分组 ----
    shock = shock_analysis(yr)
    (OUT / "shock_analysis.json").write_text(
        json.dumps(shock, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 边际效应 1：交易成本 ----
    e7 = pd.read_csv(RES / "robustness" / "e7_cost_sensitivity.csv")
    e7 = e7[e7["test_year"].between(2014, 2025)].copy()
    e7["sharpe_diff"] = e7["sharpe_gate"] - e7["sharpe_ew"]
    cost_fit = year_cluster_slope(e7, "sharpe_diff", "cost_bps")
    lvl = (e7.groupby("cost_bps")["sharpe_diff"].mean().reset_index())
    lvl.to_csv(OUT / "marginal_cost_levels.csv", index=False, encoding="utf-8-sig")
    # 盈亏平衡成本（线性外推，注明为外推）
    b0 = float(lvl["sharpe_diff"].iloc[0])
    slope = cost_fit["beta"]
    breakeven = -b0 / slope if slope < 0 else np.inf
    cost_fit["level_0bp"] = b0
    cost_fit["breakeven_bps"] = float(breakeven)

    # ---- 边际效应 2：制度数 ----
    e4 = pd.read_csv(RES / "hypothesis_main" / "e4a_state_count.csv")
    e4 = e4[e4["test_year"].between(2014, 2025)].copy()
    e4["n_states"] = e4["model"].str.extract(r"S(\d)").astype(int)
    e4 = e4.merge(by_year[["year", "sharpe_ew"]].rename(columns={"year": "test_year"}),
                  on="test_year", how="left")
    e4["sharpe_diff"] = e4["sharpe"] - e4["sharpe_ew"]
    state_fit = year_cluster_slope(e4, "sharpe_diff", "n_states")
    state_lvl = e4.groupby("n_states")["sharpe_diff"].mean().reset_index()
    state_lvl.to_csv(OUT / "marginal_states_levels.csv", index=False, encoding="utf-8-sig")
    s_vals = state_lvl.set_index("n_states")["sharpe_diff"].to_dict()
    state_step = {"d_2to3": float(s_vals.get(3, np.nan) - s_vals.get(2, np.nan)),
                  "d_3to4": float(s_vals.get(4, np.nan) - s_vals.get(3, np.nan))}

    # ---- 边际效应 3：训练窗长度 ----
    e8 = pd.read_csv(RES / "robustness" / "e8_train_length.csv")
    e8 = e8[e8["test_year"].between(2014, 2025)].copy()
    e8["sharpe_diff"] = e8["sharpe_gate"] - e8["sharpe_ew"]
    train_fit = year_cluster_slope(e8, "sharpe_diff", "train_years")
    train_lvl = e8.groupby("train_years")["sharpe_diff"].mean().reset_index()
    train_lvl.to_csv(OUT / "marginal_train_levels.csv", index=False, encoding="utf-8-sig")

    # ---- 制度定义离散度（e9） ----
    e9 = pd.read_csv(RES / "robustness" / "e9_state_definition.csv")
    e9 = e9[e9["test_year"].between(2014, 2025)].copy()
    e9 = e9.merge(by_year[["year", "sharpe_ew"]].rename(columns={"year": "test_year"}),
                  on="test_year", how="left")
    e9["sharpe_diff"] = e9["sharpe"] - e9["sharpe_ew"]
    e9g = (e9.groupby(["state_var", "n_states"])
           .agg(sharpe_diff=("sharpe_diff", "mean"), sharpe=("sharpe", "mean"))
           .reset_index())
    e9g.to_csv(OUT / "state_definition.csv", index=False, encoding="utf-8-sig")
    spread = float(e9g["sharpe_diff"].max() - e9g["sharpe_diff"].min())

    # ---- 理论量：Γ、Θ、持续性 ----
    ch = json.loads((RES / "theory_main" / "channel_decomposition.json").read_text(encoding="utf-8"))
    th = json.loads((RES / "theory_main" / "optimal_states.json").read_text(encoding="utf-8"))
    tr = json.loads((RES / "theory_main" / "transition.json").read_text(encoding="utf-8"))
    summary_txt = (RES / "theory_main" / "summary.txt").read_text(encoding="utf-8")

    marg = {
        "cost": cost_fit,
        "states": state_fit,
        "state_steps": state_step,
        "train": train_fit,
        "state_definition_spread": spread,
        "shock": shock,
        "gamma1": ch["gamma1"], "theta1": ch["theta1_95"], "p1": ch["p1"],
        "gamma2": ch["gamma2"], "theta2": ch["theta2_95"], "p2": ch["p2"],
        "threshold_gap1": ch["gamma1"] - ch["theta1_95"],
        "threshold_gap2": ch["gamma2"] - ch["theta2_95"],
        "delta_curve": th["curve"],
        "transition": tr,
    }
    (OUT / "marginal_effects.json").write_text(
        json.dumps(marg, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ---- 控制台汇总 ----
    print("=" * 74)
    print("年度层面（影响因素）：N =", len(yr))
    print(f"{'变量':<16}{'系数':>12}{'95%CI':>26}{'p':>8}{'R2':>8}{'秩相关':>9}")
    for _, r in det.iterrows():
        ci = f"[{r['CI下']:.4f}, {r['CI上']:.4f}]"
        print(f"{r['变量']:<16}{r['系数']:>12.4f}{ci:>26}{r['p']:>8.3f}{r['R2']:>8.3f}{r['秩相关']:>9.3f}")
    print("\n三分组均值（门控−等权夏普差）")
    for k, v in terc.items():
        print(f"  {k}: {v}")
    print("\n边际效应：交易成本")
    print(f"  斜率 = {cost_fit['beta']:.5f} /基点  CI [{cost_fit['ci_lo']:.5f}, {cost_fit['ci_hi']:.5f}]"
          f"  p = {cost_fit['p']:.3f}  N = {cost_fit['n_obs']}")
    print(f"  每 10 基点边际效应 = {cost_fit['beta'] * 10:.4f}")
    print(f"  0 基点水平 = {cost_fit['level_0bp']:.4f}；盈亏平衡成本 = {cost_fit['breakeven_bps']:.1f} 基点")
    print("  各成本水平均值：", {int(r.cost_bps): round(r.sharpe_diff, 4) for r in lvl.itertuples()})
    print("\n边际效应：制度数")
    print(f"  斜率 = {state_fit['beta']:.5f} /个状态  CI [{state_fit['ci_lo']:.5f}, {state_fit['ci_hi']:.5f}]"
          f"  p = {state_fit['p']:.3f}  N = {state_fit['n_obs']}")
    print("  各制度数均值：", {int(r.n_states): round(r.sharpe_diff, 4) for r in state_lvl.itertuples()})
    print("\n边际效应：训练窗")
    print(f"  斜率 = {train_fit['beta']:.5f} /年  CI [{train_fit['ci_lo']:.5f}, {train_fit['ci_hi']:.5f}]"
          f"  p = {train_fit['p']:.3f}  N = {train_fit['n_obs']}")
    print("  各窗长均值：", {int(r.train_years): round(r.sharpe_diff, 4) for r in train_lvl.itertuples()})
    print("\n制度定义离散度：", round(spread, 4))
    print("\n阈值距离：一阶", f"{marg['threshold_gap1']:.3e}", "二阶", f"{marg['threshold_gap2']:.3e}")
    print("\n外生冲击分组（门控−等权夏普差）")
    print(f"  冲击年（{list(SHOCK_YEARS)}）：均值 {shock['shock_mean']:.4f}")
    print(f"  对照年均值 {shock['control_mean']:.4f}；差 {shock['diff']:+.4f}"
          f"  95%CI [{shock['ci_lo']:.4f}, {shock['ci_hi']:.4f}]  p={shock['p']:.3f}")
    print(f"  市场波动率：冲击年 {shock['vol_shock']:.3f} vs 对照 {shock['vol_control']:.3f}")
    print(f"  制度切换次数：冲击年 {shock['switch_shock']:.2f} vs 对照 {shock['switch_control']:.2f}")
    print(f"  横截面离散度：冲击年 {shock['disp_shock']:.4f} vs 对照 {shock['disp_control']:.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
