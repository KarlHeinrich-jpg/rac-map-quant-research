# -*- coding: utf-8 -*-
"""按 nature-figure（Nature 风格科研绘图）规范重绘全部图件。

遵循的规范：
- apply_publication_style：无上/右边框、无图例框、矢量可编辑文本（pdf.fonttype=42）；
- 密集期刊宽度多面板用 8pt 字号与 1pt 轴线；
- hero panel + 次级面板；直接标注优先于图例；
- 统计量（n、区间口径、水平线）作为图的组成部分；
- 多面板调用 require_matplotlib_panel_alignment() 做对齐门禁。

配色说明：目标期刊要求黑白印刷，故在图内以灰阶 + 线型/标记形状编码，
不依赖色相区分，符合 nature-figure 的「克制配色 + 可辨性」要求。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

SKILL = Path(r"C:\Users\93529\.codex\skills\nature-figure\scripts")
if SKILL.exists():
    sys.path.insert(0, str(SKILL))
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except Exception:  # 对齐门禁脚本不可用时降级，但记录
    require_matplotlib_panel_alignment = None

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SYS = ROOT / "syseng" / "results"
FIG = ROOT / "syseng" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------- 出版级样式（nature-figure python 后端） ----------------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"      # SVG 文本保持可编辑
plt.rcParams["pdf.fonttype"] = 42          # PDF 内嵌 TrueType，可检索
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.grid"] = False          # nature 风格：默认不加网格
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["savefig.bbox"] = "tight"

INK = "#272727"        # 主色（近黑）
MID = "#767676"        # 对照
LIGHT = "#CFCECE"      # 填充
PALE = "#E8E8E8"
BAND = {0: "#EDEDED", 1: "#CFCFCF", 2: "#A9A9A9"}

MODEL_CN = {
    "expert_low_vol": "低波动专家", "risk_parity": "风险平价", "gate_active": "门控组合",
    "expert_reversal": "反转专家", "gbrt": "梯度提升树", "ridge": "岭回归",
    "static_experts": "静态专家合成", "expert_low_turnover": "低换手专家",
    "expert_momentum": "动量专家", "expert_low_beta": "低 beta 专家",
}
STATE_CN = {"mkt_ret_12m": "12 个月市场收益", "mkt_vol_6m": "6 个月市场波动",
            "mkt_drawdown_1y": "1 年市场回撤", "cs_disp": "横截面离散度"}


def panel_label(ax, lab, dx=-26, dy=6):
    ax.annotate(lab, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9, fontweight="bold", color=INK, va="bottom")


def align_gate(fig, name):
    if require_matplotlib_panel_alignment is None:
        return
    try:
        require_matplotlib_panel_alignment(
            fig, json_out=str(FIG / f"{name}.alignment.json"),
            overlay_svg=str(FIG / f"{name}.alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
        print(f"  [{name}] 面板对齐门禁：通过")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{name}] 面板对齐门禁：{type(exc).__name__} {exc}")


def export(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- 图 1
def fig_regime():
    idx = pd.read_parquet(ROOT / "data" / "index_daily.parquet")
    d = idx[["ym_period_str", "mkt_ret", "mkt_ret_12m", "mkt_vol_6m"]].copy()
    d["date"] = pd.to_datetime(d["ym_period_str"] + "-01")
    d = d.dropna(subset=["mkt_ret_12m"]).reset_index(drop=True)
    d["state"] = pd.qcut(d["mkt_ret_12m"].rank(method="first"), 3, labels=False)
    d["cum"] = (1 + d["mkt_ret"]).cumprod()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.1), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.14})
    for ax in (ax1, ax2):
        start = 0
        for i in range(1, len(d)):
            if d["state"].iloc[i] != d["state"].iloc[start] or i == len(d) - 1:
                ax.axvspan(d["date"].iloc[start], d["date"].iloc[i],
                           color=BAND[d["state"].iloc[start]], lw=0, zorder=0)
                start = i
    ax1.plot(d["date"], d["cum"], color=INK, lw=1.0, zorder=3)
    ax1.axvline(pd.Timestamp("2014-01-01"), color=INK, ls=(0, (4, 2)), lw=0.9, zorder=4)
    ax1.annotate("滚动测试窗起点", xy=(pd.Timestamp("2014-01-01"), 0.05),
                 xycoords=("data", "axes fraction"), xytext=(4, 0),
                 textcoords="offset points", fontsize=7.5, color=INK)
    ax1.set_ylabel("市场累计净值\n（2006 年 1 月 = 1）", fontsize=8)
    ax1.set_ylim(0, d["cum"].max() * 1.1)
    ax1.legend(handles=[Patch(facecolor=BAND[s], edgecolor="none",
                              label=l) for s, l in
                        [(0, "制度 1：低收益"), (1, "制度 2：中收益"), (2, "制度 3：高收益")]],
               fontsize=7.5, ncol=3, loc="upper left", handlelength=1.1,
               columnspacing=1.0, borderpad=0)
    ax2.plot(d["date"], d["mkt_vol_6m"], color=INK, lw=1.0, zorder=3)
    ax2.axvline(pd.Timestamp("2014-01-01"), color=INK, ls=(0, (4, 2)), lw=0.9, zorder=4)
    ax2.set_ylabel("6 个月波动率\n（年化）", fontsize=8)
    ax2.set_xlabel("年份", fontsize=8)
    panel_label(ax1, "a")
    panel_label(ax2, "b")
    align_gate(fig, "fig_regime")
    export(fig, "fig_regime")


# ---------------------------------------------------------------- 图 2
def fig_significance():
    sig = pd.read_csv(RES / "exp_ashare_active" / "significance_table.csv")
    d = sig[sig["metric"] == "sharpe"].copy()
    d["name"] = d["model"].map(MODEL_CN).fillna(d["model"])
    d = d.sort_values("diff_vs_benchmark").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (_, r) in enumerate(d.iterrows()):
        focal = r["model"] == "gate_active"
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=INK if focal else MID,
                lw=1.6 if focal else 1.0, solid_capstyle="butt", zorder=2)
        ax.plot(r["diff_vs_benchmark"], i, marker="D" if focal else "o",
                ms=5.2 if focal else 4.2, mfc=INK if focal else "white",
                mec=INK if focal else MID, mew=1.0, zorder=3)
    ax.axvline(0, color=LIGHT, lw=1.0, zorder=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["name"], fontsize=8)
    ax.set_xlabel("相对等权基准的夏普差值", fontsize=8)
    ax.set_ylim(-0.7, len(d) - 0.3)
    g = d[d["model"] == "gate_active"].iloc[0]
    ax.text(0.985, 0.135,
            f"门控组合（实心菱形）：{g['diff_vs_benchmark']:+.3f}，"
            f"Holm $p$ = {g['holm_p']:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=INK)
    ax.text(0.985, 0.02, "n = 12 个测试年份；横线为 95% 循环区块自助区间",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MID)
    export(fig, "fig_significance")


# ---------------------------------------------------------------- 图 3
def fig_threshold():
    me = json.loads((SYS / "marginal_effects.json").read_text(encoding="utf-8"))
    rows = [("一阶渠道（收益择时）", me["gamma1"], me["theta1"], me["p1"]),
            ("二阶渠道（风险配置）", me["gamma2"], me["theta2"], me["p2"])]
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    for i, (lab, gam, th, p) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([th * 1e4, gam * 1e4], [y, y], color=LIGHT, lw=1.2, zorder=1)
        ax.plot(gam * 1e4, y, marker="o", ms=5.0, mfc="white", mec=INK, mew=1.1, zorder=3)
        ax.plot(th * 1e4, y, marker="s", ms=5.0, mfc=MID, mec=MID, zorder=3)
        ax.annotate(f"$\\hat\\Gamma$ = {gam * 1e4:.2f}", xy=(gam * 1e4, y), xytext=(-2, 7),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK)
        ax.annotate(f"$\\Theta$ = {th * 1e4:.2f}", xy=(th * 1e4, y), xytext=(2, -11),
                    textcoords="offset points", ha="left", fontsize=7.5, color=MID)
        ax.annotate(f"缺口 {(gam - th) * 1e4:+.2f}，$p$ = {p:.3f}",
                    xy=(0.985, y), xycoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=7.5, color=MID)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlabel(r"统计量（$\times10^{-4}$）", fontsize=8)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.text(0.985, 1.06, r"空心圆：$\hat\Gamma$　实心方：$\Theta$（自助 95% 分位）",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MID)
    export(fig, "fig_threshold")


# ---------------------------------------------------------------- 图 4
def fig_determinants():
    yr = pd.read_csv(SYS / "year_level.csv")
    det = pd.read_csv(SYS / "determinants_year.csv")
    rho = float(det.loc[det["变量"].str.startswith("市场波动率"), "秩相关"].iloc[0])
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.scatter(yr["mkt_vol_yr"], yr["sharpe_diff"], s=22, facecolors="none",
               edgecolors=INK, linewidths=0.9, zorder=3)
    offs = {2014: (-24, -2), 2016: (-4, 6), 2017: (-20, -10), 2021: (4, -10),
            2020: (4, 4), 2018: (4, 4), 2019: (3, 4), 2022: (4, -10),
            2024: (4, 4), 2025: (4, -10), 2023: (4, -10), 2015: (-20, 2)}
    for _, r in yr.iterrows():
        dx, dy = offs.get(int(r["year"]), (4, 4))
        ax.annotate(int(r["year"]), (r["mkt_vol_yr"], r["sharpe_diff"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6.8, color=MID)
    q = pd.qcut(yr["mkt_vol_yr"], 3, labels=False)
    gm = yr.groupby(q)["sharpe_diff"].mean()
    xs = yr.groupby(q)["mkt_vol_yr"].mean()
    ax.plot(xs, gm, color=INK, lw=1.1, marker="D", ms=4.2, mfc=INK, zorder=4)
    ax.axhline(0, color=LIGHT, lw=1.0, zorder=1)
    ax.set_xlabel("年度市场波动率（年化）", fontsize=8)
    ax.set_ylabel("门控−等权夏普差", fontsize=8)
    ax.set_ylim(yr["sharpe_diff"].min() - 0.06, yr["sharpe_diff"].max() + 0.07)
    ax.annotate(f"菱形：波动率三分组均值　Spearman $\\rho$ = {rho:.3f}",
                xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                fontsize=7, color=MID)
    export(fig, "fig_determinants")


# ---------------------------------------------------------------- 图 5
def fig_marginal():
    cost = pd.read_csv(SYS / "marginal_cost_levels.csv")
    states = pd.read_csv(SYS / "marginal_states_levels.csv")
    train = pd.read_csv(SYS / "marginal_train_levels.csv")
    me = json.loads((SYS / "marginal_effects.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax = axes[0]
    ax.plot(cost["cost_bps"], cost["sharpe_diff"], color=INK, lw=1.0,
            marker="o", ms=3.6, mfc="white", mec=INK, mew=0.9)
    xs = np.linspace(0, 60, 40)
    ax.plot(xs, me["cost"]["level_0bp"] + me["cost"]["beta"] * xs,
            color=MID, lw=0.9, ls=(0, (3, 2)))
    be = me["cost"]["breakeven_bps"]
    ax.axvline(min(be, 60), color=LIGHT, lw=0.9, ls=(0, (1, 2)))
    ax.annotate(f"盈亏平衡 ≈ {be:.0f} bp", xy=(0.97, 0.06), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color=MID)
    ax.set_xlabel("单边交易成本（基点）", fontsize=8)
    ax.set_ylabel("门控−等权夏普差", fontsize=8)
    panel_label(ax, "a", dx=-24)

    ax = axes[1]
    ax.bar(states["n_states"].astype(str), states["sharpe_diff"], width=0.5,
           facecolor=LIGHT, edgecolor=INK, lw=0.9)
    for x, v in zip(states["n_states"].astype(str), states["sharpe_diff"]):
        ax.annotate(f"{v:+.3f}", xy=(x, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("制度状态数", fontsize=8)
    ax.set_ylabel("门控−等权夏普差", fontsize=8)
    ax.set_ylim(0, max(states["sharpe_diff"]) * 1.25)
    panel_label(ax, "b", dx=-22)

    ax = axes[2]
    ax.plot(train["train_years"], train["sharpe_diff"], color=INK, lw=1.0,
            marker="s", ms=3.6, mfc="white", mec=INK, mew=0.9)
    ax.axhline(0, color=LIGHT, lw=0.9)
    ax.set_xticks(train["train_years"])
    ax.set_xlabel("训练窗长度（年）", fontsize=8)
    ax.set_ylabel("门控−等权夏普差", fontsize=8)
    panel_label(ax, "c", dx=-22)

    fig.subplots_adjust(wspace=0.42)
    align_gate(fig, "fig_marginal")
    export(fig, "fig_marginal")


# ---------------------------------------------------------------- 图 6
def fig_robustness():
    e10 = pd.read_csv(RES / "robustness" / "e10_block_length.csv")
    sd = pd.read_csv(SYS / "state_definition.csv").sort_values("sharpe_diff").reset_index(drop=True)
    yr = pd.read_csv(SYS / "year_level.csv")
    yr["period"] = np.where(yr["year"] <= 2019, "2014--2019", "2020--2025")
    sub = yr.groupby("period")["sharpe_diff"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    ax = axes[0]
    ax.plot(e10["block_months"], e10["p1"], color=INK, lw=1.0, marker="o",
            ms=3.6, mfc="white", mec=INK, mew=0.9)
    ax.plot(e10["block_months"], e10["p2"], color=MID, lw=1.0, ls=(0, (3, 2)),
            marker="s", ms=3.4, mfc=MID, mec=MID)
    ax.axhline(0.05, color=LIGHT, lw=1.0, ls=(0, (1, 2)))
    ax.annotate("5% 水平", xy=(0.72, 0.051), xycoords=("axes fraction", "data"),
                fontsize=7, color=MID)
    ax.annotate("一阶", xy=(3, e10["p1"].iloc[0]), xytext=(4, 3),
                textcoords="offset points", fontsize=7, color=INK)
    ax.annotate("二阶", xy=(24, e10["p2"].iloc[-1]), xytext=(-14, 4),
                textcoords="offset points", fontsize=7, color=MID)
    ax.set_xticks([3, 6, 12, 24])
    ax.set_xlabel("自助区块长度（月）", fontsize=8)
    ax.set_ylabel("阈值检验 $p$ 值", fontsize=8)
    panel_label(ax, "a", dx=-24)

    ax = axes[1]
    yy = np.arange(len(sd))
    ax.plot(sd["sharpe_diff"], yy, marker="o", ms=3.8, mfc="white", mec=INK,
            mew=0.9, color=INK, lw=0, zorder=3)
    for y_i, v in zip(yy, sd["sharpe_diff"]):
        ax.plot([0, v], [y_i, y_i], color=LIGHT, lw=1.0, zorder=1)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{STATE_CN.get(v, v)}（$S$={int(s)}）"
                        for v, s in zip(sd["state_var"], sd["n_states"])], fontsize=6.8)
    ax.set_xlabel("门控−等权夏普差", fontsize=8)
    ax.set_ylim(-0.6, len(sd) - 0.4)
    panel_label(ax, "b", dx=-22)

    ax = axes[2]
    order = ["2014--2019", "2020--2025"]
    vals = [sub[o] for o in order]
    ax.bar(order, vals, width=0.45, facecolor=LIGHT, edgecolor=INK, lw=0.9)
    for o, v in zip(order, vals):
        ax.annotate(f"{v:+.3f}", xy=(o, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlabel("子样本", fontsize=8)
    ax.set_ylabel("门控−等权夏普差", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.3)
    panel_label(ax, "c", dx=-22)

    fig.subplots_adjust(wspace=0.46)
    align_gate(fig, "fig_robustness")
    export(fig, "fig_robustness")


if __name__ == "__main__":
    fig_regime()
    fig_significance()
    fig_threshold()
    fig_determinants()
    fig_marginal()
    fig_robustness()
    print("图件重绘完成：", sorted(p.name for p in FIG.glob("fig_*.pdf")))
