# -*- coding: utf-8 -*-
"""《系统工程》投稿版图件：黑白（灰度）中文标注，矢量 PDF + 300dpi PNG。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYS = ROOT / "syseng"
RES = SYS / "results"
FIG = SYS / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for fam in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
    if any(fam.lower() in f.lower() for f in matplotlib.font_manager.get_font_names()):
        plt.rcParams["font.sans-serif"] = [fam]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

G1, G2, G3 = "0.15", "0.55", "0.85"  # 深/中/浅灰


def fig_threshold() -> None:
    """图：互补性统计量与噪声底线的对照（阈值检验）。"""
    ch = json.loads((RES / "marginal_effects.json").read_text(encoding="utf-8"))
    labels = ["一阶渠道\n（收益择时）", "二阶渠道\n（风险配置）"]
    gamma = [ch["gamma1"], ch["gamma2"]]
    theta = [ch["theta1"], ch["theta2"]]
    pvals = [ch["p1"], ch["p2"]]

    x = np.arange(2)
    w = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(x - w / 2, np.array(gamma) * 1e4, w, label=r"互补性统计量 $\hat\Gamma$",
                facecolor="white", edgecolor="black", hatch="//", linewidth=0.9)
    b2 = ax.bar(x + w / 2, np.array(theta) * 1e4, w, label=r"噪声底线 $\Theta$（95% 分位）",
                facecolor="0.72", edgecolor="black", linewidth=0.9)
    for xi, (g, t, p) in zip(x, zip(gamma, theta, pvals)):
        ax.annotate(f"$\\hat\\Gamma-\\Theta={g - t:+.2f}\\times10^{{-4}}$\n$p={p:.3f}$".replace(
                        f"{g - t:+.2f}", f"{(g - t) * 1e4:+.2f}"),
                    xy=(xi, max(g, t) * 1e4), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"统计量（$\times10^{-4}$）")
    ax.set_ylim(0, 9.2)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.savefig(FIG / "fig_threshold.pdf")
    fig.savefig(FIG / "fig_threshold.png")
    plt.close(fig)


def fig_marginal() -> None:
    """图：三个设计维度的边际效应。"""
    cost = pd.read_csv(RES / "marginal_cost_levels.csv")
    states = pd.read_csv(RES / "marginal_states_levels.csv")
    train = pd.read_csv(RES / "marginal_train_levels.csv")
    me = json.loads((RES / "marginal_effects.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    ax = axes[0]
    ax.plot(cost["cost_bps"], cost["sharpe_diff"], marker="o", color=G1,
            mfc="white", mec="black", lw=1.4, ms=5, label="各成本水平均值")
    xs = np.linspace(0, 60, 50)
    b, a = me["cost"]["beta"], me["cost"]["level_0bp"]
    ax.plot(xs, a + b * xs, ls="--", color="black", lw=1.0,
            label=f"线性拟合 $\\beta$={b * 10:.4f}/10bp")
    be = me["cost"]["breakeven_bps"]
    ax.axvline(min(be, 60), color="0.45", lw=0.9, ls=":")
    ax.annotate(f"盈亏平衡≈{be:.0f}bp", xy=(min(be, 58), 0.02), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=9, color="0.25")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("单边交易成本（基点）")
    ax.set_ylabel("门控−等权夏普差")
    ax.set_title("(a) 交易成本", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.6, loc="lower left")

    ax = axes[1]
    sl = me["state_steps"]
    ax.bar(states["n_states"].astype(str), states["sharpe_diff"], width=0.55,
           facecolor="0.72", edgecolor="black", linewidth=0.9)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("制度状态数 $S$")
    ax.set_ylabel("门控−等权夏普差")
    ax.set_title("(b) 制度状态数", fontsize=10.5)
    ax.annotate(f"$S$: 2→3 {sl['d_2to3']:+.3f}\n$S$: 3→4 {sl['d_3to4']:+.3f}",
                xy=(0.03, 0.95), xycoords="axes fraction", va="top", fontsize=9)

    ax = axes[2]
    ax.plot(train["train_years"], train["sharpe_diff"], marker="s", color=G1,
            mfc="white", mec="black", lw=1.4, ms=5)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("训练窗长度（年）")
    ax.set_ylabel("门控−等权夏普差")
    ax.set_title("(c) 训练窗长度", fontsize=10.5)
    tf = me["train"]
    ax.annotate(f"斜率 {tf['beta']:+.4f}/年\n$p$={tf['p']:.3f}", xy=(0.97, 0.95),
                xycoords="axes fraction", va="top", ha="right", fontsize=9)
    ax.set_xticks([3, 5, 8])

    fig.tight_layout()
    fig.savefig(FIG / "fig_marginal.pdf")
    fig.savefig(FIG / "fig_marginal.png")
    plt.close(fig)


def fig_determinants() -> None:
    """图：年度市场波动率与门控相对优势（影响因素）。"""
    yr = pd.read_csv(RES / "year_level.csv")
    det = pd.read_csv(RES / "determinants_year.csv")
    rho = float(det.loc[det["变量"].str.startswith("市场波动率"), "秩相关"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.scatter(yr["mkt_vol_yr"], yr["sharpe_diff"], s=42, facecolors="none",
               edgecolors="black", linewidths=1.1, zorder=3, label="测试年份")
    # 逐个指定标签偏移，避开趋势线与相邻点
    offsets = {2014: (-30, 2), 2016: (-2, 8), 2017: (-8, -13), 2021: (5, -12),
               2020: (5, 5), 2018: (5, 5), 2019: (2, 6), 2022: (5, -12),
               2024: (5, 5), 2025: (5, -12), 2023: (5, -12), 2015: (5, 4)}
    for _, r in yr.iterrows():
        dx, dy = offsets.get(int(r["year"]), (5, 5))
        ax.annotate(int(r["year"]), (r["mkt_vol_yr"], r["sharpe_diff"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8, color="0.25")
    q = pd.qcut(yr["mkt_vol_yr"], 3, labels=["低", "中", "高"])
    gm = yr.groupby(q, observed=False)["sharpe_diff"].mean()
    xs = yr.groupby(q, observed=False)["mkt_vol_yr"].mean()
    ax.plot(xs, gm, marker="D", color="black", lw=1.2, ms=5,
            label="三分组均值（低/中/高波动）")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("年度市场波动率（年化）")
    ax.set_ylabel("门控−等权夏普差")
    ax.set_ylim(yr["sharpe_diff"].min() - 0.05, yr["sharpe_diff"].max() + 0.06)
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2, columnspacing=0.8)
    fig.savefig(FIG / "fig_determinants.pdf")
    fig.savefig(FIG / "fig_determinants.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_threshold()
    fig_marginal()
    fig_determinants()
    print("图件完成：", sorted(p.name for p in FIG.glob("*.pdf")))
