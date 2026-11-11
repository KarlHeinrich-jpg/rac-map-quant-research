# -*- coding: utf-8 -*-
"""《系统工程》投稿版新增图件（灰度、中文标注、矢量 PDF）。

依据 econ-table-figure-design 的图件规则：
- 黑白可读：同一图中用线型/标记形状区分，不依赖颜色；
- 双轴标注变量与单位；图注说明样本、口径与读图方式；
- 标题说明"图展示什么"，不写作者动作。

输出：
- fig_regime.pdf      制度状态与市场环境（2006-01--2026-07）
- fig_significance.pdf 各模型相对等权基准的夏普差值（点估计与 95% 区间）
- fig_robustness.pdf  稳健性三面板（自助区块长度、制度定义、子样本）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SYS = ROOT / "syseng" / "results"
FIG = ROOT / "syseng" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for fam in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
    if any(fam.lower() in f.lower() for f in matplotlib.font_manager.get_font_names()):
        plt.rcParams["font.sans-serif"] = [fam]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

MODEL_CN = {
    "expert_low_vol": "低波动专家",
    "risk_parity": "风险平价",
    "gate_active": "门控组合",
    "expert_reversal": "反转专家",
    "gbrt": "梯度提升树",
    "ridge": "岭回归",
    "static_experts": "静态专家合成",
    "expert_low_turnover": "低换手专家",
    "expert_momentum": "动量专家",
    "expert_low_beta": "低 beta 专家",
}
STATE_CN = {
    "mkt_ret_12m": "12 个月市场收益",
    "mkt_vol_6m": "6 个月市场波动",
    "mkt_drawdown_1y": "1 年市场回撤",
    "cs_disp": "横截面离散度",
}


def fig_regime() -> None:
    """图 1：制度状态区间、市场累计走势与波动率。"""
    idx = pd.read_parquet(ROOT / "data" / "index_daily.parquet")
    d = idx[["ym_period_str", "mkt_ret", "mkt_ret_12m", "mkt_vol_6m"]].copy()
    d["date"] = pd.to_datetime(d["ym_period_str"] + "-01")
    d = d.dropna(subset=["mkt_ret_12m"]).reset_index(drop=True)
    # 与主结果一致：12 个月市场收益的三分位制度
    d["state"] = pd.qcut(d["mkt_ret_12m"].rank(method="first"), 3, labels=False)
    d["cum"] = (1 + d["mkt_ret"]).cumprod()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12})

    shades = {0: "0.90", 1: "0.74", 2: "0.55"}
    start = None
    for i, st in enumerate(d["state"]):
        if start is None:
            start, cur = i, st
        elif st != cur or i == len(d) - 1:
            end = d["date"].iloc[i]
            ax1.axvspan(d["date"].iloc[start], end, color=shades[cur], lw=0, zorder=0)
            ax2.axvspan(d["date"].iloc[start], end, color=shades[cur], lw=0, zorder=0)
            start, cur = i, st

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=shades[s], edgecolor="0.6", lw=0.6,
                     label=f"制度 {s + 1}（{lab}）")
               for s, lab in [(0, "低收益"), (1, "中收益"), (2, "高收益")]]
    ax1.legend(handles=handles, frameon=False, fontsize=9, ncol=3,
               loc="upper left", bbox_to_anchor=(0.0, 1.02))

    ax1.plot(d["date"], d["cum"], color="black", lw=1.3, zorder=3)
    ax1.axvline(pd.Timestamp("2014-01-01"), color="black", ls="--", lw=1.0, zorder=4)
    ax1.annotate("滚动测试窗起点（2014 年 1 月）", xy=(pd.Timestamp("2014-01-01"), 0.06),
                 xycoords=("data", "axes fraction"), xytext=(6, 0),
                 textcoords="offset points", fontsize=9, color="0.15")
    ax1.set_ylabel("市场累计净值（2006 年 1 月 = 1）")
    ax1.set_ylim(0, d["cum"].max() * 1.16)

    ax2.plot(d["date"], d["mkt_vol_6m"], color="black", lw=1.2, zorder=3)
    ax2.axvline(pd.Timestamp("2014-01-01"), color="black", ls="--", lw=1.0, zorder=4)
    ax2.set_ylabel("6 个月波动率（年化）")
    ax2.set_xlabel("年份")

    fig.savefig(FIG / "fig_regime.pdf")
    fig.savefig(FIG / "fig_regime.png")
    plt.close(fig)


def fig_significance() -> None:
    """图 2：各模型相对等权基准的夏普差值（点估计与 95% 区块自助区间）。"""
    sig = pd.read_csv(RES / "exp_ashare_active" / "significance_table.csv")
    d = sig[sig["metric"] == "sharpe"].copy()
    d["name"] = d["model"].map(MODEL_CN).fillna(d["model"])
    d = d.sort_values("diff_vs_benchmark").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    y = np.arange(len(d))
    focal = d["model"] == "gate_active"
    for yi, (_, r), is_f in zip(y, d.iterrows(), focal):
        color = "black" if is_f else "0.35"
        lw = 1.8 if is_f else 1.2
        ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=color, lw=lw, zorder=2)
        mark = "D" if is_f else "o"
        ax.plot(r["diff_vs_benchmark"], yi, marker=mark, ms=6.5 if is_f else 5,
                mfc="black" if is_f else "white", mec=color, mew=1.2, zorder=3)
    ax.axvline(0, color="0.3", lw=1.0, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["name"])
    ax.set_xlabel("相对等权基准的夏普差值")
    ax.set_ylim(-0.8, len(d) - 0.2)
    gate = d[d["model"] == "gate_active"].iloc[0]
    gate_y = float(np.where(d["model"] == "gate_active")[0][0])
    ax.annotate(f"门控组合：差值 {gate['diff_vs_benchmark']:+.3f}\n"
                f"95% 区间 [{gate['ci_lo']:.3f}, {gate['ci_hi']:.3f}]\n"
                f"Holm $p$={gate['holm_p']:.3f}",
                xy=(gate["diff_vs_benchmark"], gate_y),
                xytext=(0.03, 0.30), textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8,
                                connectionstyle="arc3,rad=-0.15"))
    fig.savefig(FIG / "fig_significance.pdf")
    fig.savefig(FIG / "fig_significance.png")
    plt.close(fig)


def fig_robustness() -> None:
    """图 3：稳健性三面板（区块长度 p 值、制度定义、子样本）。"""
    e10 = pd.read_csv(RES / "robustness" / "e10_block_length.csv")
    sd = pd.read_csv(SYS / "state_definition.csv")
    yr = pd.read_csv(SYS / "year_level.csv")
    yr["period"] = np.where(yr["year"] <= 2019, "2014--2019", "2020--2025")
    sub = yr.groupby("period")["sharpe_diff"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))

    ax = axes[0]
    ax.plot(e10["block_months"], e10["p1"], marker="o", mfc="white", mec="black",
            color="black", lw=1.3, ms=5, label="一阶渠道")
    ax.plot(e10["block_months"], e10["p2"], marker="s", mfc="black", mec="black",
            color="0.45", lw=1.3, ms=5, ls="--", label="二阶渠道")
    ax.axhline(0.05, color="0.2", lw=0.9, ls=":")
    ax.annotate("5% 水平", xy=(0.62, 0.054), xycoords=("axes fraction", "data"),
                fontsize=8.5, color="0.25")
    ax.set_xlabel("自助区块长度（月）")
    ax.set_ylabel("阈值检验 $p$ 值")
    ax.set_title("(a) 推断方法", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks([3, 6, 12, 24])

    ax = axes[1]
    sd = sd.sort_values("sharpe_diff").reset_index(drop=True)
    lab = [f"{STATE_CN.get(v, v)}（$S$={int(s)}）" for v, s in zip(sd["state_var"], sd["n_states"])]
    yy = np.arange(len(sd))
    ax.plot(sd["sharpe_diff"], yy, marker="o", ms=5, mfc="white", mec="black",
            color="black", lw=1.0, ls="none")
    for y_i, v in zip(yy, sd["sharpe_diff"]):
        ax.plot([0, v], [y_i, y_i], color="0.6", lw=0.8, zorder=0)
    ax.axvline(0, color="0.3", lw=1.0)
    ax.set_yticks(yy)
    ax.set_yticklabels(lab, fontsize=8.6)
    ax.set_xlabel("相对等权基准的夏普差值")
    ax.set_title("(b) 制度定义", fontsize=10.5)

    ax = axes[2]
    order = ["2014--2019", "2020--2025"]
    vals = [sub[o] for o in order]
    ax.bar(order, vals, width=0.5, facecolor="0.72", edgecolor="black", lw=0.9)
    for o, v in zip(order, vals):
        ax.annotate(f"{v:+.3f}", xy=(o, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_ylabel("相对等权基准的夏普差值")
    ax.set_xlabel("子样本")
    ax.set_title("(c) 时间稳定性", fontsize=10.5)
    ax.set_ylim(0, max(vals) * 1.28)

    fig.tight_layout()
    fig.savefig(FIG / "fig_robustness.pdf")
    fig.savefig(FIG / "fig_robustness.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_regime()
    fig_significance()
    fig_robustness()
    print("新增图件：", sorted(p.name for p in FIG.glob("fig_*.pdf")))
