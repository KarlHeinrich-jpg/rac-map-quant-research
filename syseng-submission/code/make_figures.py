r"""生成论文图表（中文标注，出版分辨率）。

输出到 ``output/rac-map-theory/figures/``：
- fig_equity.pdf/png      门控组合与基准的累计净值
- fig_states.pdf/png      状态数比较静态（H1）
- fig_gamma.pdf/png       互补性统计量与噪声底线
- fig_turnover.pdf/png    主动暴露与换手率对照（退化问题）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backtest as bt  # noqa: E402
import panel_data as pdmod  # noqa: E402
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
OUT = Path(r"F:\work-environment\output\rac-map-theory\figures")
OUT.mkdir(parents=True, exist_ok=True)

# 中文字体
for fam in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
    if any(fam.lower() in f.lower() for f in matplotlib.font_manager.get_font_names()):
        plt.rcParams["font.sans-serif"] = [fam]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.bbox"] = "tight"

COLORS = {"gate": "#B22222", "ew": "#1F4E79", "static": "#7F7F7F", "expert": "#2E8B57"}


def feature_tensor(panel):
    names = [n for n in EXPERT_FEATURES if n in panel.features]
    return np.nan_to_num(np.stack([panel.features[n] for n in names], axis=-1), nan=0.0), names


def prepare(panel):
    _, names = feature_tensor(panel)
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
    }


def rolling_equity(panel, prep, first_test=2014, last_test=2025, seed=7, cost_bps=10.0):
    """逐测试年份训练门控并拼接测试期净值。"""
    uni, ret = panel.in_universe, panel.ret_next
    years = pd.DatetimeIndex(panel.dates).year.to_numpy()
    series = {"gate": [], "ew": [], "static": []}
    dates = []
    for ty in range(first_test, last_test + 1):
        if ty - 6 < years.min():
            continue
        f = lambda y, s: int(np.searchsorted(years, y, side=s))  # noqa: E731
        tr_lo, tr_hi = f(ty - 6, "left"), f(ty - 2, "right")
        va_lo, va_hi = f(ty - 1, "left"), f(ty - 1, "right")
        te_lo, te_hi = f(ty, "left"), f(ty, "right")
        if tr_hi - tr_lo < 12 or te_hi - te_lo < 1:
            continue
        cfg = GateConfig(seed=seed, epochs=200, objective="active")
        model, _ = train_gate(panel.regime[tr_lo:tr_hi], prep["conf"][tr_lo:tr_hi], prep["risk"][tr_lo:tr_hi],
                              prep["perf"][:, tr_lo:tr_hi].T, prep["expert_w"][:, tr_lo:tr_hi], ret[tr_lo:tr_hi], cfg)
        g = predict_gate(model, panel.regime[te_lo:te_hi], prep["conf"][te_lo:te_hi],
                         prep["risk"][te_lo:te_hi], prep["perf"][:, te_lo:te_hi].T)
        w_gate = np.einsum("tk,ktn->tn", g, prep["expert_w"][:, te_lo:te_hi])
        r_gate = bt.run_backtest(w_gate, ret[te_lo:te_hi], panel.tradable[te_lo:te_hi], uni[te_lo:te_hi], cost_bps=cost_bps)
        w_ew = np.where(uni[te_lo:te_hi] > 0, 1.0, 0.0)
        w_ew = w_ew / np.maximum(w_ew.sum(axis=1, keepdims=True), 1e-12)
        r_ew = bt.run_backtest(w_ew, ret[te_lo:te_hi], panel.tradable[te_lo:te_hi], uni[te_lo:te_hi], cost_bps=cost_bps)
        r_st = bt.run_backtest(prep["expert_w"].mean(axis=0)[te_lo:te_hi], ret[te_lo:te_hi],
                               panel.tradable[te_lo:te_hi], uni[te_lo:te_hi], cost_bps=cost_bps)
        series["gate"].append(r_gate.net)
        series["ew"].append(r_ew.net)
        series["static"].append(r_st.net)
        dates.extend(panel.dates[te_lo:te_hi])
    return {k: np.concatenate(v) for k, v in series.items()}, pd.DatetimeIndex(dates)


def fig_equity(panel, prep):
    net, dates = rolling_equity(panel, prep)
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.12})
    ax = axes[0]
    for key, label, color in (("gate", "门控组合（主动目标）", COLORS["gate"]),
                              ("ew", "等权基准", COLORS["ew"]),
                              ("static", "静态专家合成", COLORS["static"])):
        cum = np.cumprod(1.0 + net[key])
        ax.plot(dates, cum, label=label, color=color, lw=1.6)
    ax.set_ylabel("累计净值（2014 年初 = 1）")
    ax.legend(loc="upper left", frameon=False)
    sp = {k: bt.sharpe_from_series(v) for k, v in net.items()}
    ax.set_title("图 1  滚动样本外累计净值（2014—2025 年，单边成本 10bp，随机种子 7）\n"
                 "连续序列夏普：门控 %.3f，等权 %.3f，静态合成 %.3f"
                 % (sp["gate"], sp["ew"], sp["static"]), fontsize=10)

    ax2 = axes[1]
    rel = np.cumprod(1.0 + net["gate"]) / np.cumprod(1.0 + net["ew"]) - 1.0
    ax2.plot(dates, rel * 100, color=COLORS["gate"], lw=1.4)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("相对等权超额收益（%）")
    ax2.set_xlabel("")
    fig.savefig(OUT / "fig_equity.pdf")
    fig.savefig(OUT / "fig_equity.png")
    plt.close(fig)
    sharpe = {k: bt.sharpe_from_series(v) for k, v in net.items()}
    print("equity 图完成，夏普：", {k: round(v, 3) for k, v in sharpe.items()})


def fig_states():
    path = RESULTS / "hypothesis_main" / "e4a_state_count.csv"
    df = pd.read_csv(path)
    g = df.groupby(["n_states", "test_year"])[["sharpe", "mean_active"]].mean().groupby("n_states").mean()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.bar([str(i) for i in g.index], g["sharpe"], color=["#8FAADC", COLORS["gate"], "#8FAADC"], width=0.55)
    for i, (idx, row) in enumerate(g.iterrows()):
        ax.text(i, row["sharpe"] + 0.008, f"{row['sharpe']:.3f}", ha="center", fontsize=10)
    ax.set_xlabel("制度状态数 $S$")
    ax.set_ylabel("平均夏普比率")
    ax.set_ylim(0, g["sharpe"].max() * 1.18)
    ax.set_title("图 2  状态数的比较静态（H1）：内点最优", fontsize=11)
    fig.savefig(OUT / "fig_states.pdf")
    fig.savefig(OUT / "fig_states.png")
    plt.close(fig)
    print("states 图完成：", g["sharpe"].round(4).to_dict())


def fig_gamma():
    path = RESULTS / "hypothesis_main" / "hypothesis_report.json"
    rep = json.loads(path.read_text(encoding="utf-8"))
    e2 = rep["E2_complementarity"]
    gamma, theta = e2["gamma"], e2["theta_95"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(["$\\hat\\Gamma$（样本估计）", "$\\Theta$（噪声底线）"], [gamma, theta],
           color=[COLORS["gate"], "#7F7F7F"], width=0.5)
    for i, v in enumerate([gamma, theta]):
        ax.text(i, v * 1.02, f"{v:.2e}", ha="center", fontsize=9.5)
    ax.set_ylabel("互补性统计量")
    ax.set_ylim(0, max(gamma, theta) * 1.25)
    ax.set_title("图 3  互补性统计量未超过估计噪声底线（$p=%.2f$）" % e2["p_value"], fontsize=11)
    fig.savefig(OUT / "fig_gamma.pdf")
    fig.savefig(OUT / "fig_gamma.png")
    plt.close(fig)
    print(f"gamma 图完成：gamma={gamma:.3e} theta={theta:.3e} p={e2['p_value']:.3f}")


def fig_turnover():
    df = pd.read_csv(RESULTS / "exp_ashare_active" / "window_metrics.csv")
    my = df.groupby(["model", "test_year"])[["avg_turnover", "mean_active_exposure", "sharpe"]].mean().groupby("model").mean()
    show = ["gate_active", "equal_weight", "static_experts", "expert_low_vol", "gbrt"]
    labels = {"gate_active": "门控组合", "equal_weight": "等权", "static_experts": "静态合成",
              "expert_low_vol": "低波动专家", "gbrt": "梯度提升树"}
    sub = my.loc[[m for m in show if m in my.index]]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    x = np.arange(len(sub))
    ax.bar(x - 0.19, sub["mean_active_exposure"], width=0.36, label="主动暴露", color=COLORS["gate"])
    ax.bar(x + 0.19, sub["avg_turnover"], width=0.36, label="换手率", color="#8FAADC")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in sub.index], fontsize=9.5)
    ax.set_ylabel("均值")
    ax.legend(frameon=False)
    ax.set_title("图 4  主动暴露与换手率：门控确实在动态调整", fontsize=11)
    fig.savefig(OUT / "fig_turnover.pdf")
    fig.savefig(OUT / "fig_turnover.png")
    plt.close(fig)
    print("turnover 图完成：\n", sub.round(4).to_string())


def main() -> None:
    panel = pdmod.build_panel_data()
    prep = prepare(panel)
    fig_equity(panel, prep)
    fig_states()
    fig_gamma()
    fig_turnover()
    print("\n图表已输出到", OUT)


if __name__ == "__main__":
    main()
