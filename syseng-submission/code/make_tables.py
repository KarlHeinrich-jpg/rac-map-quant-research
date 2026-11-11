r"""由实验结果生成论文表格（CSV 与 LaTeX 片段）。

输出到 ``output/rac-map-theory/tables/``：
- ``table2_main.csv`` / ``table2_main.tex``       主结果表
- ``table3_significance.csv`` / ``.tex``          显著性检验表
- ``table4_gamma.csv`` / ``.tex``                 互补性阈值表
- ``table5_oracle.csv`` / ``.tex``                Oracle 与安慰剂表
- ``table6_states.csv`` / ``.tex``                状态数比较静态表
- ``table7_experts.csv`` / ``.tex``               专家定义表
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(r"F:\work-environment\output\rac-map-theory\tables")
OUT.mkdir(parents=True, exist_ok=True)

CN = {
    "expert_low_vol": "低波动专家", "risk_parity": "风险平价",
    "gate_active": "门控组合（主动目标）", "expert_reversal": "反转专家",
    "gbrt": "梯度提升树", "ridge": "岭回归", "equal_weight": "等权基准",
    "static_experts": "静态专家合成", "expert_low_turnover": "低换手专家",
    "expert_momentum": "动量专家", "expert_low_beta": "低 beta 专家",
}


def write_tex(rows: list[str], path: Path, caption: str, label: str, note: str, colspec: str) -> None:
    body = "\n".join(rows)
    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\begin{threeparttable}\n\\small\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        "\\begin{tablenotes}[flushleft]\\footnotesize\n"
        f"\\item {note}\n"
        "\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n"
    )
    path.write_text(tex, encoding="utf-8")


def table_main() -> None:
    df = pd.read_csv(RESULTS / "exp_ashare_active" / "window_metrics.csv")
    cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "cvar_5", "avg_turnover", "mean_active_exposure"]
    my = df.groupby(["model", "test_year"])[cols].mean().groupby("model").mean().sort_values("sharpe", ascending=False)
    my["cn"] = [CN.get(i, i) for i in my.index]
    my.round(4).to_csv(OUT / "table2_main.csv", encoding="utf-8-sig")

    rows = ["模型 & CAGR & 波动率 & 夏普 & Sortino & 最大回撤 & 换手率 & 主动暴露 \\\\", "\\midrule"]
    for idx, r in my.iterrows():
        name = r["cn"]
        if idx in {"gate_active", "equal_weight"}:
            name = f"\\textbf{{{name}}}"
        sharp = f"\\textbf{{{r['sharpe']:.3f}}}" if idx == "expert_low_vol" else f"{r['sharpe']:.3f}"
        rows.append(
            f"{name} & {r['cagr']:.4f} & {r['vol']:.4f} & {sharp} & {r['sortino']:.3f} & "
            f"${r['max_drawdown']:.3f}$ & {r['avg_turnover']:.3f} & {r['mean_active_exposure']:.3f} \\\\"
        )
    write_tex(rows, OUT / "table2_main.tex",
              "滚动样本外表现（2014--2025 年，年度内先对种子取平均）", "tab:main",
              "注：主动暴露为组合权重相对等权的半 $L_1$ 距离均值。交易成本为单边 10 个基点。"
              "学习模型在每年使用三个随机种子并先取平均。",
              "lrrrrrrr")
    print("表 2 完成\n", my.round(4).to_string())


def table_significance() -> None:
    df = pd.read_csv(RESULTS / "exp_ashare_active" / "significance_table.csv")
    sh = df[df.metric == "sharpe"].copy()
    sh["diff"] = sh.model.map(lambda m: None)
    sh = sh.sort_values("diff_vs_benchmark", ascending=False)
    sh["cn"] = [CN.get(i, i) for i in sh.model]
    sh.round(4).to_csv(OUT / "table3_significance.csv", index=False, encoding="utf-8-sig")

    rows = ["模型 & 差值 & 95\\% 下限 & 95\\% 上限 & 区间含零 & Holm 校正 $p$ \\\\", "\\midrule"]
    for _, r in sh.iterrows():
        name = f"\\textbf{{{r['cn']}}}" if r["model"] == "gate_active" else r["cn"]
        lo = f"${r['ci_lo']:+.3f}$" if r["ci_lo"] < 0 else f"$+{r['ci_lo']:.3f}$"
        hi = f"${r['ci_hi']:+.3f}$" if r["ci_hi"] < 0 else f"$+{r['ci_hi']:.3f}$"
        contains = "\\textbf{是}" if r["crosses_zero"] else "否"
        rows.append(f"{name} & ${r['diff_vs_benchmark']:+.3f}$ & {lo} & {hi} & {contains} & {r['holm_p']:.3f} \\\\")
    write_tex(rows, OUT / "table3_significance.tex",
              "相对等权基准的夏普差值检验（配对单位：测试年份，$n=12$）", "tab:signif",
              "注：区间由循环区块自助法（块长 2 年，10\\,000 次重抽样）构造；$p$ 值为 Wilcoxon "
              "符号秩检验并作 Holm 多重比较校正。",
              "lrrrrc")
    print("\n表 3 完成")
    print(sh[["cn", "diff_vs_benchmark", "ci_lo", "ci_hi", "crosses_zero", "holm_p"]].round(4).to_string(index=False))


def table_gamma() -> None:
    rep = json.loads((RESULTS / "hypothesis_main" / "hypothesis_report.json").read_text(encoding="utf-8"))
    e2 = rep["E2_complementarity"]
    df = pd.DataFrame([
        {"样本": "中国 A 股（$N\\approx438$）", "gamma": e2["gamma"], "theta": e2["theta_95"],
         "months": e2["n_months"], "p": e2["p_value"]},
        {"样本": "美国 ETF 10 只（$N=10$）", "gamma": None, "theta": None, "months": 0, "p": None},
    ])
    df.to_csv(OUT / "table4_gamma.csv", index=False, encoding="utf-8-sig")
    rows = [
        "样本 & $\\hat\\Gamma$ & $\\Theta$（95\\% 分位） & 有效月份 & $p$ 值 \\\\", "\\midrule",
        f"中国 A 股（$N\\approx438$） & ${e2['gamma']:.1e}$ & ${e2['theta_95']:.1e}$ & {e2['n_months']} & {e2['p_value']:.3f} \\\\",
        "美国 ETF 10 只（$N=10$） & —— & —— & 不可识别 & —— \\\\",
    ]
    write_tex(rows, OUT / "table4_gamma.tex",
              "信号—状态互补性 $\\hat\\Gamma$ 与估计噪声底线 $\\Theta$", "tab:gamma",
              "注：$\\Theta$ 由平稳区块自助法（500 次，块长 6 个月）在“无真实互补性”原假设下校准，"
              "取 $\\hat\\Gamma^*$ 分布的 95\\% 分位数。窄横截面样本因每月证券数不足 20 只而不可识别。",
              "lrrrr")
    print("\n表 4 完成：gamma=%.3e theta=%.3e p=%.3f" % (e2["gamma"], e2["theta_95"], e2["p_value"]))


def table_oracle() -> None:
    e3 = pd.read_csv(RESULTS / "hypothesis_main" / "e3_oracle.csv")
    g = e3.groupby("model")[["sharpe", "cagr", "avg_turnover"]].mean()
    names = {"equal_weight": "等权基准", "static_experts": "静态专家合成",
             "oracle_gate": "Oracle 门控（用真实条件期望收益）"}
    rep = json.loads((RESULTS / "hypothesis_main" / "hypothesis_report.json").read_text(encoding="utf-8"))
    pl = rep["E5_placebo"]
    g.round(4).to_csv(OUT / "table5_oracle.csv", encoding="utf-8-sig")
    rows = ["检验 & 夏普 & 年化收益 & 换手率 \\\\", "\\midrule"]
    for idx in ["equal_weight", "static_experts", "oracle_gate"]:
        if idx not in g.index:
            continue
        r = g.loc[idx]
        rows.append(f"{names[idx]} & {r['sharpe']:.3f} & {r['cagr']*100:.1f}\\% & {r['avg_turnover']:.3f} \\\\")
    rows += [
        "\\midrule",
        "\\multicolumn{4}{l}{\\textit{安慰剂检验（打乱收益—状态关联，200 次重抽样）}} \\\\",
        f"门控主动收益 & \\multicolumn{{3}}{{r}}{{${pl['mean_active_placebo']*100:+.3f}\\%$，"
        f"95\\% 区间 $[{pl['q05']*100:+.3f}\\%,{pl['q95']*100:+.3f}\\%]$}} \\\\",
    ]
    write_tex(rows, OUT / "table5_oracle.tex",
              "Oracle 门控（一阶互补性可利用上界）与安慰剂检验", "tab:oracle",
              "注：Oracle 门控按训练窗内各状态的资产平均持有期收益为专家加权，使用真实条件期望收益，"
              "因此不可实现，仅用于构造一阶互补性的可利用上界。",
              "lrrr")
    print("\n表 5 完成\n", g.round(4).to_string())
    print(f"  安慰剂：{pl['mean_active_placebo']*100:+.4f}%  区间 [{pl['q05']*100:+.4f}%, {pl['q95']*100:+.4f}%]")


def table_states() -> None:
    df = pd.read_csv(RESULTS / "hypothesis_main" / "e4a_state_count.csv")
    g = df.groupby(["n_states", "test_year"])[["sharpe", "mean_active"]].mean().groupby("n_states").mean()
    g.round(4).to_csv(OUT / "table6_states.csv", encoding="utf-8-sig")
    rows = ["制度状态数 $S$ & 平均夏普 & 年化主动收益 \\\\", "\\midrule"]
    for idx, r in g.iterrows():
        sharp = f"\\textbf{{{r['sharpe']:.3f}}}" if r["sharpe"] == g["sharpe"].max() else f"{r['sharpe']:.3f}"
        rows.append(f"{idx} & {sharp} & {r['mean_active']*100:.2f}\\% \\\\")
    write_tex(rows, OUT / "table6_states.tex",
              "状态数的比较静态（H1）", "tab:states",
              "注：状态按 12 个月市场收益的分位数划分，并以 one-hot 形式并入门控输入。"
              "数值为 12 个测试年份、2 个随机种子的平均。",
              "lrr")
    print("\n表 6 完成\n", g.round(4).to_string())


def table_experts() -> None:
    rows = [
        "专家 & 打分 & 经济含义 \\\\", "\\midrule",
        "动量 & $0.7\\,\\mathrm{mom}_{12-1}+0.3\\,\\mathrm{mom}_{12}$ & 中期价格趋势 \\\\",
        "反转 & $-1.0\\,\\mathrm{mom}_{1}$ & 短期过度反应修正 \\\\",
        "低波动 & $-1.0\\,\\mathrm{vol}_{60}$ & 低风险异象 \\\\",
        "低换手 & $0.4\\,\\mathrm{mom}_{12-1}-0.6\\,\\mathrm{turnover}_{20}$ & 低流动性溢价 \\\\",
        "低 beta & $0.3\\,\\mathrm{mom}_{12-1}-1.0\\,\\beta_{120}$ & 系统性风险规避 \\\\",
    ]
    write_tex(rows, OUT / "table7_experts.tex", "专家信号定义", "tab:expert-def",
              "注：全部特征在每月横截面做 1\\%--99\\% 缩尾后标准化。$\\mathrm{turnover}_{20}$ 为异常成交量"
              "（20 日均量除以 250 日均量），因公开行情接口不提供换手率字段，用成交量口径替代。",
              "lll")
    (OUT / "table7_experts.csv").write_text(
        "专家,打分,经济含义\n动量,0.7*mom12-1+0.3*mom12,中期价格趋势\n反转,-1.0*mom1,短期过度反应修正\n"
        "低波动,-1.0*vol60,低风险异象\n低换手,0.4*mom12-1-0.6*turnover20,低流动性溢价\n"
        "低beta,0.3*mom12-1-1.0*beta120,系统性风险规避\n", encoding="utf-8-sig")
    print("\n表 7 完成")


if __name__ == "__main__":
    table_experts()
    table_main()
    table_significance()
    table_gamma()
    table_oracle()
    table_states()
    print("\n全部表格已输出到", OUT)
