r"""命题 5--9 的实证实现：两条渠道分解、制度持续性、最优状态数。

输出到 ``results/theory_main/``：
- ``channel_decomposition.json``  两渠道的 \(\Gamma^{(1)},\Gamma^{(2)}\) 与噪声底线
- ``transition.json``             转移矩阵、持续性乘数与半衰期
- ``optimal_states.json``         容量约束下的 \(\hat S^\ast\) 曲线
- ``double_sort.json``            双重排序的渠道区分检验
- ``summary.txt``                 可读摘要
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

import complementarity as cp  # noqa: E402
import panel_data as pdmod  # noqa: E402
import theory_core as tc  # noqa: E402
from model import (  # noqa: E402
    EXPERT_FEATURES,
    expert_scores,
    expert_weights,
)

RESULTS = ROOT / "results"


def expert_return_matrix(panel: pdmod.PanelData) -> np.ndarray:
    """(T, K) 的各专家组合持有期收益。"""
    names = [n for n in EXPERT_FEATURES if n in panel.features]
    scores = expert_scores({n: panel.features[n] for n in names})
    ew = expert_weights(scores, panel.in_universe)
    return np.einsum("ktn,tn->tk", ew, np.nan_to_num(panel.ret_next))


def states_for(panel: pdmod.PanelData, n_states: int, var: str = "mkt_ret_12m") -> np.ndarray:
    idx = panel.regime_names.index(var) if var in panel.regime_names else 0
    return cp.assign_states(panel.regime[:, idx], n_states)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--var", default="mkt_ret_12m")
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--risk-aversion", type=float, default=5.0)
    parser.add_argument("--gate-params", type=int, default=200)
    parser.add_argument("--out", default="theory_main")
    args = parser.parse_args()

    out_dir = RESULTS / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pdmod.build_panel_data()
    er = expert_return_matrix(panel)
    print(f"panel: T={panel.n_periods} N={panel.n_assets}; expert returns {er.shape}")

    lines: list[str] = []

    # ---------------- 命题 5：两条渠道分解 ----------------
    states = states_for(panel, args.n_states, args.var)
    stats = tc.RegimeStats.build(states, er)
    dec = tc.channel_decomposition(stats, risk_aversion=args.risk_aversion,
                                  n_boot=args.n_boot, block=args.block)
    (out_dir / "channel_decomposition.json").write_text(
        json.dumps(dec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines.append("=" * 74)
    lines.append("命题 5：门控价值的两渠道分解（S=%d，状态变量=%s）" % (args.n_states, args.var))
    lines.append("=" * 74)
    lines.append(f"  制度样本数            : {dec['n_obs_by_state']}")
    lines.append(f"  一阶（收益择时）Γ(1)  : {dec['gamma1']:.6e}   噪声底线 Θ(1) = {dec['theta1_95']:.6e}   p = {dec['p1']:.3f}")
    lines.append(f"  二阶（波动择时）Γ(2)  : {dec['gamma2']:.6e}   噪声底线 Θ(2) = {dec['theta2_95']:.6e}   p = {dec['p2']:.3f}")
    lines.append(f"  效用贡献 ΔU(1)        : {dec['deltaU1']:.6e}   （λ = {args.risk_aversion}）")
    lines.append(f"  效用贡献 ΔU(2)        : {dec['deltaU2']:.6e}")
    lines.append(f"  一阶是否越过阈值      : {'是' if dec['gamma1'] > dec['theta1_95'] else '否'}")
    lines.append(f"  二阶是否越过阈值      : {'是' if dec['gamma2'] > dec['theta2_95'] else '否'}")
    lines.append("")

    # ---------------- 命题 7：制度持续性与动态乘数 ----------------
    P = tc.transition_matrix(states, stats.state_ids)
    pers = tc.persistence_multiplier(P, discount=args.discount)
    pg = tc.persistence_gain(stats, P, discount=args.discount, risk_aversion=args.risk_aversion)
    (out_dir / "transition.json").write_text(
        json.dumps({"P": P.tolist(), **{k: v for k, v in pers.items() if k != "P"},
                    "persistence_gain": pg}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines.append("=" * 74)
    lines.append("命题 7：制度持续性与动态乘数")
    lines.append("=" * 74)
    lines.append("  转移矩阵 P（行=当期制度，列=下一期制度）：")
    for i, row in enumerate(P):
        lines.append("    " + "  ".join(f"{v:.3f}" for v in row))
    lines.append(f"  对角线（持续概率）    : {np.round(pers['diag'], 3).tolist()}")
    lines.append(f"  半衰期（月）          : {np.round(pers['half_life'], 2).tolist()}")
    lines.append(f"  平均半衰期            : {pers['mean_half_life']:.2f} 月")
    lines.append(f"  平稳分布              : {np.round(pers['stationary'], 3).tolist()}")
    lines.append(f"  持续性乘数 M_s        : {np.round(pers['multiplier'], 3).tolist()}")
    lines.append(f"  Corr(M_s, Γ(1)_s)     : {pg.get('corr_multiplier_gamma1'):.3f}")
    lines.append(f"  持续性加权 Γ(1)       : {pg.get('weighted_gamma1'):.6e}（未加权 {pg.get('unweighted_gamma1'):.6e}）")
    lines.append(f"  放大倍数              : {pg.get('amplification'):.3f}")
    lines.append("")

    # ---------------- 命题 8：容量约束与最优状态数 ----------------
    stats_by_S = {S: tc.RegimeStats.build(states_for(panel, S, args.var), er) for S in (2, 3, 4, 5, 6)}
    opt = tc.optimal_states(stats_by_S, n_gate_params=args.gate_params,
                            risk_aversion=args.risk_aversion)
    (out_dir / "optimal_states.json").write_text(
        json.dumps(opt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines.append("=" * 74)
    lines.append("命题 8：容量约束下的最优状态数")
    lines.append("=" * 74)
    lines.append("   S    Γ(1)        Θ(1)        容量项      净价值 Δ(S)")
    for r in opt["curve"]:
        lines.append(f"  {r['S']}   {r['gamma1']:.3e}  {r['theta1']:.3e}  {r['capacity']:.3e}  {r['delta']:+.3e}")
    lines.append(f"  → 理论最优状态数 Ŝ* = {opt['S_star']}（门控参数 {args.gate_params} 个，λ={args.risk_aversion}）")
    lines.append(f"  → 实证选定的状态数   = 3（验证集夏普最高）")
    lines.append("")

    # ---------------- 渠道识别检验（一阶 vs 二阶） ----------------
    ds = tc.channel_identification_test(stats, n_perm=1000, block=args.block)
    (out_dir / "channel_identification.json").write_text(
        json.dumps(ds, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines.append("=" * 74)
    lines.append("渠道识别检验：一阶（条件均值）vs 二阶（条件波动）")
    lines.append("=" * 74)
    lines.append(f"  一阶变异系数 CV(1) 观测 = {ds['cv1_observed']:.6f}，置换零分布均值 = {ds['cv1_null_mean']:.6f}，p = {ds['p1']:.3f}")
    lines.append(f"  二阶变异系数 CV(2) 观测 = {ds['cv2_observed']:.6f}，置换零分布均值 = {ds['cv2_null_mean']:.6f}，p = {ds['p2']:.3f}")
    lines.append(f"  区块长度 = {ds['block']} 月，置换次数 = {ds['n_perm']}")
    lines.append("")

    # ---------------- 与模型的对照 ----------------
    lines.append("=" * 74)
    lines.append("理论预测与实证结果的对照")
    lines.append("=" * 74)
    verdict1 = "否" if dec["gamma1"] <= dec["theta1_95"] else "是"
    verdict2 = "是" if dec["gamma2"] > dec["theta2_95"] else "否"
    lines.append(f"  命题 2（阈值）：Γ(1) 越过 Θ(1)？            {verdict1}")
    lines.append(f"  命题 4（渠道）：Γ(2) 越过 Θ(2)（更易估计）？ {verdict2}")
    lines.append(f"  命题 7（持续性）：平均半衰期 {pers['mean_half_life']:.2f} 月，放大倍数 {pg.get('amplification'):.2f}")
    lines.append(f"  命题 8（容量）：Ŝ* = {opt['S_star']}，实证 = 3")
    lines.append("")

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
