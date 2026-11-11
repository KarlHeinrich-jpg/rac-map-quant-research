# -*- coding: utf-8 -*-
"""状态可预测性检验：logit 与 probit。

问题：门控的价值取决于状态识别，而状态能否用公开信息实时识别决定了策略的可实施性。
本文的渠道分解给出一个可检验含义：
- 若门控价值来自二阶（风险结构）渠道，则「风险结构制度」应当具有可预测性
  （波动率聚集），因为二阶量可以从公开价格序列低成本估计；
- 「收益水平制度」对应一阶渠道，若一阶互补性不可利用，其可预测性应当较弱。

被解释变量（均为下月状态，二值）：
- 风险结构制度：按 6 个月市场波动率三分位划分，取最高一组为 1；
- 收益水平制度：按 12 个月市场收益三分位划分，取最低一组（熊市）为 1；
- 制度切换：下月状态（按 12 个月收益划分）与本月不同为 1。
解释变量均为滞后一期的公开信息。标准误用 Newey-West（HAC，6 阶）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "syseng" / "results"
OUT.mkdir(parents=True, exist_ok=True)

VARS_CN = {
    "mkt_ret_l1": "市场收益（上月）",
    "mkt_ret_3m_l1": "市场收益（近 3 个月）",
    "mkt_vol_6m_l1": "市场波动（近 6 个月）",
    "mkt_drawdown_1y_l1": "市场回撤（近 1 年）",
    "cs_disp_l1": "横截面离散度",
    "up_share_l1": "上涨股票占比",
}


def build() -> pd.DataFrame:
    idx = pd.read_parquet(ROOT / "data" / "index_daily.parquet")
    idx["date"] = pd.to_datetime(idx["ym_period_str"] + "-01")
    d = idx[["date", "mkt_ret", "mkt_ret_3m", "mkt_vol_6m", "mkt_drawdown_1y",
             "cs_disp", "up_share", "mkt_ret_12m"]].copy()
    d = d.sort_values("date").reset_index(drop=True)

    # 状态：收益水平制度（12 个月收益三分位）与风险结构制度（6 个月波动三分位）
    d["state_ret"] = pd.qcut(d["mkt_ret_12m"].rank(method="first"), 3, labels=False)
    d["state_vol"] = pd.qcut(d["mkt_vol_6m"].rank(method="first"), 3, labels=False)

    # 被解释变量：下月状态
    d["y_bear"] = (d["state_ret"].shift(-1) == 0).astype(float)     # 下月为低收益制度
    d["y_highvol"] = (d["state_vol"].shift(-1) == 2).astype(float)  # 下月为高波动制度
    d["y_switch"] = (d["state_ret"].shift(-1) != d["state_ret"]).astype(float)

    # 解释变量：滞后一期
    for c in ["mkt_ret", "mkt_ret_3m", "mkt_vol_6m", "mkt_drawdown_1y", "cs_disp", "up_share"]:
        d[f"{c}_l1"] = d[c].shift(1)

    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    # 标准化解释变量，便于比较系数与计算平均边际效应
    xs = list(VARS_CN)
    d[[f"z_{x}" for x in xs]] = (d[xs] - d[xs].mean()) / d[xs].std(ddof=1)
    return d


def fit(d: pd.DataFrame, y: str, zs: list[str], model: str) -> dict:
    X = sm.add_constant(d[zs])
    yy = d[y]
    cls = sm.Logit if model == "logit" else sm.Probit
    res = cls(yy, X).fit(disp=False, cov_type="HAC", cov_kwds={"maxlags": 6})
    # 平均边际效应：AME_j = mean( beta_j * f(X'beta) )，logit 取 Λ(1-Λ)，probit 取 φ
    eta = np.asarray(X @ res.params, dtype=float)
    if model == "logit":
        f = (1.0 / (1.0 + np.exp(-eta)))
        dens = f * (1 - f)
    else:
        dens = np.exp(-0.5 * eta ** 2) / np.sqrt(2 * np.pi)
    ame, ame_p = {}, {}
    for z in zs:
        ame[z] = float(np.mean(res.params[z] * dens))
        ame_p[z] = float(res.pvalues[z])
    pred = (res.predict(X) > yy.mean()).astype(float)
    return {
        "model": model, "y": y, "n": int(res.nobs),
        "pseudo_r2": float(res.prsquared),
        "hit_rate": float((pred == yy).mean()),
        "base_rate": float(yy.mean()),
        "llf": float(res.llf),
        "params": {z: float(b) for z, b in res.params.items()},
        "pvalues": {z: float(p) for z, p in res.pvalues.items()},
        "ame": ame, "ame_p": ame_p,
    }


def rolling_hit(d: pd.DataFrame, y: str, zs: list[str], model: str,
                min_train: int = 60) -> float:
    """滚动样本外命中率：扩展窗，每月用历史数据估计、预测下一个月。"""
    hit, tot = 0, 0
    for t in range(min_train, len(d)):
        tr = d.iloc[:t]
        te = d.iloc[[t]]
        try:
            X = sm.add_constant(tr[zs])
            cls = sm.Logit if model == "logit" else sm.Probit
            res = cls(tr[y], X).fit(disp=False)
            p = float(res.predict(sm.add_constant(te[zs], has_constant="add")).iloc[0])
        except Exception:  # noqa: BLE001
            continue
        pred = 1.0 if p > 0.5 else 0.0
        hit += int(pred == float(te[y].iloc[0]))
        tot += 1
    return hit / tot if tot else np.nan


def main() -> None:
    d = build()
    zs = [f"z_{x}" for x in VARS_CN]
    specs = [("y_highvol", "风险结构制度（高波动）"), ("y_bear", "收益水平制度（熊市）"),
             ("y_switch", "制度切换（收益水平划分）")]
    rows, detail = [], {}
    for y, label in specs:
        for model in ("logit", "probit"):
            r = fit(d, y, zs, model)
            r["label"] = label
            r["oos_hit"] = rolling_hit(d, y, zs, model)
            detail[f"{y}_{model}"] = r
            rows.append({
                "被解释变量": label, "模型": model, "样本数": r["n"],
                "基准率": round(r["base_rate"], 3),
                "伪 R²": round(r["pseudo_r2"], 3),
                "样本内命中率": round(r["hit_rate"], 3),
                "样本外命中率": round(r["oos_hit"], 3),
            })

    pd.DataFrame(rows).to_csv(OUT / "logit_probit_summary.csv", index=False,
                              encoding="utf-8-sig")
    (OUT / "logit_probit_detail.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 78)
    print("状态可预测性检验（被解释变量：下月状态，解释变量滞后一期，HAC(6) 标准误）")
    for r in rows:
        print(f"  {r['被解释变量']:<22s} {r['模型']:<7s} n={r['样本数']:<4d} "
              f"基准率={r['基准率']:.2f} 伪R²={r['伪 R²']:.3f} "
              f"命中率={r['样本内命中率']:.3f}（样本外 {r['样本外命中率']:.3f}）")
    print()
    for key in ("y_highvol_logit", "y_bear_logit"):
        r = detail[key]
        print(f"[{r['label']}] logit 平均边际效应（每 1 个标准差）：")
        for z, cn in VARS_CN.items():
            a = r["ame"].get(f"z_{z}", np.nan)
            p = r["ame_p"].get(f"z_{z}", np.nan)
            star = "**" if p < 0.05 else ("*" if p < 0.1 else "  ")
            print(f"    {cn:<18s} {a:+.3f}  (p={p:.3f}) {star}")
        print()
    print("=" * 78)


if __name__ == "__main__":
    main()
