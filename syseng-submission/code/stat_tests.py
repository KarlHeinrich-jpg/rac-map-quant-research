r"""统计推断：以测试年份为配对单位的门控—基准比较。

推断设计
--------
- 重复单位取**测试年份**（12 个年度窗口），学习模型先在年内对随机种子取均值，
  避免把同一市场年份伪重复计为多个观测；
- 区间用**循环区块自助法**（block bootstrap，块长 2 年）构造，保留年度间的
  序列相关；
- 多模型比较用 Holm 校正，整体排序差异用 Friedman 检验；
- 报告经济量级（换手率、成本后收益）而不只报告 p 值。
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def circular_block_bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 10000,
    block: int = 2,
    seed: int = 20240101,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """循环区块自助法的均值区间。返回 (点估计, 下限, 上限)。"""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    T = x.size
    n_blocks = int(np.ceil(T / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(np.arange(s, s + block) % T) for s in starts])[:T]
        means[b] = x[idx].mean()
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm 逐步校正。"""
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)
        out[k] = running
    for k, v in pvals.items():
        if not np.isfinite(v):
            out[k] = np.nan
    return out


def friedman_test(matrix: np.ndarray) -> tuple[float, float]:
    """Friedman 检验：行=年份，列=模型。返回 (统计量, p 值)。"""
    m = np.asarray(matrix, dtype=float)
    m = m[np.isfinite(m).all(axis=1)]
    if m.shape[0] < 3 or m.shape[1] < 3:
        return np.nan, np.nan
    stat, p = stats.friedmanchisquare(*[m[:, j] for j in range(m.shape[1])])
    return float(stat), float(p)


def compute(model_year: pd.DataFrame, metric: str, base: str) -> dict:
    """计算各模型相对基准的配对差异与区间。"""
    pivot = model_year.pivot_table(index="test_year", columns="model", values=metric, aggfunc="mean")
    if base not in pivot.columns:
        return {"error": f"基准 {base} 不在结果中", "models": list(pivot.columns)}
    out: dict = {"metric": metric, "benchmark": base, "n_years": int(pivot.shape[0]), "pairs": {}}
    pvals: dict[str, float] = {}
    for col in pivot.columns:
        if col == base:
            continue
        diff = (pivot[col] - pivot[base]).to_numpy(dtype=float)
        diff = diff[np.isfinite(diff)]
        if diff.size < 3:
            continue
        point, lo, hi = circular_block_bootstrap_ci(diff)
        try:
            w = stats.wilcoxon(diff, zero_method="wilcox")
            p_raw = float(w.pvalue)
        except ValueError:
            p_raw = np.nan
        out["pairs"][col] = {
            "diff": point,
            "ci_lo": lo,
            "ci_hi": hi,
            "crosses_zero": bool(lo <= 0 <= hi),
            "wilcoxon_p": p_raw,
            "n_years": int(diff.size),
        }
        pvals[col] = p_raw
    adjusted = holm_adjust(pvals)
    for col, padj in adjusted.items():
        if col in out["pairs"]:
            out["pairs"][col]["holm_p"] = padj
    # Friedman 整体检验
    cols = [c for c in pivot.columns]
    stat, p = friedman_test(pivot[cols].to_numpy(dtype=float))
    out["friedman"] = {"stat": stat, "p": p, "n_models": len(cols), "n_years": int(pivot.shape[0])}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="实验结果目录（含 window_metrics.csv）")
    parser.add_argument("--metric", default="sharpe")
    parser.add_argument("--benchmark", default="equal_weight")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    src = Path(args.dir)
    frame = pd.read_csv(src / "window_metrics.csv")
    cols = ["cagr", "vol", "sharpe", "sortino", "max_drawdown", "cvar_5", "avg_turnover",
            "mean_active", "sharpe_active", "mean_active_exposure", "mean_n_holdings"]
    cols = [c for c in cols if c in frame.columns]
    model_year = frame.groupby(["model", "test_year"])[cols].mean().reset_index()

    report = {}
    for metric in ("sharpe", "cagr", "cvar_5", "avg_turnover", "mean_active"):
        if metric in model_year.columns:
            report[metric] = compute(model_year, metric, args.benchmark)

    out_dir = Path(args.out) if args.out else src
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "significance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # 可读表格
    rows = []
    for metric, rep in report.items():
        if "pairs" not in rep:
            continue
        for model, v in rep["pairs"].items():
            rows.append({"metric": metric, "model": model, "diff_vs_benchmark": v["diff"],
                         "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                         "crosses_zero": v["crosses_zero"],
                         "wilcoxon_p": v["wilcoxon_p"], "holm_p": v.get("holm_p")})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "significance_table.csv", index=False)

    print(f"=== 相对 {args.benchmark} 的配对检验（单位：测试年份）===")
    print(f"年份数: {report['sharpe']['n_years']}, Friedman: "
          f"stat={report['sharpe']['friedman']['stat']:.3f}, p={report['sharpe']['friedman']['p']:.4f}")
    print()
    sh = table[table.metric == "sharpe"].sort_values("diff_vs_benchmark", ascending=False)
    print(sh.to_string(index=False, float_format=lambda v: f"{v: .4f}"))
    print(f"\n结果写入 {out_dir / 'significance.json'}")


if __name__ == "__main__":
    main()
