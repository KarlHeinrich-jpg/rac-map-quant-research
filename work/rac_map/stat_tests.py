"""Window-level paired bootstrap and nonparametric model comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def paired_block_bootstrap(x: np.ndarray, y: np.ndarray, reps: int = 5000, block: int = 2, seed: int = 42) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or len(x) < 3:
        raise ValueError("paired bootstrap requires at least three paired observations")
    rng = np.random.default_rng(seed)
    n = len(x)
    starts = np.arange(n)
    draws = np.empty(reps)
    for b in range(reps):
        idx = []
        while len(idx) < n:
            start = int(rng.choice(starts))
            idx.extend((start + np.arange(block)) % n)
        idx = np.asarray(idx[:n])
        draws[b] = np.mean((x - y)[idx])
    observed = float(np.mean(x - y))
    return {"difference": observed, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p_two_sided": float(2 * min(np.mean(draws <= 0), np.mean(draws >= 0)))}


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    out = {}
    running = 0.0
    m = len(ordered)
    for rank, (key, p) in enumerate(ordered):
        adj = min(1.0, max(running, (m - rank) * p))
        running = adj
        out[key] = adj
    return out


def run_tests(metrics_path: str | Path, reference: str = "equal_weight", metric: str = "sharpe", reps: int = 5000) -> dict:
    frame = pd.read_csv(metrics_path)
    # Random initializations measure optimization uncertainty, not additional
    # realizations of the market.  Collapse seeds within each test year before
    # inference so deterministic baselines are never pseudo-replicated.
    model_year = frame.groupby(["test_year", "model"], as_index=False)[metric].mean()
    pivot = model_year.pivot(index="test_year", columns="model", values=metric).sort_index()
    models = [c for c in pivot.columns if c != reference]
    bootstrap = {}
    raw_p = {}
    for model in models:
        pair = pivot[[model, reference]].dropna()
        result = paired_block_bootstrap(pair[model].to_numpy(), pair[reference].to_numpy(), reps=reps, block=2)
        try:
            result["wilcoxon_p"] = float(wilcoxon(pair[model], pair[reference], zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            result["wilcoxon_p"] = 1.0
        bootstrap[model] = result
        raw_p[model] = result["wilcoxon_p"]
    adjusted = holm_adjust(raw_p)
    for model, p in adjusted.items():
        bootstrap[model]["wilcoxon_holm_p"] = p
    available = [pivot[m].dropna().to_numpy() for m in pivot.columns]
    friedman = friedmanchisquare(*available) if len(available) >= 3 and all(len(x) >= 3 for x in available) else None
    return {"metrics_path": str(Path(metrics_path).resolve()), "reference": reference, "metric": metric,
            "pairs": bootstrap, "friedman": None if friedman is None else {"statistic": float(friedman.statistic), "pvalue": float(friedman.pvalue)},
            "unit": "test year after within-year seed averaging; paired circular block bootstrap with block length 2",
            "n_test_years": int(len(pivot)),
            "seed_treatment": "seeds quantify optimization uncertainty and are averaged within test year before market-level inference"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="work/rac_map/results/rolling/rolling_metrics.csv")
    parser.add_argument("--reference", default="equal_weight")
    parser.add_argument("--metric", default="sharpe")
    parser.add_argument("--reps", type=int, default=5000)
    parser.add_argument("--output", default="work/rac_map/results/rolling/stat_tests.json")
    args = parser.parse_args()
    result = run_tests(args.metrics, args.reference, args.metric, args.reps)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
