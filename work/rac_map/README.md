# RAC-MAP reproducible prototype

This directory contains the current data-backed implementation for the
RAC-MAP study. It uses the public Yahoo Finance download only for a runnable
prototype. For a paper, freeze the downloaded files and replace the source
with CRSP/Compustat, WRDS, or another licensed source.

## Scope and provenance

- Assets: SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ, LQD, DBC.
- Requested window: 2010-01-01 through 2026-09-07 (Yahoo end date is exclusive).
- Observed feature window after a 60-day warm-up: 2010-03-31 through 2026-09-04.
- Main paper test period: 2024-01-01 through 2025-12-31.
- 2026 is saved as an external monitoring period, not part of the completed-year test.

`data.py` saves raw prices, a causal feature parquet, MASA-compatible
`ETF_10_1d.csv`, FinRL-compatible `finrl_processed.csv`, and `metadata.json`
with retrieval time and SHA-256 hashes.

## Run

From the repository root, with the shared interpreter:

```powershell
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\data.py
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\train_models.py --timesteps 5000
D:\Codex\PythonEnv\Scripts\python.exe work\rac_map\run_experiment.py --no-train
```

The experiment writes `results/metrics.csv`, per-period equity CSV files, and
PNG charts. The 1,000-timestep run checked into this workspace is a smoke test,
not a statistical claim; final results require multiple seeds, a frozen data
snapshot, and a pre-registered evaluation protocol.

## Baselines and models

- `buy_hold`, `equal_weight`, `risk_parity`: transparent non-RL baselines.
- `finrl_ppo`: PPO with the FinRL portfolio state/action contract and
  profit-oriented reward.
- `masa_td3`: two TD3 actors (profit and risk rewards) combined with a
  market-observer-conditioned gate, matching MASA's information flow.
- `masa_compatible`: deterministic MASA-inspired approximation useful for
  debugging the observer/controller logic.
- `static_multi_agent`: fixed mixture of specialized signals.
- `rac_map`: regime-adaptive, confidence-weighted gate with recent agent
  performance and volatility-aware risk scores. The environment's `rac_map`
  reward also subtracts turnover and empirical rolling CVaR penalties.
- `rac_map_gate.py`: learnable PyTorch gate with the same inputs, ready to be
  trained jointly with TD3/PPO experts; the smoke backtest uses the deterministic
  policy so its behavior remains inspectable.

The upstream repositories are retained in `../MASA` and `../FinRL` at their
cloned commits. Their historical Python/Gym/TA-Lib and Alpaca dependencies do
not import cleanly under Python 3.13, so this implementation is explicitly a
modern compatible reproduction rather than a byte-identical rerun.
