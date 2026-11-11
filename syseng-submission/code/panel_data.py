r"""面板容器：把长表月度面板转换成回测所需的宽表矩阵。

核心对象
--------
``PanelData`` 保存日期索引、证券代码、特征矩阵、执行/退出价与持有期收益矩阵，
并保证所有矩阵的行列严格对齐。所有特征均由形成日（t 月末）及之前的信息构成，
持有期收益由执行日（t+1 月首日）到退出日（t+2 月首日）计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# 用于专家信号的特征（全部为形成日可得）
SIGNAL_FEATURES = ["mom120", "mom20", "reversal_1m", "vol60", "turnover20", "beta120", "mom250"]

REGIME_FEATURES = [
    "mkt_ret_1m",
    "mkt_ret_3m",
    "mkt_ret_12m",
    "mkt_vol_6m",
    "mkt_trend_6m",
    "mkt_drawdown_1y",
    "cs_disp",
    "up_share",
]


def _winsorize_zscore(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.notna().sum() < 5:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile(lower), s.quantile(upper)
    clipped = s.clip(lo, hi)
    std = clipped.std()
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (clipped - clipped.mean()) / std


@dataclass
class PanelData:
    dates: pd.DatetimeIndex
    codes: list[str]
    features: dict[str, np.ndarray]          # 特征名 -> (T, N)
    ret_next: np.ndarray                     # (T, N) 持有期收益
    tradable: np.ndarray                     # (T, N) 1=可买入
    in_universe: np.ndarray                  # (T, N) 1=进入当期股票池
    regime: np.ndarray                       # (T, R) 市场状态变量
    regime_names: list[str]
    industry: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_periods(self) -> int:
        return len(self.dates)

    @property
    def n_assets(self) -> int:
        return len(self.codes)

    def slice_periods(self, lo: int, hi: int) -> "PanelData":
        return PanelData(
            dates=self.dates[lo:hi],
            codes=self.codes,
            features={k: v[lo:hi] for k, v in self.features.items()},
            ret_next=self.ret_next[lo:hi],
            tradable=self.tradable[lo:hi],
            in_universe=self.in_universe[lo:hi],
            regime=self.regime[lo:hi],
            regime_names=self.regime_names,
            industry=self.industry,
            meta=self.meta,
        )


def build_panel_data(
    panel_path: str | Path | None = None,
    *,
    drop_delisted: bool = False,
    liquid_quantile: float | None = None,
) -> PanelData:
    path = Path(panel_path) if panel_path else DATA / "panel_monthly.parquet"
    panel = pd.read_parquet(path)
    panel["ym"] = panel["ym"].astype(str)

    frames = []
    for ym, grp in panel.groupby("ym", sort=True):
        rows = grp[grp["in_universe"] | grp["is_delist_buffer"]].copy()
        if rows.empty:
            continue
        if liquid_quantile is not None:
            thr = rows["amount_mean"].quantile(liquid_quantile)
            rows = rows[rows["amount_mean"] >= thr]
        frames.append(rows)
    panel = pd.concat(frames, ignore_index=True) if frames else panel.iloc[:0]

    if drop_delisted:
        panel = panel[~panel["is_delist_buffer"]]

    codes = sorted(panel["code"].unique())
    code_idx = {c: i for i, c in enumerate(codes)}
    # 时间索引用形成月份（YYYY-MM），而非各证券各自的月末交易日，
    # 否则同一形成月会因停牌差异被拆成多个"期"，导致滚动窗错位。
    ym_keys = sorted(panel["ym"].unique())
    date_idx = {ym: i for i, ym in enumerate(ym_keys)}
    dates = pd.DatetimeIndex([pd.Timestamp(f"{ym}-01") for ym in ym_keys])
    T, N = len(ym_keys), len(codes)

    # 行业以整数编码保存，避免 object 矩阵占用数百 MB
    ind_codes, ind_uniques = pd.factorize(panel["industry"].astype("string"))
    n_industry = max(len(ind_uniques), 1)

    ret_next = np.full((T, N), np.nan)
    tradable = np.zeros((T, N))
    in_uni = np.zeros((T, N))
    industry = np.full((T, N), -1, dtype=np.int16)

    ti = panel["ym"].map(date_idx).to_numpy()
    ni = panel["code"].map(code_idx).to_numpy()
    ret_next[ti, ni] = panel["ret_next"].to_numpy(dtype=float)
    tradable[ti, ni] = panel["tradable"].to_numpy(dtype=float)
    in_uni[ti, ni] = 1.0
    industry[ti, ni] = ind_codes.astype(np.int16)

    # 特征：每月横截面极值缩尾 + 标准化
    features: dict[str, np.ndarray] = {}
    panel = panel.assign(reversal_1m=-panel["mom20"])
    for name in SIGNAL_FEATURES:
        if name not in panel.columns:
            continue
        col = panel.groupby("ym")[name].transform(_winsorize_zscore)
        arr = np.full((T, N), np.nan)
        arr[ti, ni] = col.to_numpy(dtype=float)
        features[name] = arr

    # 制度变量：滚动 z 分数，只用历史（扩展窗均值/标准差，不含未来信息）
    mkt = pd.read_parquet(DATA / "index_daily.parquet")
    mkt["ym"] = mkt["ym_period_str"].astype(str)
    mkt = mkt.drop_duplicates(subset=["ym"]).set_index("ym").reindex(ym_keys)
    regime_list, regime_names = [], []
    for col in REGIME_FEATURES:
        if col not in mkt.columns:
            continue
        s = mkt[col].astype(float)
        mean = s.expanding(24).mean()
        sd = s.expanding(24).std()
        z = (s - mean) / sd.replace(0, np.nan)
        regime_list.append(z.to_numpy(dtype=float))
        regime_names.append(col)
    regime = np.column_stack(regime_list) if regime_list else np.zeros((T, 1))
    regime = np.nan_to_num(regime, nan=0.0)

    meta = {
        "panel_path": str(path),
        "n_codes": N,
        "n_periods": T,
        "mean_universe": float(in_uni.sum(axis=1).mean()),
        "drop_delisted": drop_delisted,
        "liquid_quantile": liquid_quantile,
    }
    return PanelData(
        dates=dates,
        codes=codes,
        features=features,
        ret_next=ret_next,
        tradable=tradable,
        in_universe=in_uni,
        regime=regime,
        regime_names=regime_names,
        industry=industry,
        meta=meta,
    )


def slice_by_dates(panel: PanelData, start: str | None, end: str | None) -> tuple[int, int]:
    lo = int(np.searchsorted(panel.dates, pd.Timestamp(start))) if start else 0
    hi = int(np.searchsorted(panel.dates, pd.Timestamp(end), side="right")) if end else panel.n_periods
    return lo, hi


if __name__ == "__main__":
    p = build_panel_data()
    print("dates:", p.dates.min().date(), "..", p.dates.max().date(), "T =", p.n_periods)
    print("codes:", p.n_assets)
    print("mean universe:", round(p.meta["mean_universe"], 1))
    print("features:", {k: v.shape for k, v in p.features.items()})
    print("regime names:", p.regime_names)
    finite = np.isfinite(p.ret_next).sum()
    print("finite ret_next cells:", finite)
    print("mean cross-sectional finite:", finite / max(p.n_periods, 1))
