"""A 股点时（point-in-time）面板构造。

输入
----
``data/a_share_daily.parquet``       日频后复权行情（download_a_share.py 产出）
``data/a_share_securities.parquet``  证券主表（含退市标记）
``data/a_share_index.parquet``       指数日频行情

输出
----
``data/panel_monthly.parquet``  月末长表面板：每行 = (code, 形成月末 t)
``data/panel_manifest.json``    面板构造清单
``data/regime_monthly.parquet`` 市场状态序列

执行时点约定
------------
1. t 月末收盘后可观测全部特征与市场状态 -> 形成目标权重。
2. 在 **t+1 月第一个交易日**按该日收盘价建仓（保守：不给任何盘中止损/择时优势）。
3. 持有至 **t+2 月第一个交易日**收盘，按该日收盘价换仓。
4. 故单期持有收益为 ``close_first(t+2)/close_first(t+1) - 1``，与"月末形成、次月持有"一致。

点时纪律
--------
1. 流动性、历史长度、ST 状态、涨跌停均只使用形成日及之前（或当期执行日）的信息。
2. **退市证券保留在样本内**：首次无行情的月份收益记为 -1（等价于全部损失），
   这是控制幸存者偏差的标准做法；另提供剔除退市证券的对照样本。
3. 后复权价之比即真实总收益率，无需额外调整分红送转。
4. 涨跌停不可成交：在执行月首三个交易日触及涨停的证券，当期买入概率为零，
   在组合中以 ``tradable=0`` 标记并由回测引擎按比例再分配其权重。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ST_PATTERN = r"ST|\*|退|摘牌"
LIQUIDITY_QUANTILE = 0.30
MIN_MONTH_DAYS = 10
LIMIT_EPS = 0.002  # 判定涨跌停的容差


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(DATA / "a_share_daily.parquet")
    sec = pd.read_parquet(DATA / "a_share_securities.parquet")
    idx = pd.read_parquet(DATA / "a_share_index.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    idx["date"] = pd.to_datetime(idx["date"])
    for col in ("close", "volume", "amount", "turnover"):
        daily[col] = pd.to_numeric(daily[col], errors="coerce")
    return daily, sec, idx


# --------------------------------------------------------------------------
# 日频特征
# --------------------------------------------------------------------------
def build_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """日频收益率、动量、波动率、流动性与涨跌停标记。"""
    frame = daily.sort_values(["code", "date"]).reset_index(drop=True)
    g = frame.groupby("code", sort=False)

    frame["ret"] = g["close"].pct_change()
    invalid = frame["ret"].abs() > 0.45
    frame.loc[invalid, "ret"] = np.nan
    frame["log_amount"] = np.log1p(frame["amount"].clip(lower=0))

    for w in (20, 60, 120, 250):
        frame[f"mom{w}"] = g["close"].transform(lambda s: s.pct_change(w))

    frame["vol60"] = g["ret"].transform(lambda s: s.rolling(60, min_periods=40).std() * np.sqrt(252))
    frame["vol20"] = g["ret"].transform(lambda s: s.rolling(20, min_periods=15).std() * np.sqrt(252))
    frame["turnover20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    frame["turnover60"] = g["turnover"].transform(lambda s: s.rolling(60, min_periods=40).mean())
    frame["turnover_cv60"] = g["turnover"].transform(
        lambda s: s.rolling(60, min_periods=40).std() / s.rolling(60, min_periods=40).mean()
    )
    frame["amount20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    frame["n_hist"] = g.cumcount() + 1

    # 与市场的相关系数（低 beta 代理）
    market_ret = frame.groupby("date")["ret"].transform("mean")
    frame["mkt_ret"] = market_ret
    frame["ret_x_mkt"] = frame["ret"] * market_ret
    frame["mkt_sq"] = market_ret**2
    roll = frame.groupby("code", sort=False)
    cov = roll["ret_x_mkt"].transform(lambda s: s.rolling(120, min_periods=80).mean()) - roll[
        "ret"
    ].transform(lambda s: s.rolling(120, min_periods=80).mean()) * roll["mkt_ret"].transform(
        lambda s: s.rolling(120, min_periods=80).mean()
    )
    var = roll["mkt_sq"].transform(lambda s: s.rolling(120, min_periods=80).mean()) - roll[
        "mkt_ret"
    ].transform(lambda s: s.rolling(120, min_periods=80).mean()) ** 2
    frame["beta120"] = cov / var.replace(0, np.nan)

    frame["pct_limit"] = frame["code"].map(_limit_pct)
    frame["high_limit"] = (frame["ret"] >= frame["pct_limit"] - LIMIT_EPS).astype(float)
    frame["low_limit"] = (frame["ret"] <= -frame["pct_limit"] + LIMIT_EPS).astype(float)
    return frame


def _limit_pct(code: str) -> float:
    """按板块返回日涨跌幅限制。"""
    if code.startswith(("688", "689")):
        return 0.20
    if code.startswith(("300", "301")):
        return 0.20
    return 0.10


# --------------------------------------------------------------------------
# 月末面板
# --------------------------------------------------------------------------
def build_monthly_panel(daily_feat: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    frame = daily_feat.copy()
    frame["ym"] = frame["date"].dt.to_period("M")

    month_panel = frame.groupby(["code", "ym"], sort=True).agg(
        m_first=("date", "first"),
        m_last=("date", "last"),
        n_days=("date", "size"),
        amount_mean=("amount", "mean"),
        turnover_mean=("turnover", "mean"),
        ret_std_d=("ret", "std"),
        ret_min=("ret", "min"),
        ret_max=("ret", "max"),
        limit_up_days=("high_limit", "sum"),
        limit_dn_days=("low_limit", "sum"),
    ).reset_index()

    # 当月首日 / 末日快照
    mfirst = (
        frame.merge(month_panel[["code", "ym", "m_first"]], on=["code", "ym"], how="left")
        .query("date == m_first")[["code", "ym", "close", "high_limit", "low_limit", "n_hist", "amount"]]
        .rename(columns={"close": "close_first", "high_limit": "limit_up_first", "low_limit": "limit_dn_first"})
    )
    mlast_cols = [
        "mom20", "mom60", "mom120", "mom250", "vol60", "vol20",
        "turnover20", "turnover60", "turnover_cv60", "amount20", "n_hist", "log_amount", "beta120",
    ]
    mlast = (
        frame.merge(month_panel[["code", "ym", "m_last"]], on=["code", "ym"], how="left")
        .query("date == m_last")[["code", "ym", "close", *mlast_cols]]
        .rename(columns={"close": "close_last"})
    )

    panel = (
        month_panel.merge(mfirst, on=["code", "ym"], how="left")
        .merge(mlast, on=["code", "ym"], how="left")
        .merge(sec[["code", "name", "exchange", "list_date", "industry", "is_delisted"]], on="code", how="left")
    )
    panel["ym_period"] = panel["ym"]
    panel["ym"] = panel["ym"].astype(str)
    panel["month_end"] = panel["m_last"]
    panel = panel.sort_values(["code", "ym_period"]).reset_index(drop=True)

    g = panel.groupby("code", sort=False)
    panel["exec_date"] = g["m_first"].shift(-1)
    panel["exit_date"] = g["m_first"].shift(-2)
    panel["ym_next"] = g["ym_period"].shift(-1)
    panel["close_exec"] = g["close_first"].shift(-1)
    panel["close_exit"] = g["close_first"].shift(-2)

    # 持有期收益：执行日收盘 -> 退出日收盘
    contiguous = panel["ym_next"] == (panel["ym_period"] + 1)
    valid_ret = contiguous & panel["close_exec"].notna() & panel["close_exit"].notna()
    panel["ret_next"] = np.where(valid_ret, panel["close_exit"] / panel["close_exec"] - 1.0, np.nan)

    # 退市缓冲：有执行价但无退出价 => 在执行月内退市，收益记为 -1（全部损失）
    panel["is_delist_buffer"] = panel["close_exec"].notna() & panel["close_exit"].isna()
    panel.loc[panel["is_delist_buffer"], "ret_next"] = -1.0

    # 执行月首三个交易日的涨跌停状态：涨停不可买入，跌停不可卖出
    panel["tradable"] = _build_tradability(frame, panel)
    return panel


def _build_tradability(frame: pd.DataFrame, panel: pd.DataFrame) -> np.ndarray:
    """执行月首三个交易日内涨停（不可买）或全月长期停牌 => 不可交易。"""
    exec_map = panel[["code", "ym_period", "exec_date"]].copy()
    exec_map["exec_ym"] = exec_map["ym_period"] + 1
    merged = frame.merge(
        exec_map[["code", "exec_ym", "exec_date"]],
        left_on=["code", "ym"],
        right_on=["code", "exec_ym"],
        how="inner",
    )
    merged = merged[merged["date"] >= merged["exec_date"]]
    merged["day_rank"] = merged.groupby(["code", "exec_ym"]).cumcount()
    early = merged[merged["day_rank"] < 3]
    blocked = early.groupby(["code", "exec_ym"])["high_limit"].max().rename("blocked").reset_index()
    blocked["ym_period"] = (blocked["exec_ym"] - 1).astype(str)
    out = panel[["code", "ym_period"]].merge(
        blocked[["code", "ym_period", "blocked"]], on=["code", "ym_period"], how="left"
    )
    tradable = 1.0 - out["blocked"].fillna(0.0).to_numpy()
    # 执行月无行情（停牌整月）不可交易
    tradable = np.where(panel["close_exec"].isna().to_numpy() & ~panel["is_delist_buffer"].to_numpy(), 0.0, tradable)
    return tradable


def build_market_regimes(idx: pd.DataFrame) -> pd.DataFrame:
    """由指数行情构造月度市场状态变量（点时：只用当月及之前数据）。"""
    bench = idx[idx["index_code"] == "sh000985"].copy()
    if bench.empty:
        bench = idx[idx["index_code"] == "sh000300"].copy()
    if bench.empty:
        bench = idx[idx["index_code"] == "sh000001"].copy()
    bench = bench.sort_values("date")
    bench["ret"] = bench["close"].pct_change()
    bench["mkt_ret_1m"] = bench["close"].pct_change(21)
    bench["mkt_ret_3m"] = bench["close"].pct_change(63)
    bench["mkt_ret_12m"] = bench["close"].pct_change(252)
    bench["mkt_vol_20d"] = bench["ret"].rolling(20, min_periods=15).std() * np.sqrt(252)
    bench["mkt_vol_60d"] = bench["ret"].rolling(60, min_periods=40).std() * np.sqrt(252)
    bench["ma60"] = bench["close"].rolling(60, min_periods=40).mean()
    bench["ma200"] = bench["close"].rolling(200, min_periods=150).mean()
    bench["trend_60d"] = bench["close"] / bench["ma60"] - 1.0
    bench["trend_200d"] = bench["close"] / bench["ma200"] - 1.0
    roll_max = bench["close"].rolling(252, min_periods=120).max()
    bench["drawdown_1y"] = bench["close"] / roll_max - 1.0
    bench["ym"] = bench["date"].dt.to_period("M")

    agg = bench.groupby("ym").agg(
        bench_close=("close", "last"),
        mkt_ret_1m=("mkt_ret_1m", "last"),
        mkt_ret_3m=("mkt_ret_3m", "last"),
        mkt_ret_12m=("mkt_ret_12m", "last"),
        mkt_vol_20d=("mkt_vol_20d", "last"),
        mkt_vol_60d=("mkt_vol_60d", "last"),
        trend_60d=("trend_60d", "last"),
        trend_200d=("trend_200d", "last"),
        drawdown_1y=("drawdown_1y", "last"),
    ).reset_index()
    agg["ym"] = agg["ym"].astype(str)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-delisted", action="store_true", help="剔除退市缓冲观测（幸存者样本对照）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    daily, sec, idx = load_inputs()
    print(f"daily rows={len(daily):,} codes={daily['code'].nunique():,}", flush=True)
    feat = build_daily_features(daily)
    panel = build_monthly_panel(feat, sec)
    if args.drop_delisted:
        panel = panel[~panel["is_delist_buffer"]]

    out_name = args.out or ("panel_monthly_survivor.parquet" if args.drop_delisted else "panel_monthly.parquet")
    panel.to_parquet(DATA / out_name, index=False, compression="zstd")

    regimes = build_market_regimes(idx)
    regimes.to_parquet(DATA / "regime_monthly.parquet", index=False, compression="zstd")

    manifest = {
        "source_daily_rows": int(len(daily)),
        "source_codes": int(daily["code"].nunique()),
        "panel_rows": int(len(panel)),
        "panel_codes": int(panel["code"].nunique()),
        "months": int(panel["ym"].nunique()),
        "month_min": str(panel["month_end"].min().date()),
        "month_max": str(panel["month_end"].max().date()),
        "drop_delisted": bool(args.drop_delisted),
        "n_delist_buffer": int(panel["is_delist_buffer"].sum()),
        "output": out_name,
        "execution_rule": "weights formed at month-end t close; executed at first trading day of t+1 close; "
                          "return measured to first trading day of t+2 close",
        "rules": {
            "st_pattern": ST_PATTERN,
            "liquidity_quantile": LIQUIDITY_QUANTILE,
            "min_month_days": MIN_MONTH_DAYS,
            "adjustment": "backward-adjusted close (fqt=2)",
            "delisting_return": "-1.0 in the first month without a quote",
        },
    }
    (DATA / "panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
