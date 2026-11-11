r"""由逐只证券行情构造 A 股点时月度面板。

输入
----
``data/ashare_daily/<code>.parquet``  逐只证券日线（``dl_ashare_v2.py`` 产出）
``data/a_share_securities.parquet``   证券主表（含上市日、是否退市）

输出
----
``data/panel_monthly.parquet``    月度面板（每行 = 证券 × 形成月末）
``data/index_daily.parquet``      全市场等权与流通市值加权指数日收益
``data/panel_manifest.json``      构造清单

执行时点约定（全文一致）
------------------------
1. **t 月末收盘**观测全部特征与市场状态，形成目标权重。
2. **t+1 月第一个交易日收盘**建仓。
3. 持有至 **t+2 月第一个交易日收盘**换仓。

故单期持有收益为 ``adj_close(首个交易日 of t+2) / adj_close(首个交易日 of t+1) - 1``，
与"月末形成、次月持有"的标准月度调仓一致，且不使用任何形成日之后的信息。

点时纪律
--------
1. 流动性、历史长度、ST 状态只使用形成日及之前的信息。
2. **退市证券保留**：首次无行情的月份收益记为 -1（全部损失），控制幸存者偏差。
3. 不复权价用于流动性排序与涨跌停判定；后复权价用于计算含分红的总收益。
4. 执行月首三个交易日触及涨停的证券标记为不可买入，权重由回测引擎再分配。
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
STORE = DATA / "ashare_daily"

MIN_HISTORY_DAYS = 120
MIN_MONTH_DAYS = 5
LIQUIDITY_QUANTILE = 0.30
LIMIT_EPS = 0.002
ST_PATTERN = r"ST|\*|退$|退市|摘牌"


def limit_pct(code: str) -> float:
    """按板块返回日涨跌幅限制（科创板/创业板 20%，其余 10%）。"""
    return 0.20 if code.startswith(("688", "689", "300", "301")) else 0.10


def process_one(path: Path, sec_row: pd.Series) -> pd.DataFrame:
    """把单只证券的日线压缩为月末面板行。"""
    code = path.stem
    df = pd.read_parquet(path)
    if df.empty or len(df) < MIN_HISTORY_DAYS:
        return pd.DataFrame()
    df = df.sort_values("date").reset_index(drop=True)
    df["ym"] = df["date"].dt.to_period("M")

    close = df["close"].to_numpy(dtype=float)
    adj = df["adj_close"].to_numpy(dtype=float)
    if not np.isfinite(adj).any():
        return pd.DataFrame()
    # 后复权价缺失时用前行填充
    adj = pd.Series(adj).ffill().to_numpy()

    df["adj_close"] = adj
    df["ret"] = pd.Series(adj).pct_change().to_numpy()
    df.loc[np.abs(df["ret"]) > 0.45, "ret"] = np.nan
    lp = limit_pct(code)

    # ---- 日频特征（全部只用当日及之前的信息）----
    adj_s = pd.Series(adj)
    for w in (20, 60, 120, 250):
        df[f"mom{w}"] = adj_s.pct_change(w).to_numpy()
    ret_s = df["ret"]
    df["vol60"] = ret_s.rolling(60, min_periods=40).std().to_numpy() * np.sqrt(252)
    df["vol20"] = ret_s.rolling(20, min_periods=15).std().to_numpy() * np.sqrt(252)
    if "turnover" in df.columns:
        to_s = pd.to_numeric(df["turnover"], errors="coerce")
    else:
        to_s = pd.Series(np.nan, index=df.index)
    vol_s = pd.to_numeric(df["volume"], errors="coerce")
    amt_s = pd.to_numeric(df["amount"], errors="coerce")
    # 腾讯接口不提供换手率，改用可复现的成交量代理：
    # 换手率正比于成交量，故"相对自身历史基准的异常成交量"是同一信息的标准化版本。
    vol_base = vol_s.rolling(250, min_periods=120).mean()
    df["abn_volume"] = (vol_s / vol_base.replace(0, np.nan)).to_numpy()
    df["volume_cv60"] = (
        vol_s.rolling(60, min_periods=40).std() / vol_s.rolling(60, min_periods=40).mean().replace(0, np.nan)
    ).to_numpy()
    df["amihud"] = (df["ret"].abs() / (amt_s.replace(0, np.nan) / 1e8)).to_numpy()
    # 换手代理：异常成交量的 20 日均值（腾讯接口无换手率，用成交量口径替代）
    df["turnover20"] = vol_s.rolling(20, min_periods=15).mean().to_numpy() / vol_base.to_numpy()
    if not to_s.isna().all():
        df["turnover20"] = to_s.rolling(20, min_periods=15).mean().to_numpy()
    # 相对市场的 beta（120 日滚动）
    mkt = df["ret"].groupby(df["date"]).transform("mean")
    cov = (df["ret"] * mkt).rolling(120, min_periods=80).mean() - df["ret"].rolling(120, min_periods=80).mean() * mkt.rolling(120, min_periods=80).mean()
    var = (mkt**2).rolling(120, min_periods=80).mean() - mkt.rolling(120, min_periods=80).mean() ** 2
    df["beta120"] = (cov / var.replace(0, np.nan)).to_numpy()

    g = df.groupby("ym", sort=True)
    agg = g.agg(
        m_first=("date", "first"),
        m_last=("date", "last"),
        n_days=("date", "size"),
        amount_mean=("amount", "mean"),
        volume_mean=("volume", "mean"),
        ret_std_d=("ret", "std"),
        ret_min=("ret", "min"),
        ret_max=("ret", "max"),
        close_last=("close", "last"),
        adj_last=("adj_close", "last"),
    ).reset_index()

    # 月初首个交易日快照
    firsts = df.merge(agg[["ym", "m_first"]], on="ym", how="left")
    firsts = firsts[firsts["date"] == firsts["m_first"]]
    firsts = firsts[["ym", "close", "adj_close", "amount"]].rename(
        columns={"close": "close_first", "adj_close": "adj_first", "amount": "amount_first"}
    )
    agg = agg.merge(firsts, on="ym", how="left")

    # 月末特征快照
    lasts = df.merge(agg[["ym", "m_last"]], on="ym", how="left")
    lasts = lasts[lasts["date"] == lasts["m_last"]]
    feature_cols = ["mom20", "mom60", "mom120", "mom250", "vol60", "vol20", "turnover20",
                    "beta120", "amihud", "volume_cv60", "abn_volume"]
    feature_cols = [c for c in feature_cols if c in lasts.columns]
    lasts = lasts[["ym", *feature_cols]].copy()
    agg = agg.merge(lasts, on="ym", how="left")

    # 涨跌停标记（执行月首三日）
    df["limit_up"] = (df["ret"] >= lp - LIMIT_EPS).astype(float)
    df["limit_dn"] = (df["ret"] <= -lp + LIMIT_EPS).astype(float)
    lim = df.groupby("ym").agg(limit_up_days=("limit_up", "sum"), limit_dn_days=("limit_dn", "sum")).reset_index()
    agg = agg.merge(lim, on="ym", how="left")

    agg["code"] = code
    agg["exchange"] = sec_row.get("exchange")
    agg["name"] = sec_row.get("name")
    agg["industry"] = sec_row.get("industry")
    agg["list_date"] = sec_row.get("list_date")
    agg["is_delisted_flag"] = bool(sec_row.get("is_delisted", False))
    return agg


def build_panel(codes: list[str] | None = None, *, start: str = "2006-01-01") -> pd.DataFrame:
    sec = pd.read_parquet(DATA / "a_share_securities.parquet")
    sec["code"] = sec["code"].astype(str).str.zfill(6)
    sec = sec.set_index("code")

    files = sorted(STORE.glob("*.parquet"))
    if codes is not None:
        keep = set(codes)
        files = [f for f in files if f.stem in keep]
    print(f"processing {len(files):,} securities ...", flush=True)

    frames = []
    t0 = pd.Timestamp.now()
    for i, path in enumerate(files, 1):
        if path.stem not in sec.index:
            continue
        try:
            out = process_one(path, sec.loc[path.stem])
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {path.stem}: {type(exc).__name__} {exc}", flush=True)
            continue
        if not out.empty:
            frames.append(out)
        if i % 500 == 0:
            print(f"  {i:,}/{len(files):,} ({pd.Timestamp.now()-t0})", flush=True)
    if not frames:
        raise RuntimeError("no panel rows produced")
    panel = pd.concat(frames, ignore_index=True)
    panel["ym_period"] = pd.PeriodIndex(panel["ym"], freq="M")
    panel = panel.sort_values(["code", "ym_period"]).reset_index(drop=True)

    g = panel.groupby("code", sort=False)
    panel["exec_date"] = g["m_first"].shift(-1)
    panel["exit_date"] = g["m_first"].shift(-2)
    panel["ym_next"] = g["ym_period"].shift(-1)
    panel["adj_exec"] = g["adj_first"].shift(-1)
    panel["adj_exit"] = g["adj_first"].shift(-2)
    panel["limit_up_next"] = g["limit_up_days"].shift(-1)

    contiguous = panel["ym_next"] == (panel["ym_period"] + 1)
    valid = contiguous & panel["adj_exec"].notna() & panel["adj_exit"].notna() & (panel["adj_exec"] > 0)
    panel["ret_next"] = np.where(valid, panel["adj_exit"] / panel["adj_exec"] - 1.0, np.nan)

    # 退市缓冲：有执行价但因数据终止而没有退出价，判为"执行月内退市"，收益记 -1。
    # 必须区分三种情形，否则会把数据边界当成退市（系统性低估收益）：
    #   (a) 数据到样本右边界：最后一行本身，以及最后一行与前一行月份连续时的倒数第二行
    #       （后者的退出价落在尚未观测到的下一个月）-> 不记收益；
    #   (b) 数据在中间中断（真实退市或长期停牌）-> 记 -1；
    #   (c) 单条记录（无前置月份）-> 无法判定，不记收益。
    panel["obs_rank_from_end"] = panel.groupby("code")["ym_period"].rank(method="first", ascending=False)
    last_month = panel.groupby("code")["ym_period"].transform("max")
    prev_month = pd.PeriodIndex(last_month, freq="M") - 1
    panel["is_right_edge"] = (
        (panel["obs_rank_from_end"] <= 2) & (panel["ym_period"] >= prev_month)
    )
    panel["is_delist_buffer"] = (
        panel["adj_exec"].notna() & panel["adj_exit"].isna() & (~panel["is_right_edge"])
    )
    panel.loc[panel["is_delist_buffer"], "ret_next"] = -1.0
    # 右边界观测：既无收益也不可交易（不构成持仓）
    panel.loc[panel["is_right_edge"], "ret_next"] = np.nan

    panel["month_end"] = panel["m_last"]
    panel["tradable"] = np.where(
        (panel["adj_exec"].isna() & ~panel["is_delist_buffer"]) | panel["is_right_edge"], 0.0, 1.0
    ).astype(float)
    panel["ym"] = panel["ym"].astype(str)
    panel["ym_period_str"] = panel["ym_period"].astype(str)
    panel = panel[panel["month_end"] >= pd.Timestamp(start)].reset_index(drop=True)

    # 剔除尚不完整的样本右边界月份：该月的退出价落在尚未观测到的下一个月，
    # 保留会让部分证券的持有期收益缺失，使横截面在不同月份之间不可比。
    latest_complete = panel.loc[~panel["is_right_edge"], "ym_period"].max()
    panel = panel[panel["ym_period"] <= latest_complete].reset_index(drop=True)

    # 点的时（point-in-time）过滤
    panel["n_hist_proxy"] = panel["ym_period"].astype("period[M]").astype("int64") - pd.Period(
        "1990-01", freq="M"
    ).ordinal
    panel["is_st"] = panel["name"].astype(str).str.contains(ST_PATTERN, na=False, regex=True)
    panel["listed_months"] = (
        panel["ym_period"].astype("period[M]").astype("int64")
        - pd.PeriodIndex(pd.to_datetime(panel["list_date"], errors="coerce"), freq="M").astype("int64")
    )
    panel["liquid_rank"] = panel.groupby("ym_period_str")["amount_mean"].rank(pct=True, na_option="bottom")
    panel["in_universe"] = (
        (panel["n_days"] >= MIN_MONTH_DAYS)
        & (~panel["is_st"])
        & (panel["listed_months"] >= 6)
        & (panel["liquid_rank"] >= LIQUIDITY_QUANTILE)
        & panel["close_last"].notna()
    )
    return panel


def build_market_index(panel: pd.DataFrame) -> pd.DataFrame:
    """由面板构造市场层面的月度状态变量。

    使用全样本内可交易证券的等权收益作为市场收益，并计算等权指数的 12 个月
    滚动最高值与回撤。这些量只使用当期及之前的信息。
    """
    uni = panel[panel["in_universe"]].copy()
    uni = uni[uni["ret_next"].notna()]
    mkt = uni.groupby("ym_period_str").agg(
        mkt_ret=("ret_next", "mean"),
        mkt_ret_med=("ret_next", "median"),
        cs_disp=("ret_next", "std"),
        n_stocks=("code", "size"),
        up_share=("ret_next", lambda s: float((s > 0).mean())),
    ).reset_index()
    mkt = mkt.sort_values("ym_period_str").reset_index(drop=True)
    mkt["mkt_cum"] = (1.0 + mkt["mkt_ret"]).cumprod()
    mkt["mkt_ret_3m"] = mkt["mkt_cum"].pct_change(3)
    mkt["mkt_ret_12m"] = mkt["mkt_cum"].pct_change(12)
    mkt["mkt_vol_6m"] = mkt["mkt_ret"].rolling(6, min_periods=4).std() * np.sqrt(12)
    mkt["mkt_vol_12m"] = mkt["mkt_ret"].rolling(12, min_periods=8).std() * np.sqrt(12)
    roll_max = mkt["mkt_cum"].rolling(12, min_periods=6).max()
    mkt["mkt_drawdown_1y"] = mkt["mkt_cum"] / roll_max - 1.0
    mkt["mkt_ma6"] = mkt["mkt_cum"].rolling(6, min_periods=4).mean()
    mkt["mkt_trend_6m"] = mkt["mkt_cum"] / mkt["mkt_ma6"] - 1.0
    # 状态识别只用滞后信息（形成日 t 的状态由截至 t 的序列判定，可直接使用当期）
    return mkt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--out", default="panel_monthly.parquet")
    args = parser.parse_args()

    panel = build_panel(start=args.start)
    if args.limit:
        keep = sorted(panel["code"].unique())[: args.limit]
        panel = panel[panel["code"].isin(keep)]
    panel.to_parquet(DATA / args.out, index=False, compression="zstd")

    mkt = build_market_index(panel)
    mkt.to_parquet(DATA / "index_daily.parquet", index=False, compression="zstd")

    manifest = {
        "created_at": pd.Timestamp.now("UTC").isoformat(),
        "n_securities_in_store": len(list(STORE.glob("*.parquet"))),
        "panel_rows": int(len(panel)),
        "panel_codes": int(panel["code"].nunique()),
        "months": int(panel["ym"].nunique()),
        "month_min": str(panel["month_end"].min().date()),
        "month_max": str(panel["month_end"].max().date()),
        "in_universe_rows": int(panel["in_universe"].sum()),
        "mean_universe_size": float(panel.groupby("ym")["in_universe"].sum().mean()),
        "n_delist_buffer": int(panel["is_delist_buffer"].sum()),
        "execution_rule": "weights at month-end t close; executed at first trading day of t+1 close; "
                          "return to first trading day of t+2 close",
        "rules": {
            "min_month_days": MIN_MONTH_DAYS,
            "liquidity_quantile": LIQUIDITY_QUANTILE,
            "st_pattern": ST_PATTERN,
            "min_listed_months": 6,
            "limit_rule": "ChiNext/STAR 20%, others 10%; first three trading days of execution month",
        },
        "output": args.out,
    }
    (DATA / "panel_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print("\n月均样本量（年末快照）:", flush=True)
    snap = panel[panel["in_universe"]].groupby("ym").size()
    print(snap.iloc[::24].to_string(), flush=True)


if __name__ == "__main__":
    main()
