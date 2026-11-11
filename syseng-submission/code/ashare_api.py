"""A 股日频行情适配层。

出口可用性在本机会随代理服务状态变化，因此本模块把「取数」抽象为可切换的适配器：

- ``EastmoneyAdapter``：东方财富 ``push2his`` 行情接口，单请求返回全历史，效率最高；
  仅在网络出口可达时可用（该域名对大陆直连与部分出口会重置连接）。
- ``TencentAdapter``：腾讯 ``web.ifzq.gtimg.cn`` 复权 K 线接口，直连可达；
  单次最多 640 根 K 线，按日期窗口分块回补全历史。

两者统一输出列 ``date, open, close, high, low, volume, amount``，并统一使用
**后复权/前复权**口径以计算含分红送转的总收益（见 ``ADJUST`` 说明）。
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import net  # noqa: E402

EM_KLINE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TX_KLINE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

OUT_COLUMNS = ["date", "open", "close", "high", "low", "volume", "amount"]
ADJ_COLUMNS = ["date", "adj_open", "adj_close", "adj_high", "adj_low", "adj_volume"]


class NoDataError(RuntimeError):
    pass


def to_tencent_symbol(code: str) -> str:
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh{code}"
    return f"sz{code}"


def to_eastmoney_secid(code: str) -> str:
    market = "1" if code.startswith(("600", "601", "603", "605", "688", "689", "900")) else "0"
    return f"{market}.{code}"


def _parse_tencent_rows(rows: list, out_cols: list[str]) -> pd.DataFrame:
    """解析腾讯 K 线数组；剔除除权标记行（日期形如 ``2024-06``）。"""
    records = [row[:6] for row in rows if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(row[0]))]
    if not records:
        raise NoDataError("no valid tencent rows")
    frame = pd.DataFrame(records, columns=out_cols)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in out_cols[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna(subset=["date"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 东方财富适配器
# --------------------------------------------------------------------------
class EastmoneyAdapter:
    name = "eastmoney"
    supports_full_history = True

    @staticmethod
    def available() -> bool:
        try:
            net.get_json(
                EM_KLINE,
                {
                    "secid": "1.600519",
                    "fields1": "f1,f2",
                    "fields2": "f51,f53",
                    "klt": 101,
                    "fqt": 2,
                    "beg": "20240101",
                    "end": "20240201",
                },
                timeout=15,
                retries=1,
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def fetch(code: str, start: str = "1990-12-19", end: str = "2099-12-31") -> pd.DataFrame:
        payload = net.get_json(
            EM_KLINE,
            {
                "secid": to_eastmoney_secid(code),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": 101,
                "fqt": 2,
                "beg": start.replace("-", ""),
                "end": end.replace("-", ""),
            },
            timeout=30,
        )
        data = payload.get("data")
        if not data or not data.get("klines"):
            raise NoDataError(f"{code}: empty eastmoney payload")
        cols = ["date", "open", "close", "high", "low", "volume", "amount",
                "amplitude", "pct_chg", "change", "turnover"]
        frame = pd.DataFrame([line.split(",") for line in data["klines"]], columns=cols)
        frame["date"] = pd.to_datetime(frame["date"])
        for col in OUT_COLUMNS[1:] + ["turnover"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame[["date", *OUT_COLUMNS[1:], "turnover"]]


# --------------------------------------------------------------------------
# 腾讯适配器
# --------------------------------------------------------------------------
_TX_MONTH = re.compile(r"^\d{4}-\d{2}$")


class TencentAdapter:
    """腾讯行情适配器。

    单次请求最多返回 640 根 K 线，故按日期窗口向前分块回补。提供两种口径：

    - ``fetch_raw``：不复权真实价格与成交量，用于流动性排序、涨跌停判定与
      实际成交价（真实价格水平可解释，不会因复权基准变化而漂移）；
    - ``fetch_adj``：后复权价格（``hfq``），用于计算含分红送转的持有期总收益。
      后复权价恒为正，且相邻两日之比等于真实总收益率。
    """

    name = "tencent"
    supports_full_history = False
    CHUNK = 640
    MAX_CHUNKS = 40
    RAW_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"
    ADJ_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

    @staticmethod
    def available() -> bool:
        try:
            net.get_json(
                TencentAdapter.RAW_URL,
                {"param": "sh600519,day,2024-01-01,2024-02-01,320"},
                timeout=15,
                retries=1,
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def _one(cls, symbol: str, beg: str, end: str, *, adjusted: bool) -> pd.DataFrame:
        """单次请求。历史上溯到起点之前时服务端返回空数组，此时返回空表而非报错。"""
        if adjusted:
            payload = net.get_json(
                cls.ADJ_URL, {"param": f"{symbol},day,{beg},{end},{cls.CHUNK},hfq"}, timeout=30
            )
        else:
            payload = net.get_json(
                cls.RAW_URL, {"param": f"{symbol},day,{beg},{end},{cls.CHUNK}"}, timeout=30
            )
        data = payload.get("data")
        if not isinstance(data, dict) or not data:
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
        inner = data.get(symbol) or next(iter(data.values()))
        if not isinstance(inner, dict):
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
        key = next((k for k in inner if "day" in k), None)
        if key is None or not inner[key]:
            return pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
        cols = ["date", "open", "close", "high", "low", "volume"]
        try:
            return _parse_tencent_rows(inner[key], cols)
        except NoDataError:
            return pd.DataFrame(columns=cols)

    @classmethod
    def _fetch(cls, code: str, start: str, end: str, *, adjusted: bool) -> pd.DataFrame:
        symbol = to_tencent_symbol(code)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end < "2090-01-01" else pd.Timestamp.today().normalize()
        empty = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
        chunks: list[pd.DataFrame] = []
        cursor_end = end_ts
        seen: set[pd.Timestamp] = set()
        for _ in range(cls.MAX_CHUNKS):
            frame = cls._one(
                symbol, start_ts.strftime("%Y-%m-%d"), cursor_end.strftime("%Y-%m-%d"), adjusted=adjusted
            )
            frame = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
            if frame.empty:
                break
            chunks.append(frame)
            first_date = frame["date"].min()
            if first_date <= start_ts or first_date in seen:
                break
            seen.add(first_date)
            cursor_end = first_date - pd.Timedelta(days=1)
            if len(frame) < cls.CHUNK // 4:
                break
        if not chunks:
            raise NoDataError(f"{code}: no tencent data")
        out = pd.concat(chunks, ignore_index=True)
        return out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    @classmethod
    def fetch_raw(cls, code: str, start: str = "2005-01-01", end: str = "2099-12-31") -> pd.DataFrame:
        frame = cls._fetch(code, start, end, adjusted=False)
        frame["amount"] = frame["close"] * frame["volume"]
        return frame[OUT_COLUMNS]

    @classmethod
    def fetch_adj(cls, code: str, start: str = "2005-01-01", end: str = "2099-12-31") -> pd.DataFrame:
        frame = cls._fetch(code, start, end, adjusted=True)
        frame = frame.rename(
            columns={
                "open": "adj_open",
                "close": "adj_close",
                "high": "adj_high",
                "low": "adj_low",
                "volume": "adj_volume",
            }
        )
        return frame[ADJ_COLUMNS]


# --------------------------------------------------------------------------
# 适配器选择
# --------------------------------------------------------------------------
_ADAPTERS = [EastmoneyAdapter, TencentAdapter]
_CACHE: dict[str, object] = {}


def pick_adapter(prefer: str | None = None) -> object:
    """选择可用适配器，结果进程内缓存。"""
    key = prefer or "auto"
    if key in _CACHE:
        return _CACHE[key]
    order = _ADAPTERS
    if prefer:
        order = [a for a in _ADAPTERS if a.name == prefer] + [a for a in _ADAPTERS if a.name != prefer]
    for adapter in order:
        if adapter.available():
            _CACHE[key] = adapter
            return adapter
    raise RuntimeError("no A-share data adapter is reachable")


def fetch_prices(code: str, *, start: str = "2005-01-01", end: str = "2099-12-31", prefer: str | None = None) -> pd.DataFrame:
    """统一取数入口：返回 ``date, open, close, high, low, volume, amount, adj_close``。

    同时给出不复权价格（用于流动性与涨跌停判定）与后复权价（用于总收益）。
    东方财富适配器一次即可返回全历史且自带成交额；腾讯适配器需要分块。
    """
    adapter = pick_adapter(prefer)
    if adapter.name == "eastmoney":
        frame = adapter.fetch(code, start=start, end=end)
        frame = frame.rename(columns={"close": "raw_close"})
        return frame
    raw = adapter.fetch_raw(code, start=start, end=end)
    adj = adapter.fetch_adj(code, start=start, end=end)[["date", "adj_close"]]
    merged = raw.merge(adj, on="date", how="left")
    merged["adj_close"] = merged["adj_close"].ffill()
    return merged


if __name__ == "__main__":
    import time

    for adapter in _ADAPTERS:
        t0 = time.time()
        print(f"{adapter.name:11s} available={adapter.available()} ({time.time()-t0:.1f}s)")
    chosen = pick_adapter()
    print("chosen:", chosen.name)
    t0 = time.time()
    frame = fetch_prices("600519", start="2005-01-01")
    print(f"600519 rows={len(frame)} {frame['date'].min().date()}..{frame['date'].max().date()} ({time.time()-t0:.1f}s)")
    print(frame.head(3).to_string())
    print(frame.tail(3).to_string())
    r = frame["adj_close"].pct_change()
    print(f"adj daily ret: max={r.max():.4f} min={r.min():.4f}  |r|>0.105: {int((r.abs()>0.105).sum())}")
