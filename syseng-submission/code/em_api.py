"""东方财富（Eastmoney）公开行情接口客户端。

只使用公开、无需鉴权的 HTTP 接口获取 A 股日频行情、指数行情与证券列表。
所有请求带本地文件缓存，可重复运行而不重复下载。

网络说明
--------
本机 ``HTTP_PROXY``/``HTTPS_PROXY`` 环境变量指向已失效的 127.0.0.1:7890，
而系统代理实际监听 127.0.0.1:7897。本模块在导入时清除失效代理变量，
并显式使用可由 ``DSH_EM_PROXY`` 覆盖的代理解析顺序。

数据口径说明
------------
``fqt=2`` 为**后复权**价格。后复权价序列在任一日期 t 之后只相差一个常数
复权因子，因此相邻两日后复权价之比等于真实持有期总收益率（含分红送转），
可直接用于横截面组合收益计算，不引入未来信息。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

import pandas as pd
import requests

# 清除指向失效端口的代理变量，避免 requests 继承坏代理。
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    if os.environ.get(_var, "").endswith(":7890"):
        os.environ.pop(_var, None)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "raw_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://quote.eastmoney.com/",
    "Accept": "*/*",
}


def _proxy_candidates() -> list[dict | None]:
    override = os.environ.get("DSH_EM_PROXY")
    if override:
        if override.lower() in {"none", "direct", ""}:
            return [None]
        return [{"http": override, "https": override}]
    return [
        {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"},
        None,
    ]


class _Pacer:
    """自适应的请求节流器。

    服务端在并发过高时会直接断开连接，因此这里维护一个全局的最小请求间隔，
    失败时上调、连续成功时缓慢下调，把整体吞吐稳定在服务端可接受的水平。
    """

    def __init__(self, min_interval: float = 0.02, max_interval: float = 1.5) -> None:
        self._lock = threading.Lock()
        self._interval = min_interval
        self._min = min_interval
        self._max = max_interval
        self._next_allowed = 0.0
        self._ok_streak = 0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_allowed:
                    self._next_allowed = now + self._interval
                    return
                delay = self._next_allowed - now
            time.sleep(min(delay, 0.2))

    def on_success(self) -> None:
        with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 25:
                self._ok_streak = 0
                self._interval = max(self._min, self._interval * 0.8)

    def on_failure(self) -> None:
        with self._lock:
            self._ok_streak = 0
            self._interval = min(self._max, max(self._interval * 1.6, 0.05))

    @property
    def interval(self) -> float:
        with self._lock:
            return self._interval


PACER = _Pacer()


_LOCAL = threading.local()


def _session() -> requests.Session:
    sess = getattr(_LOCAL, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update(_HEADERS)
        sess.proxies = {}
        _LOCAL.session = sess
    return sess


class EastMoneyError(RuntimeError):
    pass


def _cache_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{digest}.json"


def get_json(
    url: str,
    params: dict,
    *,
    retries: int = 6,
    timeout: float = 30.0,
    cache: bool = True,
    refresh: bool = False,
) -> dict:
    """带磁盘缓存、自适应节流与多代理回退的 JSON GET。"""
    key = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    path = _cache_path(key)
    if cache and path.exists() and not refresh:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)

    last_err: Exception | None = None
    for attempt in range(retries):
        PACER.wait()
        proxies = _proxy_candidates()[min(attempt, len(_proxy_candidates()) - 1)]
        try:
            resp = _session().get(url, params=params, timeout=timeout, proxies=proxies)
            if resp.status_code == 200:
                payload = resp.json()
                PACER.on_success()
                if cache:
                    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return payload
            last_err = EastMoneyError(f"HTTP {resp.status_code}")
            PACER.on_failure()
        except Exception as exc:  # noqa: BLE001 - 网络层重试
            last_err = exc
            PACER.on_failure()
        time.sleep(min(0.3 * (2**attempt), 2.0) + random.random() * 0.25)
    raise EastMoneyError(f"request failed: {url} {params} :: {last_err}")


# --------------------------------------------------------------------------
# 证券列表
# --------------------------------------------------------------------------
_DATACENTER = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def fetch_security_list(*, refresh: bool = False, max_pages: int = 120) -> pd.DataFrame:
    """全部 A 股证券列表（含已退市），来自东方财富数据中心 F10 基本信息表。"""
    rows: list[dict] = []
    page = 1
    total_pages = None
    while page <= max_pages:
        payload = get_json(
            _DATACENTER,
            {
                "reportName": "RPT_F10_BASIC_ORGINFO",
                "columns": "SECURITY_CODE,SECURITY_NAME_ABBR,TRADE_MARKET,LISTING_DATE,INDUSTRYCSRC1,ORG_NAME",
                "pageSize": 500,
                "pageNumber": page,
                "sortColumns": "SECURITY_CODE",
                "sortTypes": 1,
                "source": "WEB",
                "client": "WEB",
            },
            refresh=refresh,
        )
        result = payload.get("result") or {}
        data = result.get("data") or []
        if total_pages is None:
            total_pages = int(result.get("pages") or 0)
        if not data:
            break
        rows.extend(data)
        if total_pages and page >= total_pages:
            break
        page += 1

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise EastMoneyError("empty security list")
    frame = frame.rename(
        columns={
            "SECURITY_CODE": "code",
            "SECURITY_NAME_ABBR": "name",
            "TRADE_MARKET": "trade_market",
            "LISTING_DATE": "list_date",
            "INDUSTRYCSRC1": "industry",
            "ORG_NAME": "org_name",
        }
    )
    frame = frame.drop_duplicates(subset=["code"]).reset_index(drop=True)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
    frame["exchange"] = frame["code"].map(_exchange_of)
    frame["secid"] = frame["code"].map(_secid_of)
    frame = frame[frame["exchange"].notna()].reset_index(drop=True)
    keep = ["code", "name", "exchange", "secid", "list_date", "industry", "trade_market"]
    return frame[keep]


def _exchange_of(code: str) -> str | None:
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "SH"
    if code.startswith(("000", "001", "002", "003", "300", "301", "200")):
        return "SZ"
    if code.startswith(("430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "874", "875", "876", "877", "920")):
        return "BJ"
    return None


def _secid_of(code: str) -> str:
    ex = _exchange_of(code)
    market = "1" if ex == "SH" else "0"
    return f"{market}.{code}"


# --------------------------------------------------------------------------
# 日频行情
# --------------------------------------------------------------------------
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
KLINE_COLUMNS = [
    "date", "open", "close", "high", "low", "volume", "amount",
    "amplitude", "pct_chg", "change", "turnover",
]
_NUMERIC = ["open", "close", "high", "low", "volume", "amount", "amplitude", "pct_chg", "change", "turnover"]


def fetch_kline(
    secid: str,
    *,
    beg: str = "19901219",
    end: str = "20991231",
    klt: int = 101,
    fqt: int = 2,
    refresh: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """单只证券日频行情。``klt=101`` 日线；``fqt=2`` 后复权，``fqt=0`` 不复权。"""
    payload = get_json(
        KLINE_URL,
        {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "klt": klt,
            "fqt": fqt,
            "beg": beg,
            "end": end,
        },
        refresh=refresh,
        cache=cache,
    )
    data = payload.get("data")
    if not data or not data.get("klines"):
        return pd.DataFrame(columns=["date", *_NUMERIC])
    frame = pd.DataFrame([line.split(",") for line in data["klines"]], columns=KLINE_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    for col in _NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


# --------------------------------------------------------------------------
# 指数
# --------------------------------------------------------------------------
INDEX_SECIDS: dict[str, str] = {
    # 宽基
    "sh000001": "1.000001", "sz399001": "0.399001", "sz399106": "0.399106",
    "sh000300": "1.000300", "sh000905": "1.000905", "sh000852": "1.000852",
    "sh000016": "1.000016", "sz399006": "0.399006", "sh000688": "1.000688",
    "sh000985": "1.000985", "sz399101": "0.399101", "sz399102": "0.399102",
    # 风格
    "sh000015": "1.000015", "sh000922": "1.000922", "sz399321": "0.399321",
    "sh000918": "1.000918", "sh000919": "1.000919",
    # 行业（中证行业指数）
    "sh000037": "1.000037", "sh000038": "1.000038", "sh000039": "1.000039",
    "sh000040": "1.000040", "sh000041": "1.000041", "sh000042": "1.000042",
    "sh000043": "1.000043", "sh000044": "1.000044", "sh000045": "1.000045",
    "sh000046": "1.000046",
    # 主题
    "sz399975": "0.399975", "sz399997": "0.399997", "sz399989": "0.399989",
    "sh000827": "1.000827", "sz399808": "0.399808",
    # 海外与债券（用于国际对照）
    "sh000012": "1.000012", "sh000013": "1.000013",
}


def fetch_indices(*, refresh: bool = False) -> pd.DataFrame:
    frames = []
    for name, secid in INDEX_SECIDS.items():
        frame = fetch_kline(secid, beg="20000101", refresh=refresh)
        if frame.empty:
            continue
        frame["index_code"] = name
        frames.append(frame)
    if not frames:
        raise EastMoneyError("no index data downloaded")
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "index_code", "open", "close", "high", "low", "volume", "amount", "pct_chg"]]


if __name__ == "__main__":
    uni = fetch_security_list()
    print("universe:", len(uni))
    print(uni.head().to_string())
    print(uni["exchange"].value_counts().to_string())
    k = fetch_kline("1.600519", beg="20050101")
    print("600519:", len(k), k["date"].min().date(), k["date"].max().date())
