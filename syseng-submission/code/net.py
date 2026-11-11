"""统一网络访问层。

本机 ``HTTP_PROXY``/``HTTPS_PROXY`` 环境变量指向已失效的 127.0.0.1:7890，
而可用出口在“系统代理（127.0.0.1:7897）”与“直连”之间切换。为避免 requests
继承坏代理，所有请求统一经本模块发起：

1. 创建 ``trust_env=False`` 的 Session，彻底忽略环境代理变量；
2. 按 ``DSH_NET_PROXY`` -> 系统代理端口 -> 直连 的顺序探测可用出口；
3. 成功后缓存该出口，失败时自动切换并重试。
"""

from __future__ import annotations

import os
import random
import threading
import time

import requests

_DEFAULT_PROXY_PORTS = (7897, 7890, 10809, 1080)
_LOCAL = threading.local()
_STATE_LOCK = threading.Lock()
_ACTIVE_PROXY: object = "UNPROBED"  # "UNPROBED"=尚未探测；None=直连；str=代理 URL
PROBE_URL = "https://web.ifzq.gtimg.cn/appstock/app/kline/kline"


class Pacer:
    """全局自适应节流器。

    行情站点在短时间内收到过多请求后会连续拒绝。本类维护一个全局最小请求间隔：
    连续成功时缓慢下调间隔以提高吞吐，出现拒绝时成倍上调间隔。所有线程共用同一节流器。
    """

    def __init__(self, interval: float = 0.05, min_interval: float = 0.02, max_interval: float = 4.0) -> None:
        self.lock = threading.Lock()
        self.interval = interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._next = 0.0
        self._ok_streak = 0

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                if now >= self._next:
                    self._next = now + self.interval
                    return
                delay = self._next - now
            time.sleep(min(delay, 0.2))

    def success(self) -> None:
        with self.lock:
            self._ok_streak += 1
            if self._ok_streak >= 40:
                self._ok_streak = 0
                self.interval = max(self.min_interval, self.interval * 0.85)

    def failure(self) -> None:
        with self.lock:
            self._ok_streak = 0
            self.interval = min(self.max_interval, max(self.interval * 2.0, 0.1))

    def set_interval(self, value: float) -> None:
        with self.lock:
            self.interval = min(self.max_interval, max(self.min_interval, float(value)))


PACER = Pacer()


def _env_proxy() -> str | None:
    raw = os.environ.get("DSH_NET_PROXY", "").strip()
    if not raw:
        return None
    if raw.lower() in {"none", "direct", "off", "0"}:
        return ""
    return raw


def _candidates() -> list[str | None]:
    env = _env_proxy()
    if env is not None:
        return [None] if env == "" else [env]
    return [f"http://127.0.0.1:{p}" for p in _DEFAULT_PROXY_PORTS] + [None]


def _proxy_dict(proxy: str | None) -> dict:
    if proxy is None:
        return {"http": "", "https": ""}
    return {"http": proxy, "https": proxy}


def _new_session() -> requests.Session:
    sess = requests.Session()
    sess.trust_env = False  # 关键：不继承失效的环境代理
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }
    )
    return sess


def session() -> requests.Session:
    sess = getattr(_LOCAL, "sess", None)
    if sess is None:
        sess = _new_session()
        _LOCAL.sess = sess
    return sess


def _active() -> object:
    """返回当前已知可用出口；``_UNPROBED`` 表示尚未探测，``None`` 表示直连。"""
    with _STATE_LOCK:
        return _ACTIVE_PROXY


def _set_active(proxy: str | None) -> None:
    global _ACTIVE_PROXY
    with _STATE_LOCK:
        _ACTIVE_PROXY = proxy


def get(
    url: str,
    params: dict | None = None,
    *,
    retries: int = 4,
    timeout: float = 30.0,
    headers: dict | None = None,
    allow_proxy_switch: bool = True,
    pace: bool = True,
) -> requests.Response:
    """发起 GET，自动在可用出口之间回退，并在服务端拒绝时自适应降速。

    出口策略：一旦探测到可用出口（很可能是"直连"），后续请求**始终优先使用它**；
    只有当该出口本身失败时才回到候选列表重新探测。若每次失败都重置探测状态，
    会在多个失效代理端口之间反复空转，把重试预算全部浪费掉。
    """
    known = _active()
    if known == "UNPROBED":
        order: list[str | None] = _candidates()
    elif allow_proxy_switch:
        order = [known] + [c for c in _candidates() if c != known]  # type: ignore[list-item]
    else:
        order = [known]  # type: ignore[list-item]

    last_err: Exception | None = None
    for attempt in range(max(retries, len(order))):
        proxy = order[min(attempt, len(order) - 1)]
        if pace:
            PACER.wait()
        try:
            resp = session().get(
                url,
                params=params,
                timeout=timeout,
                proxies=_proxy_dict(proxy),
                headers=headers,
            )
            if resp.status_code < 500:
                _set_active(proxy)
                return resp
            last_err = RuntimeError(f"HTTP {resp.status_code}")
            if pace:
                PACER.failure()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if pace:
                PACER.failure()
        time.sleep(min(0.4 * (2**attempt), 3.0) + random.random() * 0.3)
    raise RuntimeError(f"GET failed: {url} {params} :: {last_err}")


def get_json(url: str, params: dict | None = None, *, pace: bool = True, **kwargs) -> dict:
    resp = get(url, params, pace=pace, **kwargs)
    resp.raise_for_status()
    payload = resp.json()
    # 服务端限流时可能返回 200 但内容为空；把这种情况计为一次失败以触发降速
    if pace:
        if payload:
            PACER.success()
        else:
            PACER.failure()
    return payload


def probe(url: str, *, timeout: float = 10.0) -> tuple[bool, str]:
    """探测某个出口是否可用。返回 (是否成功, 说明)。"""
    for proxy in _candidates():
        try:
            resp = session().get(url, timeout=timeout, proxies=_proxy_dict(proxy))
            if resp.status_code < 500:
                return True, f"{'direct' if proxy is None else proxy} -> HTTP {resp.status_code}"
        except Exception:  # noqa: BLE001
            continue
    return False, "all routes failed"


def probe_and_pin(timeout: float = 20.0) -> str | None:
    """探测可用出口并把它固定下来，避免运行中反复在失效代理之间空转。

    返回被固定的出口（``None`` 表示直连）。探测失败时返回 ``None`` 并保持直连。
    """
    for proxy in _candidates():
        try:
            resp = session().get(
                PROBE_URL,
                params={"param": "sh600519,day,2024-01-01,2026-01-01,640"},
                timeout=timeout,
                proxies=_proxy_dict(proxy),
            )
            if resp.status_code < 500 and len(resp.content) > 1000:
                _set_active(proxy)
                return proxy
        except Exception:  # noqa: BLE001
            continue
    _set_active(None)
    return None


if __name__ == "__main__":
    for target in (
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh600519,day,2024-01-01,2024-02-01,320,qfq",
        "https://www.csindex.com.cn",
    ):
        ok, msg = probe(target)
        print(f"{'OK ' if ok else 'ERR'} {msg:34s} {target[:70]}")
