"""抓取层的最小契约与限速。

具体的抓取实现（scrapling）留到爬虫任务；这里只定义调用方需要的形状，
使 robots 判定与站点采样可以在不触网的前提下被测试与复用。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

__all__ = ["DomainThrottle", "FetchError", "FetchResponse", "Fetcher", "origin_of"]


class FetchError(RuntimeError):
    """网络层失败：超时、连接错误、DNS 失败等。"""


@dataclass(frozen=True, slots=True)
class FetchResponse:
    url: str
    status: int
    text: str


class Fetcher(Protocol):
    def get(self, url: str) -> FetchResponse: ...


def origin_of(url: str) -> str:
    """返回 ``scheme://host[:port]``。robots 与限速都以此为单位。"""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"只接受 http/https 地址：{url!r}")
    return f"{parts.scheme}://{parts.netloc}"


class DomainThrottle:
    """按域名的最小请求间隔。

    时钟与休眠可注入，使限速行为能被断言而无需真的等待。
    """

    def __init__(
        self,
        min_interval_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_seconds <= 0:
            raise ValueError("最小请求间隔必须为正数")
        self._min_interval = min_interval_seconds
        self._clock = clock
        self._sleep = sleep
        self._last_request: dict[str, float] = {}

    def acquire(self, url: str) -> None:
        """必要时等待，直到该域名允许发起下一次请求。"""
        origin = origin_of(url)
        now = self._clock()
        last = self._last_request.get(origin)
        if last is not None:
            waited = now - last
            if waited < self._min_interval:
                self._sleep(self._min_interval - waited)
                now = self._clock()
        self._last_request[origin] = now
