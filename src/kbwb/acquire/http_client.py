"""基于 httpx 的抓取实现。

用于站点采样这类"取回整页 HTML"的场景。爬虫的字段抽取在任务 4.3 走
scrapling；两者分开是有意的——采样只需要原始 HTML，不需要抽取能力。

编码判定单独成一层：中文站点常不声明编码或声明与实际不符，解码错了后面
的选择器推断全是乱码，因此优先信 meta 标签而非 HTTP 头。
"""

import re
from urllib.parse import urlsplit

import httpx

from kbwb.acquire.fetching import FetchError, FetchResponse

__all__ = ["HttpxFetcher", "decode_html"]

DEFAULT_TIMEOUT = 20.0

#: 页面头部的 charset 声明，兼容 <meta charset> 与 <meta http-equiv>
_CHARSET_RE = re.compile(rb'charset\s*=\s*["\']?\s*([\w-]+)', re.IGNORECASE)

#: 只扫描开头一段——charset 声明按规范必须出现在文档很靠前的位置
_SNIFF_BYTES = 2048

_FALLBACK_ENCODINGS = ("utf-8", "gb18030")


def decode_html(body: bytes, declared: str | None) -> str:
    """按 meta 声明 → HTTP 头声明 → 常见编码的顺序尝试解码。

    全部失败时以替换字符兜底而不是抛异常：一页乱码总好过整次采样中断。
    """
    match = _CHARSET_RE.search(body[:_SNIFF_BYTES])
    sniffed = match.group(1).decode("ascii", "ignore") if match else None
    for encoding in (sniffed, declared, *_FALLBACK_ENCODINGS):
        if not encoding:
            continue
        try:
            return body.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("utf-8", "replace")


class HttpxFetcher:
    """实现 :class:`kbwb.acquire.fetching.Fetcher` 契约。

    限速由 :class:`~kbwb.acquire.fetching.DomainThrottle` 在调用方一侧负责，
    此处只管取回单个 URL。
    """

    def __init__(
        self,
        *,
        user_agent: str,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.Client | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._client = client or httpx.Client(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        )
        self._owns_client = client is None

    def get(self, url: str) -> FetchResponse:
        scheme = urlsplit(url).scheme
        if scheme not in ("http", "https"):
            raise ValueError(f"只接受 http/https 地址：{url!r}")
        try:
            response = self._client.get(url, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc
        declared = response.charset_encoding
        return FetchResponse(
            url=str(response.url),
            status=response.status_code,
            text=decode_html(response.content, declared),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpxFetcher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
