"""真实 HTTP 抓取。

中文站点常不在 header 里正确声明编码，或声明与实际不符——解码错了后面
一切都是乱码，因此编码判定单独成一层并单独验证。
"""

import httpx
import pytest

from kbwb.acquire.fetching import FetchError, FetchResponse
from kbwb.acquire.http_client import HttpxFetcher, decode_html

UA = "kbwb/0.1 (+https://example.cn)"


class TestDecoding:
    def test_meta_charset_wins_over_header(self):
        body = '<meta charset="gb18030"><p>教务处通知</p>'.encode("gb18030")
        assert "教务处通知" in decode_html(body, declared="iso-8859-1")

    def test_http_equiv_meta_recognised(self):
        body = (
            '<meta http-equiv="Content-Type" content="text/html; charset=gb2312">'
            "<p>学籍证明</p>"
        ).encode("gb18030")
        assert "学籍证明" in decode_html(body, declared=None)

    def test_declared_encoding_used_when_no_meta(self):
        assert "转专业" in decode_html("<p>转专业</p>".encode("gb18030"), declared="gb18030")

    def test_utf8_default(self):
        assert "通知公告" in decode_html("<p>通知公告</p>".encode("utf-8"), declared=None)

    def test_falls_back_without_raising(self):
        # 声明错误且 meta 缺失时不得抛异常，宁可给出可读性下降的文本
        assert isinstance(decode_html(b"\xff\xfe\x00bad", declared="utf-8"), str)

    def test_unknown_declared_encoding_ignored(self):
        assert "通知" in decode_html("<p>通知</p>".encode("utf-8"), declared="x-nonsense")


def _fetcher(handler):
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, headers={"User-Agent": UA})
    return HttpxFetcher(user_agent=UA, client=client)


class TestFetching:
    def test_returns_status_and_text(self):
        def handler(request):
            return httpx.Response(200, text="<p>正文</p>")

        response = _fetcher(handler).get("https://x.edu.cn/a.htm")
        assert isinstance(response, FetchResponse)
        assert response.status == 200 and "正文" in response.text

    def test_sends_the_identifiable_user_agent(self):
        seen = {}

        def handler(request):
            seen["ua"] = request.headers.get("user-agent")
            return httpx.Response(200, text="ok")

        _fetcher(handler).get("https://x.edu.cn/a.htm")
        assert seen["ua"] == UA

    def test_error_status_is_returned_not_raised(self):
        def handler(request):
            return httpx.Response(404, text="")

        assert _fetcher(handler).get("https://x.edu.cn/a.htm").status == 404

    def test_reports_the_final_url_after_redirect(self):
        def handler(request):
            if request.url.path == "/old.htm":
                return httpx.Response(301, headers={"Location": "/new.htm"})
            return httpx.Response(200, text="ok")

        assert _fetcher(handler).get("https://x.edu.cn/old.htm").url.endswith("/new.htm")

    def test_network_failure_becomes_fetch_error(self):
        def handler(request):
            raise httpx.ConnectTimeout("timed out")

        with pytest.raises(FetchError):
            _fetcher(handler).get("https://x.edu.cn/a.htm")

    def test_non_http_url_rejected_before_request(self):
        called = []

        def handler(request):
            called.append(request)
            return httpx.Response(200, text="ok")

        with pytest.raises(ValueError):
            _fetcher(handler).get("file:///etc/passwd")
        assert called == []

    def test_decodes_gb18030_page(self):
        def handler(request):
            return httpx.Response(
                200,
                content='<meta charset="gb2312"><p>教务处</p>'.encode("gb18030"),
                headers={"Content-Type": "text/html"},
            )

        assert "教务处" in _fetcher(handler).get("https://x.edu.cn/a.htm").text

    def test_close_is_safe_to_call_twice(self):
        fetcher = _fetcher(lambda r: httpx.Response(200, text="ok"))
        fetcher.close()
        fetcher.close()
