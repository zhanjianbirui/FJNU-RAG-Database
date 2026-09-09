"""站点采样器。

site-analysis 规格三条：
- 从列表型入口页识别详情页链接，采样不超过配置上限，并返回数量与 URL；
- 被 robots 禁止时**不发起请求**，报告被禁 URL 与对应规则；
- 入口页 4xx/5xx/超时时报告失败原因并继续处理其余入口，而非整体中止。
"""

import pytest

from kbwb.acquire.fetching import DomainThrottle, FetchError, FetchResponse
from kbwb.acquire.sampling import Sampler, build_user_agent
from kbwb.config.site import parse_site_config

HOST = "https://jwc.example.edu.cn"

LIST_PAGE = f"""
<html><body>
  <ul class="news">
    <li><a href="/tzgg/2026/0301.htm">关于办理学籍证明的通知</a></li>
    <li><a href="/tzgg/2026/0302.htm">转专业申请通知</a></li>
    <li><a href="{HOST}/tzgg/2026/0303.htm">课程重修安排</a></li>
    <li><a href="/admin/secret.htm">后台</a></li>
    <li><a href="https://other.example.cn/x.htm">站外链接</a></li>
    <li><a href="#top">锚点</a></li>
    <li><a href="/tzgg/2026/0301.htm">重复链接</a></li>
  </ul>
</body></html>
"""

DETAIL = "<html><body><div class='content'><h1>标题</h1><p>正文</p></div></body></html>"

ROBOTS = "User-agent: *\nDisallow: /admin/\n"


def _config(**overrides):
    base = {
        "name": "jwc",
        "entry_urls": [f"{HOST}/tzgg.htm"],
        "contact": "mailto:admin@example.edu.cn",
        "selectors": {"body": "div.content", "title": "h1"},
    }
    return parse_site_config({**base, **overrides})


class StubFetcher:
    def __init__(self, pages=None, robots=ROBOTS, errors=None, statuses=None):
        self.pages = pages if pages is not None else {f"{HOST}/tzgg.htm": LIST_PAGE}
        self.robots = robots
        self.errors = errors or {}
        self.statuses = statuses or {}
        self.requested = []

    def get(self, url):
        self.requested.append(url)
        if url.endswith("/robots.txt"):
            return FetchResponse(url=url, status=200, text=self.robots)
        if url in self.errors:
            raise FetchError(self.errors[url])
        status = self.statuses.get(url, 200)
        if status >= 400:
            return FetchResponse(url=url, status=status, text="")
        return FetchResponse(url=url, status=status, text=self.pages.get(url, DETAIL))


def _sampler(fetcher, **kwargs):
    clock = iter(range(10_000))
    return Sampler(
        fetcher,
        contact="mailto:admin@example.edu.cn",
        throttle=DomainThrottle(1.0, clock=lambda: next(clock), sleep=lambda _: None),
        **kwargs,
    )


class TestUserAgent:
    def test_identifies_project_and_contact(self):
        ua = build_user_agent("mailto:admin@example.edu.cn")
        assert "kbwb" in ua.lower()
        assert "mailto:admin@example.edu.cn" in ua

    def test_rejects_blank_contact(self):
        with pytest.raises(ValueError):
            build_user_agent("")


class TestSampling:
    def test_collects_detail_pages_from_the_entry(self):
        result = _sampler(StubFetcher()).sample(_config())
        urls = {page.url for page in result.pages}
        assert f"{HOST}/tzgg/2026/0301.htm" in urls
        assert f"{HOST}/tzgg/2026/0302.htm" in urls

    def test_reports_count_and_urls(self):
        result = _sampler(StubFetcher()).sample(_config())
        assert result.sampled_count == len(result.pages)
        assert all(page.url.startswith(HOST) for page in result.pages)

    def test_resolves_relative_and_absolute_links_alike(self):
        result = _sampler(StubFetcher()).sample(_config())
        assert f"{HOST}/tzgg/2026/0303.htm" in {p.url for p in result.pages}

    def test_deduplicates_links(self):
        result = _sampler(StubFetcher()).sample(_config())
        urls = [p.url for p in result.pages]
        assert len(urls) == len(set(urls))

    def test_skips_offsite_links(self):
        result = _sampler(StubFetcher()).sample(_config())
        assert not any("other.example.cn" in p.url for p in result.pages)

    def test_ignores_pure_anchors(self):
        result = _sampler(StubFetcher()).sample(_config())
        assert not any(p.url.endswith("#top") for p in result.pages)

    def test_honours_the_sample_ceiling(self):
        config = _config(sampling={"max_samples": 2})
        result = _sampler(StubFetcher()).sample(config)
        assert len(result.pages) <= 2

    def test_default_ceiling_applies_without_explicit_config(self):
        # 规格要求默认有上限，避免对目标站点造成压力
        assert _config().sampling.max_samples > 0

    def test_deny_patterns_exclude_links(self):
        config = _config(deny_patterns=[r"0302\.htm$"])
        result = _sampler(StubFetcher()).sample(config)
        assert not any(p.url.endswith("0302.htm") for p in result.pages)

    def test_allow_patterns_restrict_links(self):
        config = _config(allow_patterns=[r"/tzgg/2026/0301"])
        result = _sampler(StubFetcher()).sample(config)
        assert {p.url for p in result.pages} == {f"{HOST}/tzgg/2026/0301.htm"}


class TestDetailLinkIdentification:
    """回归：真实栏目页的全站导航排在文章列表之前，曾把采样配额吃光，
    导致一个详情页都采不到（对 ccs.fjnu.edu.cn 实测发现）。"""

    NAV_LINKS = "".join(
        f'<a href="/{name}/list.htm">{name}</a>' for name in
        ("xygk", "xyjj", "xrld", "zzjg", "yxrc", "kydt", "zsgz", "bkzs", "yjszs", "txdt")
    )
    ARTICLES = "".join(
        f'<li><a href="/db/a4/c15003a4494{i:02d}/page.htm">通知{i}</a></li>' for i in range(6)
    )
    COLUMN_PAGE = f"""<html><body>
      <div class="nav-menu">{NAV_LINKS}</div>
      <div class="list-right"><ul class="news">{ARTICLES}</ul></div>
      <div class="footer"><a href="/about/list.htm">关于我们</a></div>
    </body></html>"""

    def _sample(self, max_samples):
        fetcher = StubFetcher(pages={f"{HOST}/kydt/list.htm": self.COLUMN_PAGE})
        return _sampler(fetcher).sample(
            _config(entry_urls=[f"{HOST}/kydt/list.htm"],
                    sampling={"max_samples": max_samples})
        )

    def test_scarce_quota_goes_to_detail_pages_not_navigation(self):
        urls = [p.url for p in self._sample(3).pages]
        assert len(urls) == 3
        assert all("/page.htm" in url for url in urls), urls

    def test_navigation_links_are_excluded(self):
        urls = {p.url for p in self._sample(20).pages}
        assert not any(url.endswith("/xygk/list.htm") for url in urls)

    def test_footer_links_are_excluded(self):
        urls = {p.url for p in self._sample(20).pages}
        assert not any("/about/" in url for url in urls)

    def test_detail_pages_are_all_collected_when_quota_allows(self):
        urls = [p.url for p in self._sample(20).pages]
        assert sum("/page.htm" in url for url in urls) == 6

    def test_page_entirely_inside_a_noise_container_still_yields_links(self):
        # 少数站点把整页包在 class="wrapper-nav" 之类的容器里；不能因此一无所获
        page = f'<html><body><div class="nav"><a href="/a/1.htm">甲</a>' \
               f'<a href="/a/2.htm">乙</a></div></body></html>'
        fetcher = StubFetcher(pages={f"{HOST}/kydt/list.htm": page})
        result = _sampler(fetcher).sample(_config(entry_urls=[f"{HOST}/kydt/list.htm"]))
        assert len(result.pages) == 2


class TestRobotsCompliance:
    def test_disallowed_url_is_never_requested(self):
        fetcher = StubFetcher()
        _sampler(fetcher).sample(_config(entry_urls=[f"{HOST}/tzgg.htm"]))
        assert f"{HOST}/admin/secret.htm" not in fetcher.requested

    def test_disallowed_url_is_reported_with_its_rule(self):
        result = _sampler(StubFetcher()).sample(_config())
        blocked = [s for s in result.skipped if "/admin/" in s.url]
        assert blocked and "/admin/" in blocked[0].rule
        assert blocked[0].reason == "robots"

    def test_disallowed_entry_is_not_requested(self):
        fetcher = StubFetcher(robots="User-agent: *\nDisallow: /\n")
        result = _sampler(fetcher).sample(_config())
        assert fetcher.requested == [f"{HOST}/robots.txt"]
        assert result.pages == ()
        assert result.skipped

    def test_robots_fetched_before_any_page(self):
        fetcher = StubFetcher()
        _sampler(fetcher).sample(_config())
        assert fetcher.requested[0].endswith("/robots.txt")


class TestRateLimiting:
    def test_throttle_is_applied_per_request(self):
        waits = []
        clock = iter([0.0] * 50)
        throttle = DomainThrottle(2.0, clock=lambda: next(clock), sleep=waits.append)
        sampler = Sampler(
            StubFetcher(), contact="mailto:a@b.cn", throttle=throttle
        )
        sampler.sample(_config())
        # 第一次请求无需等待，之后每次都受最小间隔约束
        assert waits and all(wait == pytest.approx(2.0) for wait in waits)


class TestFailureIsolation:
    def test_entry_http_error_is_reported_and_others_continue(self):
        second = f"{HOST}/other.htm"
        fetcher = StubFetcher(
            pages={f"{HOST}/tzgg.htm": LIST_PAGE, second: LIST_PAGE},
            statuses={f"{HOST}/tzgg.htm": 500},
        )
        result = _sampler(fetcher).sample(
            _config(entry_urls=[f"{HOST}/tzgg.htm", second])
        )
        assert any("500" in f.reason for f in result.failures)
        assert result.pages  # 第二个入口仍然产出了样本

    def test_entry_timeout_is_reported(self):
        fetcher = StubFetcher(errors={f"{HOST}/tzgg.htm": "连接超时"})
        result = _sampler(fetcher).sample(_config())
        assert any("超时" in f.reason for f in result.failures)

    def test_detail_page_failure_does_not_abort_sampling(self):
        fetcher = StubFetcher(statuses={f"{HOST}/tzgg/2026/0301.htm": 404})
        result = _sampler(fetcher).sample(_config())
        assert any("404" in f.reason for f in result.failures)
        assert result.pages

    def test_result_is_immutable(self):
        result = _sampler(StubFetcher()).sample(_config())
        with pytest.raises(Exception):
            result.pages = ()
