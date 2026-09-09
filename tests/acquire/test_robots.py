"""robots.txt 解析与判定。

web-crawling 规格：被 robots 禁止的 URL MUST 跳过且**不发起请求**，并记入
跳过原因；site-analysis 规格进一步要求向用户报告**对应的 robots 规则**，
因此判定结果必须携带命中的那条指令，而不只是一个布尔值。
"""

import pytest

from kbwb.acquire.robots import RobotsGate, RobotsRules

UA = "kbwb/0.1 (+mailto:admin@example.edu.cn)"

SIMPLE = """
User-agent: *
Disallow: /admin/
Disallow: /search
Crawl-delay: 2
"""

GROUPED = """
User-agent: *
Disallow: /

User-agent: kbwb
Disallow: /private/
Allow: /
"""

ALLOW_OVERRIDE = """
User-agent: *
Disallow: /docs/
Allow: /docs/public/
"""

WILDCARDS = """
User-agent: *
Disallow: /*.pdf$
Disallow: /tmp/*/cache
"""


class TestParsing:
    def test_disallowed_path(self):
        rules = RobotsRules.parse(SIMPLE)
        assert rules.evaluate("/admin/users", UA).allowed is False

    def test_allowed_path(self):
        rules = RobotsRules.parse(SIMPLE)
        assert rules.evaluate("/tzgg/1.htm", UA).allowed is True

    def test_reports_the_matching_rule(self):
        decision = RobotsRules.parse(SIMPLE).evaluate("/admin/users", UA)
        assert "Disallow" in decision.rule and "/admin/" in decision.rule

    def test_allowed_decision_has_no_rule(self):
        assert RobotsRules.parse(SIMPLE).evaluate("/ok", UA).rule is None

    def test_prefix_matching_not_exact(self):
        # Disallow: /search 同样覆盖 /search?q=x 与 /searchpage
        assert RobotsRules.parse(SIMPLE).evaluate("/search?q=x", UA).allowed is False

    def test_crawl_delay_parsed(self):
        assert RobotsRules.parse(SIMPLE).crawl_delay(UA) == 2.0

    def test_missing_crawl_delay_is_none(self):
        assert RobotsRules.parse(GROUPED).crawl_delay(UA) is None

    def test_directives_are_case_insensitive(self):
        rules = RobotsRules.parse("user-agent: *\ndisallow: /x/")
        assert rules.evaluate("/x/y", UA).allowed is False

    def test_comments_and_blank_lines_ignored(self):
        rules = RobotsRules.parse("# 注释\n\nUser-agent: *\nDisallow: /x/  # 尾注")
        assert rules.evaluate("/x/y", UA).allowed is False

    def test_empty_document_allows_everything(self):
        assert RobotsRules.parse("").evaluate("/anything", UA).allowed is True

    def test_empty_disallow_means_allow_all(self):
        rules = RobotsRules.parse("User-agent: *\nDisallow:")
        assert rules.evaluate("/anything", UA).allowed is True


class TestGroupSelection:
    def test_specific_group_wins_over_wildcard(self):
        rules = RobotsRules.parse(GROUPED)
        assert rules.evaluate("/tzgg/1.htm", UA).allowed is True

    def test_specific_group_rules_apply(self):
        rules = RobotsRules.parse(GROUPED)
        assert rules.evaluate("/private/x", UA).allowed is False

    def test_other_agents_fall_back_to_wildcard(self):
        rules = RobotsRules.parse(GROUPED)
        assert rules.evaluate("/tzgg/1.htm", "SomeOtherBot/1.0").allowed is False


class TestLongestMatchWins:
    def test_allow_overrides_shorter_disallow(self):
        rules = RobotsRules.parse(ALLOW_OVERRIDE)
        assert rules.evaluate("/docs/public/a.htm", UA).allowed is True

    def test_disallow_still_applies_elsewhere(self):
        rules = RobotsRules.parse(ALLOW_OVERRIDE)
        assert rules.evaluate("/docs/private/a.htm", UA).allowed is False


class TestWildcards:
    def test_star_and_dollar(self):
        rules = RobotsRules.parse(WILDCARDS)
        assert rules.evaluate("/files/a.pdf", UA).allowed is False

    def test_dollar_anchors_the_end(self):
        rules = RobotsRules.parse(WILDCARDS)
        assert rules.evaluate("/files/a.pdf.htm", UA).allowed is True

    def test_star_in_the_middle(self):
        rules = RobotsRules.parse(WILDCARDS)
        assert rules.evaluate("/tmp/abc/cache", UA).allowed is False


class RecordingFetcher:
    """记录抓取过的 URL，用于断言被禁的 URL 确实没被请求。"""

    def __init__(self, pages=None, robots_status=200, robots_body=SIMPLE):
        self.pages = pages or {}
        self.robots_status = robots_status
        self.robots_body = robots_body
        self.requested = []

    def get(self, url):
        from kbwb.acquire.fetching import FetchResponse

        self.requested.append(url)
        if url.endswith("/robots.txt"):
            return FetchResponse(url=url, status=self.robots_status, text=self.robots_body)
        return FetchResponse(url=url, status=200, text=self.pages.get(url, "<html></html>"))


class TestRobotsGate:
    def test_disallowed_url_is_reported_with_its_rule(self):
        gate = RobotsGate(RecordingFetcher(), user_agent=UA)
        decision = gate.check("https://jwc.example.edu.cn/admin/x")
        assert decision.allowed is False
        assert "/admin/" in decision.rule

    def test_robots_is_fetched_once_per_origin(self):
        fetcher = RecordingFetcher()
        gate = RobotsGate(fetcher, user_agent=UA)
        gate.check("https://jwc.example.edu.cn/a")
        gate.check("https://jwc.example.edu.cn/b")
        assert fetcher.requested.count("https://jwc.example.edu.cn/robots.txt") == 1

    def test_separate_origins_fetch_separately(self):
        fetcher = RecordingFetcher()
        gate = RobotsGate(fetcher, user_agent=UA)
        gate.check("https://a.example.edu.cn/x")
        gate.check("https://b.example.edu.cn/x")
        assert len([u for u in fetcher.requested if u.endswith("robots.txt")]) == 2

    def test_missing_robots_allows_everything(self):
        gate = RobotsGate(RecordingFetcher(robots_status=404), user_agent=UA)
        assert gate.check("https://jwc.example.edu.cn/admin/x").allowed is True

    def test_server_error_fails_closed(self):
        """5xx 时无法确认许可，按禁止处理——合规上宁可少抓也不冒犯。"""
        gate = RobotsGate(RecordingFetcher(robots_status=503), user_agent=UA)
        decision = gate.check("https://jwc.example.edu.cn/x")
        assert decision.allowed is False

    def test_network_failure_fails_closed(self):
        class Failing:
            def get(self, url):
                from kbwb.acquire.fetching import FetchError

                raise FetchError("连接超时")

        assert RobotsGate(Failing(), user_agent=UA).check("https://x.example.cn/a").allowed is False

    def test_decision_names_the_robots_url(self):
        gate = RobotsGate(RecordingFetcher(), user_agent=UA)
        decision = gate.check("https://jwc.example.edu.cn/admin/x")
        assert decision.robots_url == "https://jwc.example.edu.cn/robots.txt"

    def test_crawl_delay_exposed_for_the_origin(self):
        gate = RobotsGate(RecordingFetcher(), user_agent=UA)
        assert gate.crawl_delay("https://jwc.example.edu.cn/a") == 2.0

    def test_non_http_url_rejected(self):
        gate = RobotsGate(RecordingFetcher(), user_agent=UA)
        with pytest.raises(ValueError):
            gate.check("ftp://jwc.example.edu.cn/a")
