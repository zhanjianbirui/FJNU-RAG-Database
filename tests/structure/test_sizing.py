"""页面类型判定与规模估算。

site-structure 规格：
- 至少区分列表页、单页与未知；无法判定时标记未知并说明原因，MUST NOT 猜测；
- 规模在不逐页翻遍的前提下估算，MUST 标明所采用的估算方式；
- 无法估算时标记未知，MUST NOT 给出推测值；
- 估算结果 MUST 以约数形式呈现，MUST NOT 显示为精确数字。
"""

from pathlib import Path

import pytest

from kbwb.structure.regions import segment_page
from kbwb.structure.sizing import (
    EstimateMethod,
    PageType,
    approximate,
    classify_page,
    estimate_size,
    probe_last_page,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ccs"


def _html(name):
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


class TestPageType:
    def test_list_page_recognised(self):
        assert classify_page(segment_page(_html("list"))).type is PageType.LIST

    def test_article_page_is_single(self):
        assert classify_page(segment_page(_html("article"))).type is PageType.SINGLE

    def test_home_with_entries_is_a_list(self):
        assert classify_page(segment_page(_html("home"))).type is PageType.LIST

    def test_person_page_is_single(self):
        """人物页有正文（教师简介）、无条目列表，属单页。"""
        assert classify_page(segment_page(_html("person"))).type is PageType.SINGLE

    NEITHER = """<html><body>
      <div class="nav-menu"><a href="/a">栏目一</a><a href="/b">栏目二</a></div>
      <div class="footer">版权所有</div>
    </body></html>"""

    def test_page_without_body_or_list_is_unknown(self):
        assert classify_page(segment_page(self.NEITHER)).type is PageType.UNKNOWN

    def test_unknown_carries_a_reason(self):
        assert classify_page(segment_page(self.NEITHER)).reason

    def test_pagination_presence_is_recorded(self):
        assert classify_page(segment_page(_html("list"))).has_pagination is True
        assert classify_page(segment_page(_html("article"))).has_pagination is False

    def test_pagination_raises_confidence(self):
        with_paging = classify_page(segment_page(_html("list")))
        without = classify_page(segment_page(_html("home")))
        assert with_paging.confidence > without.confidence

    def test_empty_page_is_unknown(self):
        assert classify_page(segment_page("")).type is PageType.UNKNOWN

    def test_result_is_immutable(self):
        result = classify_page(segment_page(_html("list")))
        with pytest.raises(Exception):
            result.type = PageType.SINGLE


class TestSizeFromText:
    """真实分页区原文：每页 10 记录 总共 104 记录 … 页码 1/11"""

    def test_real_list_page_total_items(self):
        estimate = estimate_size(_html("list"), entries_on_page=10)
        assert estimate.items == 104

    def test_method_is_recorded(self):
        estimate = estimate_size(_html("list"), entries_on_page=10)
        assert estimate.method is EstimateMethod.ITEM_COUNT_TEXT

    def test_page_count_also_captured(self):
        assert estimate_size(_html("list"), entries_on_page=10).pages == 11

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("共 420 条", 420),
            ("共有 88 条记录", 88),
            ("总共 104 记录", 104),
            ("总数：37", 37),
            ("共 256 篇", 256),
        ],
    )
    def test_item_count_phrasings(self, text, expected):
        html = f"<html><body><div class='wp_paging'>{text}<a href='/list2.htm'>2</a></div></body></html>"
        assert estimate_size(html, entries_on_page=10).items == expected

    def test_ordinary_text_is_not_mistaken_for_a_count(self):
        html = "<html><body><div class='c'><p>本次共 3 位专家出席会议。</p></div></body></html>"
        assert estimate_size(html, entries_on_page=10).method is EstimateMethod.UNKNOWN


class TestSizeFromPagination:
    def test_max_page_link_used_when_no_count_text(self):
        html = """<html><body><div class="wp_paging">
          <a href="/kydt/list2.htm">2</a><a href="/kydt/list11.htm">尾页</a>
        </div></body></html>"""
        estimate = estimate_size(html, entries_on_page=10)
        assert estimate.pages == 11
        assert estimate.items == 110
        assert estimate.method is EstimateMethod.PAGINATION_LINKS

    def test_single_page_column(self):
        html = "<html><body><div class='list'><ul><li><a href='/a.htm'>甲</a></li></ul></div></body></html>"
        estimate = estimate_size(html, entries_on_page=1)
        assert estimate.method is EstimateMethod.UNKNOWN


class TestUnknown:
    def test_no_signal_yields_unknown(self):
        assert estimate_size("<html><body><p>无</p></body></html>", entries_on_page=0).method \
            is EstimateMethod.UNKNOWN

    def test_unknown_has_no_invented_values(self):
        estimate = estimate_size("<html><body><p>无</p></body></html>", entries_on_page=0)
        assert estimate.items is None and estimate.pages is None

    def test_unknown_renders_as_unknown(self):
        estimate = estimate_size("<html><body><p>无</p></body></html>", entries_on_page=0)
        assert "未知" in estimate.display


class TestApproximateDisplay:
    """规格：以约数形式呈现，不得显示为精确数字。"""

    @pytest.mark.parametrize(("value", "shown"), [(104, "100"), (37, "30"), (1240, "1200"), (7, "7")])
    def test_rounding(self, value, shown):
        assert shown in approximate(value)

    def test_carries_an_approximation_marker(self):
        assert approximate(104).startswith("约")

    def test_real_estimate_is_not_exact(self):
        display = estimate_size(_html("list"), entries_on_page=10).display
        assert "104" not in display, f"精确条目数出现在展示文本中：{display}"
        assert "约" in display

    def test_display_names_the_method(self):
        assert "条数文本" in estimate_size(_html("list"), entries_on_page=10).display


class TestDoublingProbe:
    """仅有「下一页」而无页码时，以有限次试探定位末页。"""

    def _exists(self, last):
        return lambda page: page <= last

    def test_finds_the_last_page(self):
        assert probe_last_page(self._exists(11), budget=20).page == 11

    def test_uses_logarithmic_requests(self):
        result = probe_last_page(self._exists(1000), budget=40)
        assert result.page == 1000
        assert result.requests <= 25, f"用了 {result.requests} 次请求，不是对数量级"

    def test_single_page(self):
        assert probe_last_page(self._exists(1), budget=10).page == 1

    def test_budget_exhausted_reports_lower_bound(self):
        result = probe_last_page(self._exists(10_000), budget=4)
        assert result.exhausted is True
        assert result.page >= 1

    def test_requests_never_exceed_budget(self):
        result = probe_last_page(self._exists(10_000), budget=6)
        assert result.requests <= 6

    def test_nothing_exists(self):
        assert probe_last_page(lambda page: False, budget=10).page == 0


class TestAdjacentTextHazard:
    """回归：分页区文本曾取整棵子树的 text_content()，相邻文本因此粘连。

    「总数：37」紧挨着页码链接「2」被读成「总数：372」。
    """

    def test_count_adjacent_to_a_page_link(self):
        html = "<html><body><div class='wp_paging'>总数：37<a href='/list2.htm'>2</a></div></body></html>"
        assert estimate_size(html, entries_on_page=10).items == 37

    def test_count_with_surrounding_markup(self):
        html = ("<html><body><div class='wp_paging'>"
                "<span>共</span><b>420</b><span>条</span>"
                "<a href='/list42.htm'>尾页</a></div></body></html>")
        assert estimate_size(html, entries_on_page=10).items == 420
