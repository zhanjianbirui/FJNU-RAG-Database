"""页面分区识别的契约。

site-structure 规格：分区类型取自固定集合，每个分区 MUST 携带定位它的
选择器与置信度，MUST NOT 只给布尔判定。正文识别失败时置信度为零并标记，
但该页面仍可参与模板归类——结构信息依然有效。
"""

from pathlib import Path

import lxml.html
import pytest

from kbwb.structure.regions import (
    REGION_TYPES,
    PageRegions,
    Region,
    RegionType,
    segment_page,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ccs"


def _html(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")


class TestContractShape:
    def test_region_types_are_a_closed_set(self):
        assert set(REGION_TYPES) == {
            RegionType.NAVIGATION,
            RegionType.ARTICLE_BODY,
            RegionType.ARTICLE_META,
            RegionType.ARTICLE_LIST,
            RegionType.PAGINATION,
            RegionType.ATTACHMENT,
            RegionType.FOOTER,
            RegionType.SIDEBAR,
        }

    def test_every_region_carries_selector_and_confidence(self):
        result = segment_page(_html("article"))
        for region in result.regions:
            assert isinstance(region.selector, str) and region.selector
            assert 0.0 <= region.confidence <= 1.0

    def test_region_type_is_from_the_closed_set(self):
        for region in segment_page(_html("article")).regions:
            assert region.type in REGION_TYPES

    def test_confidence_is_not_a_boolean_in_disguise(self):
        """置信度必须是真实的分级信号，不能只取 0 或 1。"""
        values = {r.confidence for r in segment_page(_html("article")).regions}
        assert values - {0.0, 1.0}, f"所有置信度都是 0/1：{values}"

    def test_regions_are_immutable(self):
        region = segment_page(_html("article")).regions[0]
        with pytest.raises(Exception):
            region.confidence = 0.5

    def test_result_is_immutable(self):
        result = segment_page(_html("article"))
        with pytest.raises(Exception):
            result.regions = ()

    def test_lookup_by_type(self):
        result = segment_page(_html("article"))
        body = result.of(RegionType.ARTICLE_BODY)
        assert body is None or isinstance(body, Region)

    def test_result_is_a_page_regions(self):
        assert isinstance(segment_page(_html("article")), PageRegions)


class TestUnparseableInput:
    def test_broken_html_does_not_raise(self):
        assert isinstance(segment_page("<<<not html"), PageRegions)

    def test_empty_document_yields_no_regions(self):
        assert segment_page("").regions == ()


class TestArticleBody:
    """正文分区。规格要求抽出的正文不含标题、发布时间与导航。"""

    def _body(self, name="article"):
        return segment_page(_html(name)).of(RegionType.ARTICLE_BODY)

    def test_body_region_present(self):
        assert self._body() is not None

    def test_selector_uniquely_locates_the_element(self):
        """选择器的首个匹配必须就是正文元素——组 3 的教训。"""
        doc = lxml.html.fromstring(_html("article"))
        matched = doc.cssselect(self._body().selector)
        assert matched, "选择器匹配不到任何元素"
        text = " ".join(matched[0].text_content().split())
        assert "为进一步提升国家自然科学基金" in text

    def test_body_excludes_navigation(self):
        doc = lxml.html.fromstring(_html("article"))
        text = " ".join(doc.cssselect(self._body().selector)[0].text_content().split())
        assert "学院概况" not in text and "师资队伍" not in text

    def test_body_excludes_title_and_publish_time(self):
        doc = lxml.html.fromstring(_html("article"))
        text = " ".join(doc.cssselect(self._body().selector)[0].text_content().split())
        assert "发布时间" not in text
        assert not text.startswith("学院举办2026年度")

    def test_body_keeps_the_whole_article(self):
        doc = lxml.html.fromstring(_html("article"))
        text = " ".join(doc.cssselect(self._body().selector)[0].text_content().split())
        assert len(text) > 400, f"正文只剩 {len(text)} 字，疑似被截断"

    def test_confidence_reflects_purity(self):
        # 正文容器几乎全是正文时，置信度应接近上限
        assert self._body().confidence >= 0.9


class TestBodyDetectionFailure:
    """规格：找不到可靠正文区时，置信度为零并标记，但页面仍可继续被处理。"""

    NO_ARTICLE = """<html><body>
      <div class="nav-menu"><a href="/a">栏目一</a><a href="/b">栏目二</a></div>
      <div class="footer">版权所有</div>
    </body></html>"""

    def test_confidence_is_zero(self):
        assert segment_page(self.NO_ARTICLE).of(RegionType.ARTICLE_BODY).confidence == 0.0

    def test_failure_is_marked(self):
        assert segment_page(self.NO_ARTICLE).of(RegionType.ARTICLE_BODY).note

    def test_page_still_yields_a_result(self):
        """不得因正文缺失而中止——该页面的结构信息仍然有效，需能继续参与后续处理。"""
        result = segment_page(self.NO_ARTICLE)
        assert isinstance(result, PageRegions)
        assert result.regions

    def test_document_remains_parseable_for_downstream(self):
        assert lxml.html.fromstring(self.NO_ARTICLE) is not None


def _region_text(name, region):
    doc = lxml.html.fromstring(_html(name))
    matched = doc.cssselect(region.selector)
    return " ".join(matched[0].text_content().split()) if matched else ""


class TestStructuralRegions:
    """导航、页脚、侧栏。它们同样带 class，仅靠标签名无法与正文区分。"""

    def test_navigation_found_on_article_page(self):
        region = segment_page(_html("article")).of(RegionType.NAVIGATION)
        assert region is not None
        assert "学院概况" in _region_text("article", region)

    def test_navigation_is_not_the_body(self):
        result = segment_page(_html("article"))
        assert result.of(RegionType.NAVIGATION).selector != result.of(RegionType.ARTICLE_BODY).selector

    def test_footer_found(self):
        region = segment_page(_html("article")).of(RegionType.FOOTER)
        assert region is not None
        assert "版权" in _region_text("article", region) or "©" in _region_text("article", region)

    def test_sidebar_absent_is_reported_as_absent(self):
        # 该站点页面没有侧栏，不应凭空造出一个
        assert segment_page(_html("article")).of(RegionType.SIDEBAR) is None

    def test_navigation_confidence_exceeds_footer(self):
        """导航的链接密度高于页脚，置信度应体现这一差别而非一律给满分。"""
        result = segment_page(_html("article"))
        assert result.of(RegionType.NAVIGATION).confidence > result.of(RegionType.FOOTER).confidence

    def test_selectors_uniquely_locate(self):
        doc = lxml.html.fromstring(_html("article"))
        for region in segment_page(_html("article")).regions:
            matched = doc.cssselect(region.selector)
            assert matched, region.selector


class TestPagination:
    def test_pagination_found_on_list_page(self):
        region = segment_page(_html("list")).of(RegionType.PAGINATION)
        assert region is not None

    def test_pagination_region_contains_page_links(self):
        region = segment_page(_html("list")).of(RegionType.PAGINATION)
        doc = lxml.html.fromstring(_html("list"))
        hrefs = " ".join(a.get("href", "") for a in doc.cssselect(region.selector)[0].xpath(".//a"))
        assert "list2.htm" in hrefs

    def test_no_pagination_on_article_page(self):
        assert segment_page(_html("article")).of(RegionType.PAGINATION) is None


class TestArticleList:
    def test_list_region_found_on_list_page(self):
        region = segment_page(_html("list")).of(RegionType.ARTICLE_LIST)
        assert region is not None

    def test_list_region_is_not_the_navigation(self):
        result = segment_page(_html("list"))
        assert result.of(RegionType.ARTICLE_LIST).selector != result.of(RegionType.NAVIGATION).selector

    def test_list_region_holds_multiple_entries(self):
        region = segment_page(_html("list")).of(RegionType.ARTICLE_LIST)
        doc = lxml.html.fromstring(_html("list"))
        assert len(doc.cssselect(region.selector)[0].xpath(".//a[@href]")) >= 5

    def test_no_list_region_on_article_page(self):
        # 详情页不应被误判为含条目列表
        assert segment_page(_html("article")).of(RegionType.ARTICLE_LIST) is None


class TestArticleMeta:
    def test_meta_found_on_article_page(self):
        region = segment_page(_html("article")).of(RegionType.ARTICLE_META)
        assert region is not None

    def test_meta_contains_the_publish_date(self):
        region = segment_page(_html("article")).of(RegionType.ARTICLE_META)
        assert "2026-03-11" in _region_text("article", region)

    def test_meta_is_short(self):
        region = segment_page(_html("article")).of(RegionType.ARTICLE_META)
        assert len(_region_text("article", region)) <= 60


class TestAttachment:
    WITH_ATTACHMENT = """<html><body>
      <div class="nav-menu"><a href="/a">栏目</a></div>
      <div class="wp_articlecontent"><p>%s</p>
        <div class="attach"><a href="/f/申请表.pdf">学籍证明申请表</a>
                            <a href="/f/须知.docx">办理须知</a></div>
      </div>
    </body></html>""" % ("办理学籍证明需携带身份证原件及复印件各一份，到教务处窗口提交申请。" * 4)

    def test_attachment_region_found(self):
        assert segment_page(self.WITH_ATTACHMENT).of(RegionType.ATTACHMENT) is not None

    def test_attachment_region_holds_document_links(self):
        region = segment_page(self.WITH_ATTACHMENT).of(RegionType.ATTACHMENT)
        doc = lxml.html.fromstring(self.WITH_ATTACHMENT)
        hrefs = [a.get("href") for a in doc.cssselect(region.selector)[0].xpath(".//a[@href]")]
        assert any(h.endswith(".pdf") for h in hrefs)

    def test_no_attachment_region_when_absent(self):
        assert segment_page(_html("article")).of(RegionType.ATTACHMENT) is None


class TestArticleListScoping:
    """回归：条目列表区曾选中整个 body。

    页面的直接子元素也各自含链接，"条目最多者"因此落在最外层；
    人工核对真实首页夹具时发现。
    """

    def test_home_list_is_not_the_whole_body(self):
        region = segment_page(_html("home")).of(RegionType.ARTICLE_LIST)
        assert region is not None
        assert region.selector not in ("body", "html", "html > body")

    def test_home_list_is_the_actual_news_list(self):
        region = segment_page(_html("home")).of(RegionType.ARTICLE_LIST)
        doc = lxml.html.fromstring(_html("home"))
        element = doc.cssselect(region.selector)[0]
        entries = [c for c in element if isinstance(c.tag, str)]
        assert 5 <= len(entries) <= 30, f"条目数 {len(entries)} 不像一份新闻列表"

    def test_innermost_container_wins(self):
        """外层容器包住合格的内层容器时，真正的列表在里面。"""
        html = """<html><body><div class="outer">
          <ul class="inner">%s</ul>
        </div></body></html>""" % "".join(
            f'<li><a href="/a{i}.htm">条目{i}</a></li>' for i in range(8)
        )
        region = segment_page(html).of(RegionType.ARTICLE_LIST)
        assert region.selector == "ul.inner"

    def test_entries_must_share_a_tag(self):
        """形态各异的零散链接不构成条目列表。"""
        html = """<html><body><div class="mixed">
          <p><a href="/1">一</a></p><span><a href="/2">二</a></span>
          <h3><a href="/3">三</a></h3><em><a href="/4">四</a></em>
          <b><a href="/5">五</a></b><i><a href="/6">六</a></i>
        </div></body></html>"""
        assert segment_page(html).of(RegionType.ARTICLE_LIST) is None
