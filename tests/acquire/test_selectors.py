"""候选选择器推断。

site-analysis 规格：
- 每个字段返回按命中率**降序**的候选列表，每个候选标明在多少个样本上成功
  抽取到非空内容；
- 某字段在所有样本中都定位不到时，返回**空候选列表并标记为"需人工指定"**，
  而不是给出一个命中率为零的选择器。
"""

import pytest

from kbwb.acquire.sampling import SampledPage
from kbwb.acquire.selectors import FIELDS, infer_selectors

NAV = """
<div id="header"><a href="/">首页</a><a href="/tzgg.htm">通知公告</a></div>
<div class="sidebar"><a href="/a">短</a></div>
"""
FOOTER = '<div class="footer">版权所有 &copy; 某某大学</div>'


def _page(url, *, body, title="关于办理学籍证明的通知", date="2026-03-01",
          department=None, attachment=None):
    dept_html = f'<span class="dept">{department}</span>' if department else ""
    att_html = (
        f'<div class="attach"><a href="{attachment}">申请表下载</a></div>'
        if attachment
        else ""
    )
    return SampledPage(
        url=url,
        html=f"""<html><body>
          {NAV}
          <div class="content">
            <h1 class="title">{title}</h1>
            <span class="date">{date}</span>
            {dept_html}
            <div class="article">{body}</div>
            {att_html}
          </div>
          {FOOTER}
        </body></html>""",
    )


LONG_BODY = "办理学籍证明需携带身份证原件及复印件各一份，到教务处窗口提交申请。" * 6


@pytest.fixture
def pages():
    return [
        _page(f"https://x.edu.cn/{i}.htm", body=LONG_BODY, attachment="/f/申请表.pdf")
        for i in range(3)
    ]


class TestFieldCoverage:
    def test_all_fields_present_in_result(self, pages):
        result = infer_selectors(pages)
        assert set(result.fields) == set(FIELDS)

    def test_reports_sample_count(self, pages):
        assert infer_selectors(pages).samples == 3


class TestRanking:
    def test_candidates_sorted_by_hit_rate_descending(self, pages):
        candidates = infer_selectors(pages).fields["title"].candidates
        rates = [c.hit_rate for c in candidates]
        assert rates == sorted(rates, reverse=True)

    def test_top_body_candidate_targets_the_article(self, pages):
        top = infer_selectors(pages).fields["body"].candidates[0]
        assert "article" in top.selector or "content" in top.selector

    def test_candidate_reports_hits_and_samples(self, pages):
        top = infer_selectors(pages).fields["title"].candidates[0]
        assert top.hits == 3
        assert top.samples == 3
        assert top.hit_rate == pytest.approx(1.0)

    def test_partial_hit_rate_is_reflected(self):
        pages = [
            _page("https://x.edu.cn/1.htm", body=LONG_BODY, department="教务处"),
            _page("https://x.edu.cn/2.htm", body=LONG_BODY, department="学生处"),
            _page("https://x.edu.cn/3.htm", body=LONG_BODY),
        ]
        top = infer_selectors(pages).fields["department"].candidates[0]
        assert top.hits == 2
        assert top.hit_rate == pytest.approx(2 / 3)

    def test_no_zero_hit_candidates_returned(self, pages):
        for inference in infer_selectors(pages).fields.values():
            assert all(c.hit_rate > 0 for c in inference.candidates)


class TestTextDensity:
    def test_navigation_is_not_chosen_as_body(self, pages):
        top = infer_selectors(pages).fields["body"].candidates[0].selector
        assert "header" not in top and "sidebar" not in top

    def test_footer_is_not_chosen_as_body(self, pages):
        top = infer_selectors(pages).fields["body"].candidates[0].selector
        assert "footer" not in top

    def test_short_text_blocks_are_not_body_candidates(self, pages):
        selectors = {c.selector for c in infer_selectors(pages).fields["body"].candidates}
        assert not any("date" in s for s in selectors)


class TestExtractionFaithfulness:
    """回归：命中率必须按真实抽取路径（取第一个匹配元素）计算。

    对 ccs.fjnu.edu.cn 实测时，department 被推成 div.container 且标 100%，
    实际抽出来是整条导航栏——页面上确有某个 div.container 满足条件，但
    抽取取到的是排在前面的那个导航壳。
    """

    # 第一个 div.box 复刻真实页面的导航壳：文本长到不满足部门条件，
    # 但页面上另有一个 div.box 满足——旧实现据此把 div.box 记成 100%。
    NAV_TEXT = (
        "管理入口 旧版回顾 首页 学院概况 学院简介 现任领导 机构设置 师资队伍 "
        "优秀人才 网络空间安全系 计算机科学与技术系 软件工程系 实验教学中心"
    )
    AMBIGUOUS = """<html><body>
      <div class="box">%s</div>
      <div class="box"><span class="dept">教务处</span></div>
      <div class="main">%s</div>
    </body></html>""" % (NAV_TEXT, "正文内容需要足够长才会被判定为正文区域。" * 8)

    def _pages(self):
        return [SampledPage(url=f"https://x.edu.cn/{i}", html=self.AMBIGUOUS) for i in range(3)]

    def test_ambiguous_selector_is_not_credited(self):
        """div.box 的第一个匹配是导航壳，不应被记为 department 的命中。"""
        candidates = infer_selectors(self._pages()).fields["department"].candidates
        assert "div.box" not in {c.selector for c in candidates}

    def test_unambiguous_selector_is_credited(self):
        top = infer_selectors(self._pages()).fields["department"].best
        assert top is not None and top.selector == "span.dept"

    def test_every_candidate_actually_extracts(self):
        """草稿里的每个候选，用它去抽都必须真能拿到该字段。"""
        import lxml.html

        result = infer_selectors(self._pages())
        document = lxml.html.fromstring(self.AMBIGUOUS)
        for name in ("title", "date", "department"):
            for candidate in result.fields[name].candidates:
                assert document.cssselect(candidate.selector), candidate.selector


class TestBodyDensity:
    """回归：正文评分用文本密度，而非原始文本长度。

    只按长度排序时最外层容器总是胜出——它包住整页，文本自然最多。
    """

    WRAPPER = """<html><body>
      <div class="wrapper">
        <div class="menu-col">%s</div>
        <div class="article">%s</div>
      </div>
    </body></html>""" % (
        "".join(f'<a href="/c{i}/list.htm">栏目名称{i}</a>' for i in range(40)),
        "办理学籍证明需携带身份证原件及复印件各一份，到教务处窗口提交申请。" * 4,
    )

    def test_article_outranks_the_outer_wrapper(self):
        pages = [SampledPage(url=f"https://x.edu.cn/{i}", html=self.WRAPPER) for i in range(3)]
        top = infer_selectors(pages).fields["body"].best.selector
        assert top == "div.article", top


class TestManualFallback:
    def test_absent_field_is_marked_manual(self, pages):
        # 样本中没有任何部门信息
        inference = infer_selectors(pages).fields["department"]
        assert inference.candidates == ()
        assert inference.needs_manual is True

    def test_present_field_is_not_marked_manual(self, pages):
        assert infer_selectors(pages).fields["title"].needs_manual is False

    def test_empty_sample_set_marks_every_field_manual(self):
        result = infer_selectors([])
        assert result.samples == 0
        assert all(inference.needs_manual for inference in result.fields.values())

    def test_unparseable_html_does_not_abort(self, pages):
        broken = [SampledPage(url="https://x.edu.cn/bad", html="<<<not html")] + pages
        result = infer_selectors(broken)
        assert result.fields["title"].candidates


class TestSpecificFields:
    def test_date_field_detected(self, pages):
        assert "date" in infer_selectors(pages).fields["date"].candidates[0].selector

    def test_attachment_selector_targets_the_link(self, pages):
        top = infer_selectors(pages).fields["attachment"].candidates[0].selector
        assert "a" in top

    def test_attachment_selector_uses_the_extension_actually_present(self, pages):
        """回归：曾对配置中的每个扩展名都生成候选，导致草稿给出站点上
        并不存在的扩展名（.pdf 的样本推出 a[href$=".docx"]），一抓即空。"""
        candidates = infer_selectors(pages).fields["attachment"].candidates
        assert all(".pdf" in c.selector for c in candidates)
        assert candidates[0].selector.endswith('a[href$=".pdf"]')

    def test_multiple_extensions_each_get_their_own_candidate(self):
        mixed = [
            _page("https://x.edu.cn/1.htm", body=LONG_BODY, attachment="/f/a.pdf"),
            _page("https://x.edu.cn/2.htm", body=LONG_BODY, attachment="/f/b.docx"),
        ]
        selectors = {c.selector for c in infer_selectors(mixed).fields["attachment"].candidates}
        assert any(".pdf" in s for s in selectors)
        assert any(".docx" in s for s in selectors)

    def test_attachment_extension_list_is_honoured(self, pages):
        result = infer_selectors(pages, attachment_extensions=("doc",))
        assert result.fields["attachment"].needs_manual is True

    def test_various_date_formats_recognised(self):
        pages = [
            _page("https://x.edu.cn/1.htm", body=LONG_BODY, date="2026年3月1日"),
            _page("https://x.edu.cn/2.htm", body=LONG_BODY, date="2026/03/01"),
        ]
        assert infer_selectors(pages).fields["date"].candidates


class TestImmutability:
    def test_result_is_frozen(self, pages):
        result = infer_selectors(pages)
        with pytest.raises(Exception):
            result.samples = 99

    def test_candidate_is_frozen(self, pages):
        candidate = infer_selectors(pages).fields["title"].candidates[0]
        with pytest.raises(Exception):
            candidate.hit_rate = 0.0
