"""页面模板聚类。

site-structure 规格：聚类 MUST 同时依据页面结构特征与 URL 形状，二者一致
方可归为一类。每个模板 MUST 记录其样本 URL 与覆盖的栏目。

夹具里的文章详情页与教师个人页 URL 形状完全相同而结构不同——这正是"取
合取"所针对的情形，只看 URL 必然误并。
"""

import json
from pathlib import Path

import pytest

from kbwb.structure.templates import (
    PageSample,
    PageTemplate,
    cluster_templates,
    structure_fingerprint,
    similarity,
    url_shape,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ccs"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


def _sample(name, column="默认栏目"):
    return PageSample(
        url=MANIFEST[name]["source_url"],
        html=(FIXTURES / f"{name}.html").read_text(encoding="utf-8"),
        column=column,
    )


class TestUrlShape:
    def test_digits_are_normalised(self):
        assert url_shape("https://x.cn/a7/60/c16807a436064/page.htm") == "/#/#/#/page.htm"

    def test_literal_segments_kept(self):
        assert url_shape("https://x.cn/kydt/list.htm") == "/kydt/list.htm"

    def test_query_is_ignored(self):
        assert url_shape("https://x.cn/a/b.htm?p=2") == url_shape("https://x.cn/a/b.htm")

    def test_different_columns_differ(self):
        assert url_shape("https://x.cn/kydt/list.htm") != url_shape("https://x.cn/zsgz/list.htm")


class TestFingerprint:
    def test_same_page_same_fingerprint(self):
        html = (FIXTURES / "article.html").read_text(encoding="utf-8")
        assert structure_fingerprint(html) == structure_fingerprint(html)

    def test_fingerprint_is_not_empty(self):
        assert structure_fingerprint((FIXTURES / "article.html").read_text(encoding="utf-8"))

    def test_unparseable_html_yields_empty_fingerprint(self):
        assert structure_fingerprint("<<<not html") == frozenset()

    def test_identical_structure_scores_one(self):
        fp = structure_fingerprint((FIXTURES / "article.html").read_text(encoding="utf-8"))
        assert similarity(fp, fp) == pytest.approx(1.0)

    def test_disjoint_structures_score_zero(self):
        assert similarity(frozenset({"a"}), frozenset({"b"})) == 0.0

    def test_two_empty_fingerprints_are_not_similar(self):
        # 两个都解析失败的页面不应因此被判为同一模板
        assert similarity(frozenset(), frozenset()) == 0.0

    def test_content_differences_do_not_change_structure(self):
        base = "<html><body><div class='c'><p>%s</p></div></body></html>"
        assert structure_fingerprint(base % "甲乙丙") == structure_fingerprint(base % "丁戊己庚辛")


class TestClustering:
    def test_same_page_twice_is_one_template(self):
        samples = [_sample("article"), _sample("article")]
        assert len(cluster_templates(samples)) == 1

    def test_article_and_person_are_separate_templates(self):
        """URL 形状相同、结构不同——只看 URL 会误并，这里必须分开。"""
        templates = cluster_templates([_sample("article"), _sample("person")])
        assert len(templates) == 2

    def test_url_shape_alone_would_have_merged_them(self):
        # 证明上一条测的确实是合取的价值，而非 URL 恰好不同
        assert url_shape(MANIFEST["article"]["source_url"]) == url_shape(
            MANIFEST["person"]["source_url"]
        )

    def test_same_structure_different_url_shape_stays_separate(self):
        html = (FIXTURES / "article.html").read_text(encoding="utf-8")
        samples = [
            PageSample(url="https://x.cn/a7/60/c16807a436064/page.htm", html=html, column="甲"),
            PageSample(url="https://x.cn/news/2026/03/detail.htm", html=html, column="乙"),
        ]
        assert len(cluster_templates(samples)) == 2

    def test_list_and_article_are_separate(self):
        templates = cluster_templates([_sample("list"), _sample("article")])
        assert len(templates) == 2

    def test_empty_input_yields_no_templates(self):
        assert cluster_templates([]) == ()

    def test_threshold_controls_granularity(self):
        samples = [_sample("article"), _sample("person")]
        # 阈值降到极低时，任何有交集的页面都会被并到一起
        assert len(cluster_templates(samples, threshold=0.01)) == 1


class TestTemplateRecord:
    def test_template_records_sample_urls(self):
        template = cluster_templates([_sample("article")])[0]
        assert MANIFEST["article"]["source_url"] in template.sample_urls

    def test_template_records_covered_columns(self):
        samples = [_sample("article", column="科研动态"), _sample("article", column="通知公告")]
        template = cluster_templates(samples)[0]
        assert set(template.columns) == {"科研动态", "通知公告"}

    def test_template_has_a_stable_identifier(self):
        samples = [_sample("article"), _sample("person")]
        first = {t.template_id for t in cluster_templates(samples)}
        second = {t.template_id for t in cluster_templates(samples)}
        assert first == second

    def test_template_records_url_shape(self):
        template = cluster_templates([_sample("article")])[0]
        assert template.url_shape == "/#/#/#/page.htm"

    def test_template_is_immutable(self):
        template = cluster_templates([_sample("article")])[0]
        with pytest.raises(Exception):
            template.template_id = "x"

    def test_templates_are_ordered_by_coverage(self):
        samples = [_sample("article", "甲"), _sample("article", "乙"), _sample("person", "丙")]
        templates = cluster_templates(samples)
        assert isinstance(templates[0], PageTemplate)
        assert len(templates[0].sample_urls) >= len(templates[-1].sample_urls)


class TestChromeExclusion:
    """回归：指纹若含全站页头页脚导航，文章页与人物页会被误并。

    同一站点的所有页面共享这套外壳，它在标记集合里占大头，把内容区的差异
    淹没。实测排除前相似度高于阈值，排除后降到 0.64。
    """

    def _naive_fingerprint(self, html):
        """把外壳一并计入的指纹——即修正前的做法。"""
        import lxml.html

        document = lxml.html.fromstring(html)
        return frozenset(
            f"{n.tag}.{(n.get('class') or '').split()[0]}" if n.get("class") else n.tag
            for n in document.iter()
            if isinstance(n.tag, str)
        )

    def _html(self, name):
        return (FIXTURES / f"{name}.html").read_text(encoding="utf-8")

    def test_including_chrome_would_merge_them(self):
        naive = similarity(
            self._naive_fingerprint(self._html("article")),
            self._naive_fingerprint(self._html("person")),
        )
        assert naive >= 0.7, f"含外壳的相似度 {naive:.2f} 未超阈值，本回归的前提已不成立"

    def test_excluding_chrome_separates_them(self):
        actual = similarity(
            structure_fingerprint(self._html("article")),
            structure_fingerprint(self._html("person")),
        )
        assert actual < 0.7, f"排除外壳后相似度仍达 {actual:.2f}"

    def test_margin_is_recorded(self):
        """余量不大（实测 0.64 对阈值 0.70）。余量若消失需要重新审视判据。"""
        actual = similarity(
            structure_fingerprint(self._html("article")),
            structure_fingerprint(self._html("person")),
        )
        assert 0.5 <= actual < 0.7, f"相似度 {actual:.2f} 越出已知区间，判据可能已变"


class TestSamplingPlan:
    """按模板采样不重复：多个栏目归为同一模板时只采样一次。"""

    def test_first_candidate_of_a_shape_is_sampled(self):
        from kbwb.structure.templates import should_sample

        assert should_sample((), "https://x.cn/a7/60/c16807a436064/page.htm", samples_per_template=2)

    def test_shape_already_sampled_enough_is_skipped(self):
        from kbwb.structure.templates import should_sample

        template = PageTemplate(
            template_id="t1",
            url_shape="/#/#/#/page.htm",
            sample_urls=("https://x.cn/a7/60/c16807a436064/page.htm", "https://x.cn/b5/6b/c15003a439659/page.htm"),
            columns=("科研动态",),
        )
        assert not should_sample((template,), "https://x.cn/94/68/c15003a431208/page.htm", samples_per_template=2)

    def test_shape_under_quota_is_still_sampled(self):
        from kbwb.structure.templates import should_sample

        template = PageTemplate(
            template_id="t1",
            url_shape="/#/#/#/page.htm",
            sample_urls=("https://x.cn/a7/60/c16807a436064/page.htm",),
            columns=("科研动态",),
        )
        assert should_sample((template,), "https://x.cn/94/68/c15003a431208/page.htm", samples_per_template=3)

    def test_new_shape_is_sampled_even_when_others_are_full(self):
        from kbwb.structure.templates import should_sample

        template = PageTemplate(
            template_id="t1",
            url_shape="/#/#/#/page.htm",
            sample_urls=("a", "b"),
            columns=("甲",),
        )
        assert should_sample((template,), "https://x.cn/info/9.htm", samples_per_template=2)

    def test_second_column_with_same_shape_adds_no_requests(self):
        """两个栏目共用模板时，第二个栏目的候选不再触发抓取。"""
        from kbwb.structure.templates import should_sample

        samples = [_sample("article", column="科研动态")]
        templates = cluster_templates(samples)
        another = "https://ccs.fjnu.edu.cn/b5/6b/c15003a439659/page.htm"
        assert not should_sample(templates, another, samples_per_template=1)
