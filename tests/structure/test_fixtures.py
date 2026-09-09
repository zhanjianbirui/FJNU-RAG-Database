"""真实页面夹具的性质校验。

任务组 3 的三处缺陷全部因夹具过于理想化而未被单元测试捕获，直到对真实站点
实测才暴露。本组用例把"夹具必须复刻真实页面的复杂度"变成可执行的约束：
夹具若被替换成简化版本，这些用例会失败。
"""

import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import lxml.html
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ccs"
NAMES = ("home", "list", "article", "person", "article_multiblock")


def _doc(name):
    return lxml.html.fromstring((FIXTURES / name).with_suffix(".html").read_text(encoding="utf-8"))


def _shape(url: str) -> str:
    path = urlsplit(url).path
    return "/".join("#" if any(c.isdigit() for c in s) else s for s in path.split("/"))


@pytest.fixture(scope="module")
def manifest():
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


class TestProvenance:
    def test_every_fixture_present(self):
        for name in NAMES:
            assert (FIXTURES / f"{name}.html").is_file(), name

    def test_manifest_records_source_and_date(self, manifest):
        for name in NAMES:
            assert manifest[name]["source_url"].startswith("https://")
            assert manifest[name]["fetched"]

    def test_all_fixtures_parse(self):
        for name in NAMES:
            assert _doc(name) is not None


class TestRealWorldComplexity:
    """夹具必须带有让理想化样本漏掉缺陷的那些特征。"""

    def test_home_has_a_large_navigation(self):
        # 组 3 的缺陷之一是导航把采样配额吃光；夹具的导航必须足够大
        links = _doc("home").xpath("//a[@href]")
        assert len(links) >= 30, f"首页仅 {len(links)} 条链接，不足以复现导航挤占问题"

    def test_article_has_repeated_class_names(self):
        """同名 class 出现在多个元素上——组 3 的另一处缺陷正因此暴露：
        选择器匹配到多个元素时，抽取取的是第一个，未必是推断时看中的那个。"""
        classes = Counter(
            name
            for node in _doc("article").iter()
            if isinstance(node.tag, str)
            for name in (node.get("class") or "").split()
        )
        repeated = [name for name, count in classes.items() if count >= 2]
        assert repeated, "详情页夹具没有重复的 class 名，无法复现选择器歧义"

    def test_article_has_long_text_outside_the_body(self):
        """正文之外存在长文本块（导航栏等），否则按文本长度选正文也能蒙对，
        文本密度的必要性就测不出来。"""
        doc = _doc("article")
        body = doc.cssselect("div.wp_articlecontent")
        assert body, "夹具的正文容器已变，需重新采集夹具并更新此断言"
        body_text = " ".join(body[0].text_content().split())
        whole = " ".join(doc.text_content().split())
        outside = len(whole) - len(body_text)
        assert outside >= 150, f"正文外文本仅 {outside} 字，不足以体现密度判据"

    def test_list_page_has_pagination(self):
        html = (FIXTURES / "list.html").read_text(encoding="utf-8")
        assert "list2.htm" in html, "列表页夹具缺少分页链接，无法验证规模估算"


class TestTemplateSeparation:
    """文章页与人物页 URL 形状相同、结构不同——聚类必须能把它们分开。

    这是设计中"结构指纹与 URL 形状取合取"的直接依据：只看 URL 会误并。
    """

    def test_article_and_person_share_url_shape(self, manifest):
        assert _shape(manifest["article"]["source_url"]) == _shape(
            manifest["person"]["source_url"]
        )

    def test_article_and_person_differ_structurally(self):
        def fingerprint(name):
            return {
                f"{n.tag}.{(n.get('class') or '').split()[0] if n.get('class') else ''}"
                for n in _doc(name).iter()
                if isinstance(n.tag, str)
            }

        article, person = fingerprint("article"), fingerprint("person")
        overlap = len(article & person) / len(article | person)
        assert overlap < 0.9, f"两类页面结构过于相似（Jaccard {overlap:.2f}），无法验证聚类分离"
