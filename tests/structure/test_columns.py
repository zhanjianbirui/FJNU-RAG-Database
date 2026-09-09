"""栏目树探测。

site-structure 规格：每个栏目 MUST 携带可读名称、URL 与层级。名称无法从
站点获得时 MUST 标记为未知并提示需人工命名，MUST NOT 以 URL 片段充当名称。
导航不可识别时改以链接聚类得出候选。
"""

import json
from pathlib import Path

import pytest

from kbwb.structure.columns import (
    Column,
    ColumnTree,
    NAME_UNKNOWN,
    discover_columns,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ccs"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
HOME_URL = MANIFEST["home"]["source_url"]


def _home() -> str:
    return (FIXTURES / "home.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree() -> ColumnTree:
    return discover_columns(_home(), base_url=HOME_URL)


class TestNavigationDiscovery:
    def test_columns_are_found(self, tree):
        assert len(tree.columns) >= 10

    def test_names_are_human_readable_chinese(self, tree):
        names = {c.name for c in tree.columns}
        assert {"学院概况", "师资队伍", "科研动态"} <= names

    def test_urls_are_absolute(self, tree):
        assert all(c.url.startswith("https://") for c in tree.columns)

    def test_urls_stay_on_the_same_host(self, tree):
        assert all("ccs.fjnu.edu.cn" in c.url for c in tree.columns)

    def test_no_duplicate_urls(self, tree):
        urls = [c.url for c in tree.columns]
        assert len(urls) == len(set(urls))

    def test_home_link_is_not_a_column(self, tree):
        # 指向站点首页的链接是入口而非栏目
        assert not any(c.url.rstrip("/").endswith("main.htm") for c in tree.columns)

    def test_offsite_links_excluded(self, tree):
        assert not any("net.fjnu.edu.cn" in c.url for c in tree.columns)


class TestHierarchy:
    def test_tree_has_more_than_one_level(self, tree):
        assert max(c.depth for c in tree.columns) >= 1

    def test_top_level_columns_have_no_parent(self, tree):
        assert all(c.parent_url is None for c in tree.columns if c.depth == 0)

    def test_child_points_at_an_existing_parent(self, tree):
        urls = {c.url for c in tree.columns}
        for column in tree.columns:
            if column.parent_url is not None:
                assert column.parent_url in urls

    def test_children_helper(self, tree):
        parents = [c for c in tree.columns if tree.children_of(c.url)]
        assert parents, "没有任何栏目带子栏目，层级未被还原"

    def test_roots_helper(self, tree):
        assert tree.roots() and all(c.depth == 0 for c in tree.roots())


class TestNaming:
    def test_names_are_not_url_fragments(self, tree):
        for column in tree.columns:
            assert column.name not in column.url
            assert not column.name.endswith(".htm")

    def test_named_columns_are_not_flagged(self, tree):
        assert all(not c.needs_manual_name for c in tree.columns if c.name != NAME_UNKNOWN)

    def test_unnamed_link_is_flagged_not_invented(self):
        """链接无文本时标记为未知，而不是拿 URL 片段凑一个名字。"""
        html = """<html><body><nav>
          <a href="/tzgg/list.htm"></a>
          <a href="/kydt/list.htm">科研动态</a>
        </nav></body></html>"""
        tree = discover_columns(html, base_url="https://x.edu.cn/main.htm")
        unnamed = [c for c in tree.columns if c.url.endswith("/tzgg/list.htm")]
        assert unnamed and unnamed[0].name == NAME_UNKNOWN
        assert unnamed[0].needs_manual_name is True
        assert "tzgg" not in unnamed[0].name


class TestFallbackClustering:
    """导航不可识别时改以链接聚类得出候选，名称标记为未知。"""

    NO_NAV = """<html><body><div class="wrapper">
      %s
    </div></body></html>""" % "".join(
        f'<span><a href="/col{i}/list.htm">栏目{i}</a></span>' for i in range(8)
    )

    def test_columns_still_found_without_navigation(self):
        tree = discover_columns(self.NO_NAV, base_url="https://x.edu.cn/main.htm")
        assert tree.columns

    def test_fallback_is_reported(self):
        tree = discover_columns(self.NO_NAV, base_url="https://x.edu.cn/main.htm")
        assert tree.source == "clustering"

    def test_navigation_source_is_reported(self, tree):
        assert tree.source == "navigation"

    def test_fallback_columns_need_manual_naming(self):
        html = """<html><body><div class="wrapper">%s</div></body></html>""" % "".join(
            f'<span><a href="/col{i}/list.htm"></a></span>' for i in range(8)
        )
        tree = discover_columns(html, base_url="https://x.edu.cn/main.htm")
        assert all(c.needs_manual_name for c in tree.columns)


class TestEdgeCases:
    def test_unparseable_html_yields_empty_tree(self):
        assert discover_columns("<<<not html", base_url="https://x.edu.cn/").columns == ()

    def test_page_without_links_yields_empty_tree(self):
        assert discover_columns("<html><body><p>无链接</p></body></html>",
                                base_url="https://x.edu.cn/").columns == ()

    def test_tree_is_immutable(self, tree):
        with pytest.raises(Exception):
            tree.columns = ()

    def test_column_is_immutable(self, tree):
        with pytest.raises(Exception):
            tree.columns[0].name = "改了"

    def test_columns_are_ordered_stably(self):
        first = discover_columns(_home(), base_url=HOME_URL)
        second = discover_columns(_home(), base_url=HOME_URL)
        assert [c.url for c in first.columns] == [c.url for c in second.columns]

    def test_is_a_column(self, tree):
        assert all(isinstance(c, Column) for c in tree.columns)


class TestSitemapPath:
    """站点提供 sitemap 时优先据其建立栏目候选，以减少请求量。

    目标站点（ccs / www / jwc.fjnu.edu.cn）全部没有 sitemap，故这是备用
    路径而非主路径；实现它是为了在有 sitemap 的站点上省掉逐栏目探查。
    """

    URLSET = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://x.edu.cn/tzgg/list.htm</loc></url>
      <url><loc>https://x.edu.cn/tzgg/list2.htm</loc></url>
      <url><loc>https://x.edu.cn/kydt/list.htm</loc></url>
      <url><loc>https://x.edu.cn/a1/b2/c3a4/page.htm</loc></url>
      <url><loc>https://x.edu.cn/a5/b6/c7a8/page.htm</loc></url>
      <url><loc>https://other.cn/x.htm</loc></url>
    </urlset>"""

    INDEX = """<?xml version="1.0" encoding="UTF-8"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://x.edu.cn/sitemap-1.xml</loc></sitemap>
      <sitemap><loc>https://x.edu.cn/sitemap-2.xml</loc></sitemap>
    </sitemapindex>"""

    def test_locations_parsed(self):
        from kbwb.structure.columns import parse_sitemap

        result = parse_sitemap(self.URLSET)
        assert result.is_index is False
        assert "https://x.edu.cn/kydt/list.htm" in result.locations

    def test_index_is_recognised(self):
        from kbwb.structure.columns import parse_sitemap

        result = parse_sitemap(self.INDEX)
        assert result.is_index is True
        assert len(result.locations) == 2

    def test_malformed_xml_yields_nothing(self):
        from kbwb.structure.columns import parse_sitemap

        assert parse_sitemap("<not xml").locations == ()

    def test_columns_from_sitemap(self):
        from kbwb.structure.columns import columns_from_sitemap

        tree = columns_from_sitemap(self.URLSET, base_url="https://x.edu.cn/main.htm")
        assert tree.source == "sitemap"
        assert tree.columns

    def test_offsite_locations_excluded(self):
        from kbwb.structure.columns import columns_from_sitemap

        tree = columns_from_sitemap(self.URLSET, base_url="https://x.edu.cn/main.htm")
        assert not any("other.cn" in c.url for c in tree.columns)

    def test_urls_are_clustered_by_shape(self):
        from kbwb.structure.columns import columns_from_sitemap

        tree = columns_from_sitemap(self.URLSET, base_url="https://x.edu.cn/main.htm")
        # 两个 /#/#/#/page.htm 详情页归为一个候选，两个 list 栏目各自成候选
        assert len(tree.columns) == len({c.url for c in tree.columns})
        assert len(tree.columns) < 5

    def test_sitemap_columns_need_manual_naming(self):
        """sitemap 不含栏目名称，不得据 URL 片段伪造。"""
        from kbwb.structure.columns import columns_from_sitemap

        tree = columns_from_sitemap(self.URLSET, base_url="https://x.edu.cn/main.htm")
        assert all(c.needs_manual_name and c.name == NAME_UNKNOWN for c in tree.columns)

    def test_empty_sitemap_yields_empty_tree(self):
        from kbwb.structure.columns import columns_from_sitemap

        empty = '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>'
        assert columns_from_sitemap(empty, base_url="https://x.edu.cn/").columns == ()

    def test_navigation_path_is_unaffected(self, tree):
        # 无 sitemap 时仍走导航路径
        assert tree.source == "navigation"
