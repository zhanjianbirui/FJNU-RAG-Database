"""栏目树探测。

栏目来自站点导航，携带人类语义——"通知公告"而不是 `/tzgg/list.htm`。层级
取自导航的嵌套结构。

导航不可识别时退回链接聚类，此时**名称一律标记为未知并提示人工命名**：
拿 URL 片段充当名称看似省事，实则把机器的无知伪装成了信息。
"""

from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlsplit

import lxml.html

from kbwb.acquire.dom import text_of
from kbwb.structure.templates import url_shape

__all__ = [
    "NAME_UNKNOWN",
    "Column",
    "ColumnTree",
    "Sitemap",
    "columns_from_sitemap",
    "discover_columns",
    "parse_sitemap",
]

NAME_UNKNOWN = "未知"

#: 导航容器的 class/id 关键词
_NAV_WORDS = ("nav", "menu", "header", "topbar")

#: 导航中少于这么多条链接时，视为未能识别出导航，退回链接聚类
_MIN_NAV_COLUMNS = 3

_LIST_TAGS = ("ul", "ol")


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    url: str
    depth: int
    parent_url: str | None = None
    needs_manual_name: bool = False


@dataclass(frozen=True, slots=True)
class ColumnTree:
    columns: tuple[Column, ...]
    #: 栏目的来源："navigation" 或 "clustering"
    source: str

    def roots(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.depth == 0)

    def children_of(self, url: str) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.parent_url == url)


def _normalise(base_url: str, href: str) -> str | None:
    text = (href or "").strip()
    if not text or text.startswith(("#", "javascript:", "mailto:", "tel:")):
        return None
    resolved, _ = urldefrag(urljoin(base_url, text))
    return resolved if resolved.startswith(("http://", "https://")) else None


def _is_homepage_like(url: str, base_url: str) -> bool:
    """入口页自身及其同名变体不是栏目。

    这类站点群 CMS 常把首页镜像到 ``/_t1234/main.htm`` 之类的路径下；它们
    与入口页同名，是首页的另一个入口而非某个栏目。
    """
    path = urlsplit(url).path
    base_path = urlsplit(base_url).path
    if path in ("", "/"):
        return True
    base_leaf = base_path.rsplit("/", 1)[-1]
    return bool(base_leaf) and path.rsplit("/", 1)[-1] == base_leaf


def _candidate_links(document, base_url: str, scope=None):
    """产出 (url, 名称, 锚元素)，同源、去重、排除首页变体。"""
    host = urlsplit(base_url).netloc
    seen: set[str] = set()
    for anchor in (scope if scope is not None else document).xpath(".//a[@href]"):
        url = _normalise(base_url, anchor.get("href"))
        if url is None or urlsplit(url).netloc != host:
            continue
        if url in seen or _is_homepage_like(url, base_url):
            continue
        seen.add(url)
        yield url, text_of(anchor), anchor


def _find_navigation(document):
    """取 class/id 命中导航关键词、且链接最多的那个容器。"""
    candidates = [
        node
        for node in document.iter()
        if isinstance(node.tag, str)
        and any(
            word in f"{node.get('class', '')} {node.get('id', '')}".lower()
            for word in _NAV_WORDS
        )
        and node.xpath(".//a[@href]")
    ]
    return max(candidates, key=lambda n: len(n.xpath(".//a[@href]")), default=None)


def _own_anchor(item):
    """列表项自身的链接——嵌套子列表里的那些不算。"""
    for anchor in item.iter("a"):
        node = anchor.getparent()
        nested = False
        while node is not None and node is not item:
            if isinstance(node.tag, str) and node.tag in _LIST_TAGS:
                nested = True
                break
            node = node.getparent()
        if not nested:
            return anchor
    return None


def _depth_of(anchor, scope) -> tuple[int, object | None]:
    """按导航容器内的祖先列表项层数确定深度，并给出上一层的列表项。"""
    items = []
    node = anchor.getparent()
    while node is not None and node is not scope:
        if isinstance(node.tag, str) and node.tag == "li":
            items.append(node)
        node = node.getparent()
    depth = max(len(items) - 1, 0)
    parent_item = items[1] if len(items) > 1 else None
    return depth, parent_item


def _from_navigation(document, base_url: str, scope) -> list[Column]:
    columns: list[Column] = []
    for url, name, anchor in _candidate_links(document, base_url, scope=scope):
        depth, parent_item = _depth_of(anchor, scope)
        parent_url = None
        if parent_item is not None:
            parent_anchor = _own_anchor(parent_item)
            if parent_anchor is not None and parent_anchor is not anchor:
                parent_url = _normalise(base_url, parent_anchor.get("href"))
        columns.append(
            Column(
                name=name or NAME_UNKNOWN,
                url=url,
                depth=depth,
                parent_url=parent_url,
                needs_manual_name=not name,
            )
        )
    return _resolve_parents(columns)


def _resolve_parents(columns: list[Column]) -> list[Column]:
    """父栏目若不在结果集中（被去重或过滤掉），置空而非留下悬挂引用。"""
    urls = {c.url for c in columns}
    return [
        c
        if c.parent_url in urls and c.parent_url != c.url
        else Column(
            name=c.name,
            url=c.url,
            depth=c.depth,
            parent_url=None,
            needs_manual_name=c.needs_manual_name,
        )
        for c in columns
    ]


def _from_clustering(document, base_url: str) -> list[Column]:
    """退路：按 URL 形状聚类，每类给一个候选。

    规格要求此路径下名称一律标记为未知——没有导航就没有站点自己给出的
    栏目名，链接文本未必是栏目名称，不应据此伪造。
    """
    by_shape: dict[str, str] = {}
    for url, _name, _anchor in _candidate_links(document, base_url):
        by_shape.setdefault(url_shape(url), url)
    return [
        Column(name=NAME_UNKNOWN, url=url, depth=0, needs_manual_name=True)
        for url in by_shape.values()
    ]


def discover_columns(html: str, *, base_url: str) -> ColumnTree:
    """从页面还原栏目树。导航不可识别时退回链接聚类。"""
    try:
        document = lxml.html.fromstring(html)
    except Exception:
        return ColumnTree(columns=(), source="clustering")

    scope = _find_navigation(document)
    if scope is not None:
        columns = _from_navigation(document, base_url, scope)
        if len(columns) >= _MIN_NAV_COLUMNS:
            return ColumnTree(columns=tuple(columns), source="navigation")
    return ColumnTree(columns=tuple(_from_clustering(document, base_url)), source="clustering")


_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


@dataclass(frozen=True, slots=True)
class Sitemap:
    locations: tuple[str, ...]
    #: True 表示这是一份 sitemap 索引，其中的地址指向其他 sitemap
    is_index: bool


def parse_sitemap(xml: str) -> Sitemap:
    """解析 sitemap，兼容 urlset 与 sitemapindex。畸形内容返回空结果。"""
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(xml.strip())
    except ElementTree.ParseError:
        return Sitemap(locations=(), is_index=False)
    is_index = root.tag.endswith("sitemapindex")
    locations = tuple(
        node.text.strip()
        for node in root.iter(f"{_SITEMAP_NS}loc")
        if node.text and node.text.strip()
    )
    return Sitemap(locations=locations, is_index=is_index)


def columns_from_sitemap(xml: str, *, base_url: str) -> ColumnTree:
    """据 sitemap 建立栏目候选，按 URL 形状聚类。

    sitemap 里没有栏目名称，因此候选一律标记为需人工命名——与链接聚类同理，
    不拿 URL 片段伪造名称。
    """
    host = urlsplit(base_url).netloc
    by_shape: dict[str, str] = {}
    for url in parse_sitemap(xml).locations:
        if urlsplit(url).netloc != host or _is_homepage_like(url, base_url):
            continue
        by_shape.setdefault(url_shape(url), url)
    columns = tuple(
        Column(name=NAME_UNKNOWN, url=url, depth=0, needs_manual_name=True)
        for url in by_shape.values()
    )
    return ColumnTree(columns=columns, source="sitemap")
