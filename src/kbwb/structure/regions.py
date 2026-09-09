"""页面分区识别。

把一张页面切分为固定集合中的八类分区。每个分区携带**能唯一定位回该元素**
的选择器与置信度——置信度而非布尔判定，是因为使用者需要看到依据才能判断
是否采信（与推荐器的取向一致）。

正文分区借助 trafilatura；它返回文本而非节点，因此需把文本映射回 DOM 元素
才能给出选择器。其余七类分区自研，见后续任务。
"""

import re
from dataclasses import dataclass
from enum import StrEnum

import lxml.html
import trafilatura

from kbwb.acquire.dom import has_noise_ancestor, text_of, unique_selector

__all__ = ["REGION_TYPES", "PageRegions", "Region", "RegionType", "segment_page"]


class RegionType(StrEnum):
    NAVIGATION = "navigation"
    ARTICLE_BODY = "article_body"
    ARTICLE_META = "article_meta"
    ARTICLE_LIST = "article_list"
    PAGINATION = "pagination"
    ATTACHMENT = "attachment"
    FOOTER = "footer"
    SIDEBAR = "sidebar"


REGION_TYPES: tuple[RegionType, ...] = tuple(RegionType)

#: 用于把提取文本定位回 DOM 的特征前缀长度
_NEEDLE_CHARS = 30

#: 候选元素至少要包含提取文本的这个比例，才可能是正文容器
_MIN_CONTAINMENT = 0.6

#: 各结构性分区的 class/id 关键词。命中关键词是强信号，但不足以定分区——
#: 还要看链接密度，否则一个恰好叫 "header" 的正文容器会被误判。
_MARKERS: dict[RegionType, tuple[str, ...]] = {
    RegionType.NAVIGATION: ("nav", "menu", "header", "topbar"),
    RegionType.FOOTER: ("footer", "copyright"),
    RegionType.SIDEBAR: ("sidebar", "side-bar", "aside"),
}

#: 分页容器的 class/id 关键词
_PAGING_WORDS = ("paging", "pagination", "page_nav", "pagenav", "pager")

#: 条目列表至少要有这么多条目才成立，低于此数更可能是零星链接
_MIN_LIST_ENTRIES = 5

#: 元信息区的文本上限。超过这个长度的多半已经把正文裹进来了
_MAX_META_CHARS = 60

#: 视作附件的扩展名
_ATTACHMENT_SUFFIXES = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx")

_DATE_RE = re.compile(r"(19|20)\d{2}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}")


@dataclass(frozen=True, slots=True)
class Region:
    type: RegionType
    selector: str
    confidence: float
    note: str | None = None


@dataclass(frozen=True, slots=True)
class PageRegions:
    regions: tuple[Region, ...]

    def of(self, region_type: RegionType) -> Region | None:
        return next((r for r in self.regions if r.type is region_type), None)


def _parse(html: str):
    if not html or not html.strip():
        return None
    try:
        return lxml.html.fromstring(html)
    except Exception:
        return None


def _extract_body_text(html: str) -> str:
    text = trafilatura.extract(html, favor_precision=True, include_comments=False)
    return " ".join(text.split()) if text else ""


def _locate(document, text: str):
    """把提取到的正文文本映射回最小的包含它的 DOM 元素。"""
    needle = text[:_NEEDLE_CHARS]
    if not needle:
        return None
    best = None
    best_length = None
    for node in document.iter():
        if not isinstance(node.tag, str):
            continue
        node_text = text_of(node)
        if len(node_text) < len(text) * _MIN_CONTAINMENT or needle not in node_text:
            continue
        if best_length is None or len(node_text) < best_length:
            best, best_length = node, len(node_text)
    return best


def _body_confidence(extract_length: int, element_length: int) -> float:
    """按"正文占该元素文本的比重"给出置信度。

    元素里除正文之外还夹带得越多（标题、浏览次数、相关推荐），越说明这个
    容器选大了，置信度相应降低。
    """
    purity = min(extract_length / max(element_length, 1), 1.0)
    return round(0.5 + 0.5 * purity, 2)


def _detect_body(document, html: str) -> Region:
    """识别正文分区。识别不到时仍产出分区，但置信度为零并标记原因。"""
    text = _extract_body_text(html)
    element = _locate(document, text) if text else None
    if element is None:
        return Region(
            type=RegionType.ARTICLE_BODY,
            selector="body",
            confidence=0.0,
            note="未能识别正文区",
        )
    return Region(
        type=RegionType.ARTICLE_BODY,
        selector=unique_selector(document, element),
        confidence=_body_confidence(len(text), len(text_of(element))),
    )


def _link_ratio(element) -> float:
    """链接相对于纯文本的占比。导航与页脚链接密集，正文相反。"""
    links = len(element.xpath(".//a[@href]"))
    text_weight = len(text_of(element)) / 20
    return links / max(links + text_weight, 1e-9)


def _matches_marker(element, words: tuple[str, ...]) -> bool:
    marker = f"{element.get('class', '')} {element.get('id', '')}".lower()
    return any(word in marker for word in words)


def _detect_marked(document, region_type: RegionType) -> Region | None:
    """按 class/id 关键词定位结构性分区，取其中链接最多的那个容器。"""
    words = _MARKERS[region_type]
    candidates = [
        node
        for node in document.iter()
        if isinstance(node.tag, str)
        and _matches_marker(node, words)
        and node.xpath(".//a[@href]")
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda n: len(n.xpath(".//a[@href]")))
    return Region(
        type=region_type,
        selector=unique_selector(document, best),
        confidence=round(0.5 + 0.5 * _link_ratio(best), 2),
    )


def _detect_pagination(document) -> Region | None:
    """分页区。关键词是强信号，数字页码链接的比例决定置信度。"""
    candidates = [
        node
        for node in document.iter()
        if isinstance(node.tag, str)
        and _matches_marker(node, _PAGING_WORDS)
        and node.xpath(".//a[@href]")
    ]
    if not candidates:
        return None
    best = max(candidates, key=lambda n: len(n.xpath(".//a[@href]")))
    links = best.xpath(".//a[@href]")
    numeric = sum(1 for a in links if text_of(a).isdigit())
    return Region(
        type=RegionType.PAGINATION,
        selector=unique_selector(document, best),
        confidence=round(0.7 + 0.3 * (numeric / max(len(links), 1)), 2),
    )


def _entry_children(node) -> list:
    """同标签、且各自含链接的直接子元素——列表条目的形态特征。"""
    children = [c for c in node if isinstance(c.tag, str)]
    if not children:
        return []
    dominant = max({c.tag for c in children}, key=lambda t: sum(1 for c in children if c.tag == t))
    return [c for c in children if c.tag == dominant and c.xpath(".//a[@href]")]


def _detect_article_list(document) -> Region | None:
    """条目列表区：同标签的重复条目，不位于结构性噪声内，且取最内层容器。

    不排除 ``html``/``body`` 并取"条目最多者"会选中整个页面——页面的直接
    子元素也各自含链接，条目数反而更多。同理，外层容器若包住了另一个合格
    容器，真正的列表在里面，应取内层。
    """
    candidates: list[tuple] = []
    for node in document.iter():
        if not isinstance(node.tag, str) or node.tag in ("html", "body"):
            continue
        if has_noise_ancestor(node):
            continue
        entries = _entry_children(node)
        if len(entries) >= _MIN_LIST_ENTRIES:
            candidates.append((node, len(entries)))
    if not candidates:
        return None
    nodes = {id(node) for node, _ in candidates}
    innermost = [
        (node, count)
        for node, count in candidates
        if not any(id(d) in nodes for d in node.iterdescendants())
    ]
    best, count = max(innermost or candidates, key=lambda pair: pair[1])
    return Region(
        type=RegionType.ARTICLE_LIST,
        selector=unique_selector(document, best),
        confidence=round(0.5 + 0.5 * (count / (count + 5)), 2),
    )


def _detect_meta(document, body_selector: str | None) -> Region | None:
    """元信息区：含发布日期的短文本元素，取其中最小的那个。"""
    best = None
    best_length = None
    for node in document.iter():
        if not isinstance(node.tag, str) or has_noise_ancestor(node):
            continue
        text = text_of(node)
        if not text or len(text) > _MAX_META_CHARS or not _DATE_RE.search(text):
            continue
        if best_length is None or len(text) < best_length:
            best, best_length = node, len(text)
    if best is None:
        return None
    return Region(
        type=RegionType.ARTICLE_META,
        selector=unique_selector(document, best),
        confidence=round(0.6 + 0.4 * (1 - best_length / _MAX_META_CHARS), 2),
    )


def _is_attachment_link(anchor) -> bool:
    href = (anchor.get("href") or "").lower().split("?")[0]
    return href.endswith(_ATTACHMENT_SUFFIXES)


def _detect_attachment(document) -> Region | None:
    """附件区：文档类链接最集中的容器。"""
    best = None
    best_count = 0
    for node in document.iter():
        if not isinstance(node.tag, str) or has_noise_ancestor(node):
            continue
        count = sum(1 for a in node.xpath(".//a[@href]") if _is_attachment_link(a))
        if count == 0:
            continue
        # 取包含全部附件链接的最小容器
        if count > best_count or (count == best_count and best is not None
                                  and len(text_of(node)) < len(text_of(best))):
            best, best_count = node, count
    if best is None:
        return None
    total = len(best.xpath(".//a[@href]"))
    return Region(
        type=RegionType.ATTACHMENT,
        selector=unique_selector(document, best),
        confidence=round(0.6 + 0.4 * (best_count / max(total, 1)), 2),
    )


def segment_page(html: str) -> PageRegions:
    """把页面切分为分区。无法解析或内容为空时返回空结果，不抛异常。"""
    document = _parse(html)
    if document is None:
        return PageRegions(regions=())
    body = _detect_body(document, html)
    found = [
        body,
        _detect_marked(document, RegionType.NAVIGATION),
        _detect_marked(document, RegionType.FOOTER),
        _detect_marked(document, RegionType.SIDEBAR),
        _detect_pagination(document),
        _detect_article_list(document),
        _detect_meta(document, body.selector),
        _detect_attachment(document),
    ]
    return PageRegions(regions=tuple(r for r in found if r is not None))
