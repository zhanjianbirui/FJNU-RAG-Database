"""页面类型判定与规模估算。

判定依据来自分区识别的结果：有条目列表即列表页，有正文而无列表即单页，
两者皆无则真的不知道它是什么——此时标记为未知并说明原因，而不是猜一个。

规模估算按可靠性分四级依次尝试，并**记录实际采用的方式**。展示时一律取
约数：估算值若以精确数字呈现，使用者会据此做出它支撑不了的判断。
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import lxml.html

from kbwb.structure.regions import PageRegions, RegionType, segment_page

__all__ = [
    "EstimateMethod",
    "PageClassification",
    "PageType",
    "ProbeResult",
    "SizeEstimate",
    "approximate",
    "classify_page",
    "estimate_size",
    "probe_last_page",
]


class PageType(StrEnum):
    LIST = "list"
    SINGLE = "single"
    UNKNOWN = "unknown"


class EstimateMethod(StrEnum):
    ITEM_COUNT_TEXT = "条数文本"
    PAGE_COUNT_TEXT = "页数文本"
    PAGINATION_LINKS = "分页链接"
    PROBING = "试探"
    UNKNOWN = "未知"


#: 条目总数的常见中文写法。量词覆盖 条/篇/项/记录/个。
_ITEM_COUNT_RE = re.compile(r"(?:总共|共有|共|总数)\s*[：:]?\s*(\d+)\s*(?:条|篇|项|记录|个)?")

#: 当前页/总页数，如「页码 1/11」
_PAGE_RATIO_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*页?")

#: 总页数的直接写法
_PAGE_COUNT_RE = re.compile(r"共\s*(\d+)\s*页")

#: 分页链接中的页码，如 list11.htm
_PAGE_LINK_RE = re.compile(r"(\d+)\D*$")

#: 条数文本必须出现在分页区内才作数——正文里的「共 3 位专家」不是条目数
_PAGING_WORDS = ("paging", "pagination", "page_nav", "pagenav", "pager")


@dataclass(frozen=True, slots=True)
class PageClassification:
    type: PageType
    confidence: float
    has_pagination: bool
    reason: str | None = None


def classify_page(regions: PageRegions) -> PageClassification:
    """按分区构成判定页面类型。"""
    entries = regions.of(RegionType.ARTICLE_LIST)
    body = regions.of(RegionType.ARTICLE_BODY)
    paging = regions.of(RegionType.PAGINATION)
    has_pagination = paging is not None

    if entries is not None:
        # 分页控件的存在使「这是一个分栏目列表」更可信
        confidence = round(0.6 + (0.3 if has_pagination else 0.0) + 0.1 * entries.confidence, 2)
        return PageClassification(
            type=PageType.LIST, confidence=min(confidence, 1.0), has_pagination=has_pagination
        )
    if body is not None and body.confidence > 0:
        return PageClassification(
            type=PageType.SINGLE,
            confidence=round(0.5 + 0.5 * body.confidence, 2),
            has_pagination=has_pagination,
        )
    return PageClassification(
        type=PageType.UNKNOWN,
        confidence=0.0,
        has_pagination=has_pagination,
        reason="页面既无条目列表，也未能识别出正文区",
    )


@dataclass(frozen=True, slots=True)
class SizeEstimate:
    items: int | None
    pages: int | None
    method: EstimateMethod

    @property
    def display(self) -> str:
        """以约数呈现，并标明估算方式。"""
        if self.method is EstimateMethod.UNKNOWN or self.items is None:
            return "规模未知"
        pages = f"、{approximate(self.pages)}页" if self.pages else ""
        return f"{approximate(self.items)}条{pages}（据{self.method.value}）"


def approximate(value: int) -> str:
    """把精确数字降级为约数。

    估算值以精确数字呈现会诱导使用者据此做出它支撑不了的判断——"104 条"
    看着像点过数，其实是页数乘以每页条数推出来的。
    """
    if value < 10:
        return f"约 {value}"
    if value < 100:
        return f"约 {value // 10 * 10}"
    if value < 1000:
        return f"约 {value // 50 * 50}"
    return f"约 {value // 100 * 100}"


def _paging_text(document) -> str:
    """只取分页区内的文本。正文里的「共 3 位专家出席」不是条目数。

    按各元素自身的文本片段拼接，而不是取整棵子树的 ``text_content()``：
    后者会把相邻的文本直接连起来，「总数：37」紧挨着页码链接「2」就成了
    「总数：372」。
    """
    parts = []
    for node in document.iter():
        if not isinstance(node.tag, str):
            continue
        marker = f"{node.get('class', '')} {node.get('id', '')}".lower()
        if any(word in marker for word in _PAGING_WORDS):
            parts.extend(piece.strip() for piece in node.itertext() if piece.strip())
    return " ".join(parts)


def _max_page_link(document) -> int | None:
    pages = []
    for anchor in document.xpath("//a[@href]"):
        href = anchor.get("href") or ""
        if href.startswith("javascript:"):
            continue
        match = _PAGE_LINK_RE.search(href.rsplit("/", 1)[-1].split(".")[0])
        if match:
            pages.append(int(match.group(1)))
    return max(pages) if pages else None


def estimate_size(html: str, *, entries_on_page: int) -> SizeEstimate:
    """按可靠性依次尝试四条路径，无信号时如实返回未知。"""
    try:
        document = lxml.html.fromstring(html)
    except Exception:
        return SizeEstimate(items=None, pages=None, method=EstimateMethod.UNKNOWN)

    text = _paging_text(document)
    ratio = _PAGE_RATIO_RE.search(text)
    pages = int(ratio.group(2)) if ratio else None
    if pages is None:
        page_count = _PAGE_COUNT_RE.search(text)
        pages = int(page_count.group(1)) if page_count else None

    count = _ITEM_COUNT_RE.search(text)
    if count:
        return SizeEstimate(
            items=int(count.group(1)), pages=pages, method=EstimateMethod.ITEM_COUNT_TEXT
        )
    if pages and entries_on_page:
        return SizeEstimate(
            items=pages * entries_on_page, pages=pages, method=EstimateMethod.PAGE_COUNT_TEXT
        )

    if text:
        link_pages = _max_page_link(document)
        if link_pages and link_pages > 1 and entries_on_page:
            return SizeEstimate(
                items=link_pages * entries_on_page,
                pages=link_pages,
                method=EstimateMethod.PAGINATION_LINKS,
            )
    return SizeEstimate(items=None, pages=None, method=EstimateMethod.UNKNOWN)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    page: int
    requests: int
    exhausted: bool


def probe_last_page(exists: Callable[[int], bool], *, budget: int) -> ProbeResult:
    """先倍增再二分定位末页，请求数为对数量级。

    仅有「下一页」而无页码时使用。预算耗尽则返回已确认的下界并标记，
    不外推一个没验证过的页码。
    """
    requests = 0

    def check(page: int) -> bool | None:
        nonlocal requests
        if requests >= budget:
            return None
        requests += 1
        return exists(page)

    low, high = 0, 1
    while True:
        answer = check(high)
        if answer is None:
            return ProbeResult(page=low, requests=requests, exhausted=True)
        if not answer:
            break
        low, high = high, high * 2

    while low + 1 < high:
        middle = (low + high) // 2
        answer = check(middle)
        if answer is None:
            return ProbeResult(page=low, requests=requests, exhausted=True)
        if answer:
            low = middle
        else:
            high = middle
    return ProbeResult(page=low, requests=requests, exhausted=False)
