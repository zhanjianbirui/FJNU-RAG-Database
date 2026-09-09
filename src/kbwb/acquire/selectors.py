"""候选选择器推断。

对采样到的详情页逐字段枚举候选 CSS 选择器，按其在样本上的命中率排序。

三条规格约束决定了这里的形状：

- 候选必须携带命中率与命中样本数，用户据此判断能否采信；
- 无可用候选时返回空列表并标记"需人工指定"，而不是拿一个命中率为零的
  选择器充数——后者会让草稿看起来可用，实则一抓即空；
- 命中的判定必须**按爬虫真实的抽取方式**进行（见 :func:`_extracts`）。

正文字段额外依赖文本密度：导航、侧栏与页脚同样带 class，仅靠标签名无法与
正文区分，而它们的"文字量相对于标记体积"显著更低。
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import lxml.html

from kbwb.acquire.dom import has_noise_ancestor, text_of
from kbwb.acquire.sampling import SampledPage

__all__ = [
    "FIELDS",
    "FieldInference",
    "InferenceResult",
    "SelectorCandidate",
    "infer_selectors",
]

FIELDS = ("body", "title", "date", "department", "attachment")

DEFAULT_ATTACHMENT_EXTENSIONS = ("pdf", "doc", "docx", "xls", "xlsx")
DEFAULT_MIN_BODY_CHARS = 50

#: 常见中文日期写法：2026-03-01 / 2026年3月1日 / 2026/03/01
_DATE_RE = re.compile(r"(19|20)\d{2}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}")

#: 发布部门的常见收尾字样
_DEPARTMENT_RE = re.compile(r"(处|院|系|部|办公室|中心|科|馆|所)$")

_BODY_TAGS = frozenset({"div", "article", "section", "main", "td"})
_TITLE_TAGS = frozenset({"h1", "h2", "h3"})
_MAX_TITLE_CHARS = 120
_MAX_BODY_ELEMENTS = 3


@dataclass(frozen=True, slots=True)
class SelectorCandidate:
    selector: str
    hits: int
    samples: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.samples if self.samples else 0.0


@dataclass(frozen=True, slots=True)
class FieldInference:
    field: str
    candidates: tuple[SelectorCandidate, ...]

    @property
    def needs_manual(self) -> bool:
        """无任何候选即需人工指定。"""
        return not self.candidates

    @property
    def best(self) -> SelectorCandidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    samples: int
    fields: dict[str, FieldInference]


def _selectors_for(element) -> list[str]:
    """为一个元素枚举可用于定位它的选择器，由具体到宽泛。"""
    tag = element.tag if isinstance(element.tag, str) else ""
    if not tag:
        return []
    found: list[str] = []
    element_id = (element.get("id") or "").strip()
    if element_id and " " not in element_id:
        found.append(f"{tag}#{element_id}")
    for name in (element.get("class") or "").split():
        found.append(f"{tag}.{name}")
    if tag in _TITLE_TAGS:
        found.append(tag)
    return found


def _density_score(element) -> float:
    """文本密度评分。

    只按文本长度排序会让最外层容器胜出——它包住整页，文本自然最多。密度
    （文字量相对于标记体积）能把"字多但标记也多"的导航壳与"字多且标记少"
    的正文段落区分开。
    """
    text_length = len(text_of(element))
    if text_length == 0:
        return 0.0
    markup_length = max(len(lxml.html.tostring(element, encoding="unicode")), 1)
    return text_length**2 / markup_length


def _body_elements(document, min_body_chars: int) -> list:
    """按文本密度取前若干个块级元素，排除结构性噪声。"""
    scored = [
        (_density_score(node), node)
        for node in document.iter()
        if isinstance(node.tag, str)
        and node.tag in _BODY_TAGS
        and not has_noise_ancestor(node)
        and len(text_of(node)) >= min_body_chars
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [node for _, node in scored[:_MAX_BODY_ELEMENTS]]


def _suffixes(extensions: Sequence[str]) -> tuple[str, ...]:
    return tuple(f".{ext.lower().lstrip('.')}" for ext in extensions)


def _satisfies(field: str, node, *, extensions: Sequence[str], body_set: set) -> bool:
    """该元素能否作为此字段被成功抽取。"""
    if not isinstance(node.tag, str) or has_noise_ancestor(node):
        return False
    if field == "body":
        return node in body_set
    if field == "attachment":
        if node.tag != "a":
            return False
        href = (node.get("href") or "").lower().split("?")[0]
        return href.endswith(_suffixes(extensions))
    text = text_of(node)
    if not text:
        return False
    if field == "title":
        is_title_ish = node.tag in _TITLE_TAGS or "title" in (node.get("class") or "").lower()
        return is_title_ish and len(text) <= _MAX_TITLE_CHARS
    if field == "date":
        return bool(_DATE_RE.search(text)) and len(text) <= 60
    if field == "department":
        return bool(_DEPARTMENT_RE.search(text)) and 1 < len(text) <= 40
    return False


def _attachment_selectors(document, extensions: Sequence[str]) -> set[str]:
    """只为**实际命中的**扩展名生成候选。

    早期实现对每个附件链接把配置中的全部扩展名都生成一遍候选，它们命中数
    相同，排序由字典序决定胜负——草稿因而可能给出站点上根本不存在的扩展名，
    看着合理却一抓即空。
    """
    found: set[str] = set()
    for link in document.xpath("//a[@href]"):
        href = (link.get("href") or "").lower().split("?")[0]
        matched = next((s for s in _suffixes(extensions) if href.endswith(s)), None)
        if matched is None or has_noise_ancestor(link):
            continue
        parent = link.getparent()
        scopes = _selectors_for(parent) if parent is not None else []
        for scope in scopes:
            found.add(f'{scope} a[href$="{matched}"]')
        if not scopes:
            found.add(f'a[href$="{matched}"]')
    return found


def _candidate_selectors(document, field: str, *, extensions, body_set: set) -> set[str]:
    """自底向上枚举可能定位到该字段的选择器。"""
    if field == "attachment":
        return _attachment_selectors(document, extensions)
    nodes = body_set if field == "body" else document.iter()
    found: set[str] = set()
    for node in nodes:
        if _satisfies(field, node, extensions=extensions, body_set=body_set):
            found.update(_selectors_for(node))
    return found


def _extracts(document, selector: str, field: str, *, extensions, body_set: set) -> bool:
    """自顶向下验证：按爬虫真实的抽取方式（取第一个匹配元素）能否拿到该字段。

    仅凭"页面上存在某个满足条件的元素"就记一次命中是不够的——像
    ``div.container`` 这类匹配多个元素的选择器，抽取时取到的往往是排在前面
    的导航壳，与推断时看中的元素并非同一个。命中率必须沿真实抽取路径计算，
    否则草稿会给出看着 100%、实际抽出导航栏的选择器。
    """
    try:
        matched = document.cssselect(selector)
    except Exception:
        return False
    return bool(matched) and _satisfies(
        field, matched[0], extensions=extensions, body_set=body_set
    )


def _rank(counts: dict[str, int], samples: int) -> tuple[SelectorCandidate, ...]:
    """命中率降序；同率时偏好更具体（更长）的选择器，最后按字典序稳定排序。"""
    candidates = [
        SelectorCandidate(selector=selector, hits=hits, samples=samples)
        for selector, hits in counts.items()
        if hits > 0
    ]
    candidates.sort(key=lambda c: (-c.hits, -len(c.selector), c.selector))
    return tuple(candidates)


def infer_selectors(
    pages: Sequence[SampledPage],
    *,
    attachment_extensions: Sequence[str] = DEFAULT_ATTACHMENT_EXTENSIONS,
    min_body_chars: int = DEFAULT_MIN_BODY_CHARS,
) -> InferenceResult:
    documents = []
    for page in pages:
        try:
            documents.append(lxml.html.fromstring(page.html))
        except Exception:
            continue  # 单个样本解析失败不影响其余样本的统计

    counts: dict[str, dict[str, int]] = {field: defaultdict(int) for field in FIELDS}
    for document in documents:
        body_set = set(_body_elements(document, min_body_chars))
        for field in FIELDS:
            for selector in _candidate_selectors(
                document, field, extensions=attachment_extensions, body_set=body_set
            ):
                if _extracts(
                    document,
                    selector,
                    field,
                    extensions=attachment_extensions,
                    body_set=body_set,
                ):
                    counts[field][selector] += 1

    samples = len(documents)
    return InferenceResult(
        samples=samples,
        fields={
            field: FieldInference(field=field, candidates=_rank(counts[field], samples))
            for field in FIELDS
        },
    )
