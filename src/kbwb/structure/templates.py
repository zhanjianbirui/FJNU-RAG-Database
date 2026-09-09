"""页面模板聚类。

把结构相同的页面归为一类，使采样以模板而非栏目为单位——实测一个学院站点
的四十个栏目只对应四五种模板，按模板采样比按栏目采样少一个数量级的请求。

聚类判据是**结构指纹与 URL 形状的合取**：

- 只看 URL 形状，会把改版后的页面误并（URL 没变而结构变了），也会把文章
  详情页与教师个人页混为一谈——它们在真实站点上 URL 形状完全相同；
- 只看结构指纹，会把不同栏目的同模板页面拆开，因为内容差异会带来细微的
  指纹差别。

两者一致才归为一类。
"""

import hashlib
from dataclasses import dataclass
from urllib.parse import urlsplit

import lxml.html

from kbwb.acquire.dom import has_noise_ancestor

__all__ = [
    "PageSample",
    "PageTemplate",
    "cluster_templates",
    "should_sample",
    "similarity",
    "structure_fingerprint",
    "url_shape",
]

DEFAULT_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class PageSample:
    url: str
    html: str
    column: str


@dataclass(frozen=True, slots=True)
class PageTemplate:
    template_id: str
    url_shape: str
    sample_urls: tuple[str, ...]
    columns: tuple[str, ...]


def url_shape(url: str) -> str:
    """URL 的路径形状：含数字的段归一为 ``#``，查询串忽略。"""
    path = urlsplit(url).path
    return "/".join("#" if any(c.isdigit() for c in seg) else seg for seg in path.split("/"))


#: 指纹至少要有这么多个标记才算得上"有结构"，否则视为不可用
_MIN_TOKENS = 3


def structure_fingerprint(html: str) -> frozenset[str]:
    """页面内容区的结构指纹：「标签 + class 首词」的集合。

    两处收窄都很关键：

    **排除全站页头页脚导航。** 同一站点的所有页面共享这套外壳，它在标记
    集合里占大头，会把内容区的差异淹没——实测文章详情页与教师个人页因此
    被判为同一模板，而它们的正文区结构截然不同。

    **只取 class 首词。** 站点常在元素上叠加动效类（``wow fadeInUp``），
    它们随内容出现与否而变，会给同模板页面制造无谓的指纹差异。
    """
    try:
        document = lxml.html.fromstring(html)
    except Exception:
        return frozenset()
    tokens = set()
    for node in document.iter():
        if not isinstance(node.tag, str) or has_noise_ancestor(node):
            continue
        classes = (node.get("class") or "").split()
        tokens.add(f"{node.tag}.{classes[0]}" if classes else node.tag)
    return frozenset(tokens) if len(tokens) >= _MIN_TOKENS else frozenset()


def similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard 相似度。两个空指纹判为不相似——解析失败不构成"结构相同"。"""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _template_id(url_pattern: str, fingerprint: frozenset[str]) -> str:
    """由 URL 形状与指纹派生的稳定标识，便于跨次探测比对同一模板。"""
    digest = hashlib.sha256()
    digest.update(url_pattern.encode("utf-8"))
    for token in sorted(fingerprint):
        digest.update(b"\x00")
        digest.update(token.encode("utf-8"))
    return digest.hexdigest()[:12]


class _Cluster:
    def __init__(self, sample: PageSample, fingerprint: frozenset[str]) -> None:
        self.shape = url_shape(sample.url)
        self.fingerprint = fingerprint
        self.urls = [sample.url]
        self.columns = [sample.column]

    def accepts(self, sample: PageSample, fingerprint: frozenset[str], threshold: float) -> bool:
        return (
            url_shape(sample.url) == self.shape
            and similarity(self.fingerprint, fingerprint) >= threshold
        )

    def add(self, sample: PageSample, fingerprint: frozenset[str]) -> None:
        self.urls.append(sample.url)
        if sample.column not in self.columns:
            self.columns.append(sample.column)
        # 指纹取交集，逐步收敛到该模板的共有结构
        self.fingerprint = self.fingerprint & fingerprint or self.fingerprint

    def freeze(self) -> PageTemplate:
        return PageTemplate(
            template_id=_template_id(self.shape, self.fingerprint),
            url_shape=self.shape,
            sample_urls=tuple(self.urls),
            columns=tuple(self.columns),
        )


def cluster_templates(
    samples: list[PageSample], *, threshold: float = DEFAULT_THRESHOLD
) -> tuple[PageTemplate, ...]:
    """把样本页面聚成模板，按覆盖的样本数降序返回。"""
    clusters: list[_Cluster] = []
    for sample in samples:
        fingerprint = structure_fingerprint(sample.html)
        target = next(
            (c for c in clusters if c.accepts(sample, fingerprint, threshold)), None
        )
        if target is None:
            clusters.append(_Cluster(sample, fingerprint))
        else:
            target.add(sample, fingerprint)
    ordered = sorted(clusters, key=lambda c: (-len(c.urls), c.shape))
    return tuple(c.freeze() for c in ordered)


def should_sample(
    templates: tuple[PageTemplate, ...], url: str, *, samples_per_template: int
) -> bool:
    """判断某个候选 URL 是否还需要抓取。

    采样单位是模板而非栏目：某个 URL 形状已经攒够样本时，即便它属于另一个
    尚未探过的栏目，也不必再抓——多个栏目共用同一模板是常态，逐栏目采样会
    把请求量翻上几倍却得不到新信息。
    """
    shape = url_shape(url)
    collected = sum(
        len(template.sample_urls) for template in templates if template.url_shape == shape
    )
    return collected < samples_per_template
