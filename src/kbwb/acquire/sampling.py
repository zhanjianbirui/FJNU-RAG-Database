"""站点采样。

在遵守 robots.txt 与限速的前提下，从入口页取回若干详情页样本，供选择器
推断使用。三条行为约束直接对应 site-analysis 规格：

- 被 robots 禁止的 URL 不发起请求，并连同命中的规则一起报告；
- 采样规模有上限，避免对目标站点造成压力；
- 单个入口失败只记录该入口，不影响其余入口。
"""

import re
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlsplit

import lxml.html

from kbwb.acquire.fetching import (
    DomainThrottle,
    FetchError,
    FetchResponse,
    Fetcher,
    origin_of,
)
from kbwb.acquire.dom import has_noise_ancestor
from kbwb.acquire.robots import RobotsGate
from kbwb.config.site import SiteConfig

__all__ = [
    "FailedUrl",
    "SampleResult",
    "SampledPage",
    "Sampler",
    "SkippedUrl",
    "build_user_agent",
]

PROJECT_NAME = "kbwb"
PROJECT_VERSION = "0.1"


def build_user_agent(contact: str) -> str:
    """构造可识别的 User-Agent。

    规格要求其中含项目标识与一个可供站点管理员联系的 URL 或邮箱——这是
    "不使用反检测手段"的正面表述：主动表明身份而非隐藏身份。
    """
    if not contact or not contact.strip():
        raise ValueError("User-Agent 必须包含可联系的 URL 或邮箱")
    return f"{PROJECT_NAME}/{PROJECT_VERSION} (+{contact.strip()})"


@dataclass(frozen=True, slots=True)
class SampledPage:
    url: str
    html: str


@dataclass(frozen=True, slots=True)
class SkippedUrl:
    url: str
    reason: str
    rule: str | None = None


@dataclass(frozen=True, slots=True)
class FailedUrl:
    url: str
    reason: str


@dataclass(frozen=True, slots=True)
class SampleResult:
    pages: tuple[SampledPage, ...]
    skipped: tuple[SkippedUrl, ...]
    failures: tuple[FailedUrl, ...]

    @property
    def sampled_count(self) -> int:
        return len(self.pages)


class _Collector:
    """一次采样过程中的可变累积器，最终固化为不可变的 SampleResult。"""

    def __init__(self) -> None:
        self.pages: list[SampledPage] = []
        self.skipped: list[SkippedUrl] = []
        self.failures: list[FailedUrl] = []

    def freeze(self) -> SampleResult:
        return SampleResult(
            pages=tuple(self.pages),
            skipped=tuple(self.skipped),
            failures=tuple(self.failures),
        )


class Sampler:
    def __init__(
        self,
        fetcher: Fetcher,
        *,
        contact: str,
        throttle: DomainThrottle,
        gate: RobotsGate | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._user_agent = build_user_agent(contact)
        self._throttle = throttle
        self._gate = gate or RobotsGate(fetcher, user_agent=self._user_agent)

    def sample(self, config: SiteConfig) -> SampleResult:
        collector = _Collector()
        allow = [re.compile(p) for p in config.allow_patterns]
        deny = [re.compile(p) for p in config.deny_patterns]
        seen: set[str] = set()

        for entry in config.entry_urls:
            page = self._fetch_allowed(entry, collector)
            if page is None:
                continue
            seen.add(entry)
            for link in self._detail_links(page, allow, deny):
                if len(collector.pages) >= config.sampling.max_samples:
                    return collector.freeze()
                if link in seen:
                    continue
                seen.add(link)
                detail = self._fetch_allowed(link, collector)
                if detail is not None:
                    collector.pages.append(SampledPage(url=detail.url, html=detail.text))
        return collector.freeze()

    def _fetch_allowed(self, url: str, collector: _Collector) -> FetchResponse | None:
        """robots 判定通过后才抓取。被禁与失败分别记入不同的账。"""
        decision = self._gate.check(url)
        if not decision.allowed:
            collector.skipped.append(
                SkippedUrl(url=url, reason=decision.reason or "robots", rule=decision.rule)
            )
            return None
        self._throttle.acquire(url)
        try:
            response = self._fetcher.get(url)
        except FetchError as exc:
            collector.failures.append(FailedUrl(url=url, reason=str(exc)))
            return None
        if response.status >= 400:
            collector.failures.append(
                FailedUrl(url=url, reason=f"HTTP {response.status}")
            )
            return None
        return response

    def _detail_links(
        self, page: FetchResponse, allow: list[re.Pattern], deny: list[re.Pattern]
    ) -> list[str]:
        """从入口页识别指向**详情页**的链接。

        只按文档顺序取所有同源链接是不够的：栏目页顶部的全站导航排在文章
        列表之前，会先把采样配额吃光，一个真正的详情页都采不到。

        因此分两步收敛：先排除位于导航、侧栏、页脚等结构性区域内的链接；
        再按 URL 形状分组，同形状链接最多的一组排在前面——列表页上的文章
        链接天然共享同一套路径模板，而导航项彼此形状各异。
        """
        try:
            document = lxml.html.fromstring(page.text)
        except Exception:
            return []
        origin = origin_of(page.url)
        candidates: list[tuple[str, bool]] = []  # (url, 是否位于噪声区域)
        seen: set[str] = set()
        for anchor in document.xpath("//a[@href]"):
            candidate = self._normalise(page.url, anchor.get("href"))
            if candidate is None or candidate in seen or candidate == page.url:
                continue
            if not candidate.startswith(origin + "/") and candidate != origin:
                continue
            if allow and not any(pattern.search(candidate) for pattern in allow):
                continue
            if any(pattern.search(candidate) for pattern in deny):
                continue
            seen.add(candidate)
            candidates.append((candidate, has_noise_ancestor(anchor)))

        content_links = [url for url, noisy in candidates if not noisy]
        # 整页都被标记为噪声容器时退回全部候选，避免一个样本都取不到
        return _rank_by_shape(content_links or [url for url, _ in candidates])

    @staticmethod
    def _normalise(base: str, href: str) -> str | None:
        text = (href or "").strip()
        if not text or text.startswith(("#", "javascript:", "mailto:", "tel:")):
            return None
        resolved, _ = urldefrag(urljoin(base, text))
        return resolved if resolved.startswith(("http://", "https://")) else None


def _shape_of(url: str) -> str:
    """URL 的路径形状：含数字的段归一为 ``#``。

    同一栏目下的文章链接共享路径模板，形状因此相同；导航项各自指向不同
    栏目，形状彼此不同。
    """
    path = urlsplit(url).path
    segments = ["#" if any(ch.isdigit() for ch in seg) else seg for seg in path.split("/")]
    return "/".join(segments)


def _rank_by_shape(urls: list[str]) -> list[str]:
    """同形状链接多的组排在前面；组内保持文档顺序。"""
    if not urls:
        return []
    groups: dict[str, list[str]] = {}
    for url in urls:
        groups.setdefault(_shape_of(url), []).append(url)
    order = {shape: index for index, shape in enumerate(groups)}
    return [url for group in sorted(groups.values(),
                                    key=lambda g: (-len(g), order[_shape_of(g[0])]))
            for url in group]
