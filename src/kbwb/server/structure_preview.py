"""结构分析的临时预览服务。

**临时性质**：正式的工作台界面（任务组 12）需要完整的探测编排——栏目树、
页面类型、规模估算——这些在任务组 4 到 6 实现。本模块只把已完成的分区识别
与模板聚类接到界面上，供人工核对识别质量，不属于任务 12.x 的交付。
"""

from dataclasses import dataclass
from typing import Any

from kbwb.acquire.dom import text_of
from kbwb.acquire.fetching import DomainThrottle
from kbwb.acquire.http_client import HttpxFetcher
from kbwb.acquire.sampling import SampledPage, Sampler, build_user_agent
from kbwb.config.site import parse_site_config
from kbwb.structure.regions import RegionType, segment_page
from kbwb.structure.templates import PageSample, cluster_templates

__all__ = ["StructurePreviewService"]

DEFAULT_MAX_SAMPLES = 4
DEFAULT_MIN_INTERVAL = 2.0

#: 界面上每个分区展示的文本长度
_SNIPPET_CHARS = 90


@dataclass(frozen=True, slots=True)
class StructurePreviewService:
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL
    fetcher_factory: Any = None

    def preview(self, *, entry_url: str, contact: str, max_samples: int) -> dict:
        user_agent = build_user_agent(contact)
        config = parse_site_config(
            {
                "name": "preview",
                "entry_urls": [entry_url],
                "contact": contact,
                "selectors": {"body": "body", "title": "title"},
                "rate_limit": {
                    "min_interval_seconds": self.min_interval_seconds,
                    "max_concurrency": 1,
                },
                "sampling": {"max_samples": max_samples},
            }
        )
        fetcher = (self.fetcher_factory or HttpxFetcher)(user_agent=user_agent)
        throttle = DomainThrottle(self.min_interval_seconds)
        try:
            entry = self._fetch_entry(fetcher, throttle, entry_url)
            result = Sampler(fetcher, contact=contact, throttle=throttle).sample(config)
        finally:
            close = getattr(fetcher, "close", None)
            if close:
                close()

        pages = ([entry] if entry else []) + list(result.pages)
        return {
            "entry_url": entry_url,
            "pages": [self._describe(page) for page in pages],
            "templates": self._describe_templates(pages),
            "skipped": [
                {"url": s.url, "reason": s.reason, "rule": s.rule} for s in result.skipped
            ],
            "failures": [{"url": f.url, "reason": f.reason} for f in result.failures],
        }

    def _fetch_entry(self, fetcher, throttle, url: str) -> SampledPage | None:
        """入口页本身也值得看分区——它通常是列表页，与详情页模板不同。"""
        throttle.acquire(url)
        try:
            response = fetcher.get(url)
        except Exception:
            return None
        if response.status >= 400:
            return None
        return SampledPage(url=response.url, html=response.text)

    def _describe(self, page: SampledPage) -> dict:
        import lxml.html

        try:
            document = lxml.html.fromstring(page.html)
        except Exception:
            document = None
        regions = []
        for region in segment_page(page.html).regions:
            snippet = ""
            if document is not None:
                matched = document.cssselect(region.selector)
                if matched:
                    snippet = text_of(matched[0])[:_SNIPPET_CHARS]
            regions.append(
                {
                    "type": region.type.value,
                    "selector": region.selector,
                    "confidence": region.confidence,
                    "note": region.note,
                    "snippet": snippet,
                }
            )
        return {"url": page.url, "bytes": len(page.html), "regions": regions}

    def _describe_templates(self, pages: list[SampledPage]) -> list[dict]:
        samples = [PageSample(url=p.url, html=p.html, column="预览") for p in pages]
        return [
            {
                "template_id": t.template_id,
                "url_shape": t.url_shape,
                "sample_urls": list(t.sample_urls),
                "page_count": len(t.sample_urls),
            }
            for t in cluster_templates(samples)
        ]


REGION_ORDER = [t.value for t in RegionType]
