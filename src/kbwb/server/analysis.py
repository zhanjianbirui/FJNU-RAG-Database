"""把站点分析链路包装成工作台可调用的服务。

采样 → 选择器推断 → 配置草稿三步在此串起，并转成界面直接可渲染的结构。
"""

from dataclasses import dataclass
from typing import Any

from kbwb.acquire.draft import build_draft
from kbwb.acquire.fetching import DomainThrottle
from kbwb.acquire.http_client import HttpxFetcher
from kbwb.acquire.sampling import Sampler, build_user_agent
from kbwb.acquire.selectors import infer_selectors
from kbwb.config.site import parse_site_config

__all__ = ["SiteAnalysisService"]

DEFAULT_MAX_SAMPLES = 6
DEFAULT_MIN_INTERVAL = 2.0


@dataclass(frozen=True, slots=True)
class SiteAnalysisService:
    """默认走真实网络；测试与演示可注入自己的 fetcher 工厂。"""

    min_interval_seconds: float = DEFAULT_MIN_INTERVAL
    fetcher_factory: Any = None

    def analyze(self, *, entry_url: str, contact: str, max_samples: int) -> dict:
        user_agent = build_user_agent(contact)
        config = parse_site_config(
            {
                "name": "draft",
                "entry_urls": [entry_url],
                "contact": contact,
                # 分析阶段用不到抽取，占位以通过 schema
                "selectors": {"body": "body", "title": "title"},
                "rate_limit": {
                    "min_interval_seconds": self.min_interval_seconds,
                    "max_concurrency": 1,
                },
                "sampling": {"max_samples": max_samples},
            }
        )
        fetcher = (self.fetcher_factory or HttpxFetcher)(user_agent=user_agent)
        try:
            result = Sampler(
                fetcher,
                contact=contact,
                throttle=DomainThrottle(self.min_interval_seconds),
            ).sample(config)
        finally:
            close = getattr(fetcher, "close", None)
            if close:
                close()

        inference = infer_selectors(result.pages)
        draft = build_draft(
            name="draft",
            entry_urls=config.entry_urls,
            contact=contact,
            inference=inference,
        )
        return {
            "entry_url": entry_url,
            "sampled": [{"url": p.url, "bytes": len(p.html)} for p in result.pages],
            "skipped": [
                {"url": s.url, "reason": s.reason, "rule": s.rule} for s in result.skipped
            ],
            "failures": [{"url": f.url, "reason": f.reason} for f in result.failures],
            "fields": [
                {
                    "field": name,
                    "needs_manual": field.needs_manual,
                    "candidates": [
                        {
                            "selector": c.selector,
                            "hit_rate": c.hit_rate,
                            "hits": c.hits,
                            "samples": c.samples,
                        }
                        for c in field.candidates[:5]
                    ],
                }
                for name, field in inference.fields.items()
            ],
            "draft_yaml": draft.yaml_text,
        }
