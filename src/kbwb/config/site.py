"""站点配置 schema 与 YAML 加载。

web-crawling 规格要求接入新站点 MUST NOT 需要改代码，因此爬取行为——站点
清单、准入与排除规则、分页、字段选择器、限速与附件策略——全部由本模块定义
的配置承载。

规格同时要求配置非法时在**发起任何网络请求之前**中止并报出具体字段路径，
所以校验集中在加载边界，错误信息以点分路径（``rate_limit.max_concurrency``）
定位，并一次性列出全部问题。
"""

import re
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "AttachmentPolicy",
    "PaginationRule",
    "RateLimitPolicy",
    "SamplingPolicy",
    "SelectorSet",
    "SiteConfig",
    "SiteConfigError",
    "load_site_config",
    "parse_site_config",
]

_ALLOWED_URL_SCHEMES = ("http", "https")
_ALLOWED_CONTACT_SCHEMES = ("http", "https", "mailto")


class SiteConfigError(ValueError):
    """站点配置缺失、类型错误或无法解析。信息以字段路径定位问题。"""


def _check_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"只接受 http/https 地址，收到 {value!r}")
    if not parsed.netloc:
        raise ValueError(f"地址缺少主机名：{value!r}")
    return value


def _check_regex(value: str) -> str:
    try:
        re.compile(value)
    except re.error as exc:
        raise ValueError(f"不是合法的正则表达式（{exc}）：{value!r}") from exc
    return value


HttpUrlStr = Annotated[str, AfterValidator(_check_http_url)]
RegexStr = Annotated[str, AfterValidator(_check_regex)]


class _Frozen(BaseModel):
    """所有站点配置节点共用：不可变，且拒绝未知字段。

    拒绝未知字段是有意的——拼错的键若被静默忽略，用户会以为配置已生效。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class SelectorSet(_Frozen):
    """字段抽取选择器。正文与标题必填，其余可留空表示需人工指定。"""

    body: str
    title: str
    date: str | None = None
    department: str | None = None
    attachment: str | None = None


class PaginationRule(_Frozen):
    next_selector: str | None = None
    max_pages: int = Field(default=20, ge=1)


class RateLimitPolicy(_Frozen):
    """合规约束。间隔与并发的下限由 schema 保证，不可被配置绕过。"""

    min_interval_seconds: float = Field(default=1.0, gt=0.0)
    max_concurrency: int = Field(default=2, ge=1)
    max_retries: int = Field(default=3, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class AttachmentPolicy(_Frozen):
    enabled: bool = True
    max_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    extensions: tuple[str, ...] = ("pdf", "doc", "docx", "xls", "xlsx")


class SamplingPolicy(_Frozen):
    """站点分析的采样规模。有默认上限，避免对目标站点造成压力。"""

    max_samples: int = Field(default=20, ge=1)
    confidence_threshold: float = Field(default=0.6, gt=0.0, le=1.0)


class SiteConfig(_Frozen):
    """一个站点的完整爬取配置。"""

    name: str = Field(min_length=1)
    entry_urls: tuple[HttpUrlStr, ...] = Field(min_length=1)
    # User-Agent 需含可供站点管理员联系的 URL 或邮箱（web-crawling 规格）
    contact: str
    selectors: SelectorSet
    allow_patterns: tuple[RegexStr, ...] = ()
    deny_patterns: tuple[RegexStr, ...] = ()
    pagination: PaginationRule = PaginationRule()
    rate_limit: RateLimitPolicy = RateLimitPolicy()
    attachments: AttachmentPolicy = AttachmentPolicy()
    sampling: SamplingPolicy = SamplingPolicy()


def _check_contact(value: object) -> None:
    if not isinstance(value, str) or urlparse(value).scheme not in _ALLOWED_CONTACT_SCHEMES:
        raise SiteConfigError(
            "配置校验失败：\n  - contact: 必须是 http(s) 地址或 mailto: 邮箱，"
            "以便站点管理员据此联系（收到 %r）" % (value,)
        )


def _path(location: tuple) -> str:
    return ".".join(str(part) for part in location) or "<根>"


def _format(exc: ValidationError) -> str:
    lines = sorted(f"{_path(err['loc'])}: {err['msg']}" for err in exc.errors())
    return "配置校验失败：\n" + "\n".join(f"  - {line}" for line in lines)


def parse_site_config(data: object) -> SiteConfig:
    """校验并构造站点配置。非法时抛出 :class:`SiteConfigError` 并列出全部问题。"""
    if not isinstance(data, dict):
        raise SiteConfigError(
            f"站点配置的顶层必须是键值映射，收到 {type(data).__name__}"
        )
    if "contact" in data:
        _check_contact(data["contact"])
    try:
        return SiteConfig(**data)
    except ValidationError as exc:
        raise SiteConfigError(_format(exc)) from exc


def load_site_config(path: Path | str) -> SiteConfig:
    """从 YAML 文件加载站点配置。文件问题与字段问题都以该路径为线索报告。"""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SiteConfigError(f"无法读取站点配置 {file_path}：{exc}") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SiteConfigError(f"站点配置 {file_path} 不是合法的 YAML：{exc}") from exc
    if document is None:
        raise SiteConfigError(f"站点配置 {file_path} 为空")
    try:
        return parse_site_config(document)
    except SiteConfigError as exc:
        raise SiteConfigError(f"{file_path}\n{exc}") from exc
