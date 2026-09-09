"""站点配置草稿的生成与写入。

草稿的格式与爬虫实际消费的配置完全一致——推断结果因此可以直接编辑后投产，
不需要人工在两种格式间转译。

YAML 手工拼装而非 ``yaml.dump``，因为规格要求低置信度字段旁**带提示注释**，
而序列化器不保留注释。
"""

from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Sequence

from kbwb.acquire.selectors import FIELDS, InferenceResult

__all__ = [
    "DraftField",
    "DraftWriteResult",
    "MANUAL_PLACEHOLDER",
    "SiteConfigDraft",
    "build_draft",
    "write_draft",
]

#: 必填选择器无候选时的占位符。刻意醒目——编造一个看似合理的选择器，
#: 会让草稿"看起来能用"，实际一抓即空。
MANUAL_PLACEHOLDER = "TODO:需人工指定"

#: 这两个字段在 schema 中必填，不能留空
_REQUIRED_FIELDS = ("body", "title")

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True, slots=True)
class DraftField:
    name: str
    selector: str | None
    hit_rate: float
    needs_manual: bool
    low_confidence: bool


@dataclass(frozen=True, slots=True)
class SiteConfigDraft:
    yaml_text: str
    fields: tuple[DraftField, ...]

    @property
    def low_confidence_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.low_confidence)

    @property
    def manual_fields(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.needs_manual)


@dataclass(frozen=True, slots=True)
class DraftWriteResult:
    path: Path
    existing_path: Path | None
    renamed: bool
    message: str


def _scalar(value: str) -> str:
    """以单引号包裹，使含 ``"``、``[``、``$`` 的选择器安全落进 YAML。"""
    return "'" + value.replace("'", "''") + "'"


def _build_fields(inference: InferenceResult, threshold: float) -> tuple[DraftField, ...]:
    fields = []
    for name in FIELDS:
        best = inference.fields[name].best
        if best is None:
            fields.append(
                DraftField(
                    name=name,
                    selector=MANUAL_PLACEHOLDER if name in _REQUIRED_FIELDS else None,
                    hit_rate=0.0,
                    needs_manual=True,
                    low_confidence=False,
                )
            )
        else:
            fields.append(
                DraftField(
                    name=name,
                    selector=best.selector,
                    hit_rate=best.hit_rate,
                    needs_manual=False,
                    low_confidence=best.hit_rate < threshold,
                )
            )
    return tuple(fields)


def _selector_line(field: DraftField, samples: int) -> str:
    if field.needs_manual:
        value = _scalar(field.selector) if field.selector else "null"
        return f"  {field.name}: {value}  # 需人工指定：{samples} 个样本中均未定位到该字段"
    line = f"  {field.name}: {_scalar(field.selector)}"
    if field.low_confidence:
        line += f"  # 命中率偏低（{field.hit_rate:.0%}），需人工确认"
    else:
        line += f"  # 命中率 {field.hit_rate:.0%}"
    return line


def build_draft(
    *,
    name: str,
    entry_urls: Sequence[str],
    contact: str,
    inference: InferenceResult,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> SiteConfigDraft:
    fields = _build_fields(inference, confidence_threshold)
    entries = "\n".join(f"  - {_scalar(url)}" for url in entry_urls)
    selectors = "\n".join(_selector_line(field, inference.samples) for field in fields)
    text = f"""# 由站点分析生成的配置草稿，基于 {inference.samples} 个样本页面。
# 注释中的命中率表示该选择器在多少比例的样本上抽取到了非空内容。
# 带「需人工指定」或「命中率偏低」的字段请人工确认后再投入爬取。
name: {_scalar(name)}
entry_urls:
{entries}
contact: {_scalar(contact)}
selectors:
{selectors}
"""
    return SiteConfigDraft(yaml_text=text, fields=fields)


def _free_path(path: Path) -> Path:
    """在不覆盖已有文件的前提下找一个可用路径。"""
    for index in count(1):
        candidate = path.with_name(f"{path.stem}.draft-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise AssertionError("unreachable")  # pragma: no cover


def write_draft(draft: SiteConfigDraft, path: Path | str) -> DraftWriteResult:
    """写出草稿。目标已存在时改写到新路径，并在结果中告知两个路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(draft.yaml_text, encoding="utf-8")
        return DraftWriteResult(
            path=target, existing_path=None, renamed=False, message=f"草稿已写入 {target}"
        )
    destination = _free_path(target)
    destination.write_text(draft.yaml_text, encoding="utf-8")
    return DraftWriteResult(
        path=destination,
        existing_path=target,
        renamed=True,
        message=(
            f"{target} 已存在，未覆盖；草稿写入 {destination}。"
            "请比对两份配置后自行合并。"
        ),
    )
