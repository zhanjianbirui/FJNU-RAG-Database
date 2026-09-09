"""站点配置草稿生成与写入。

site-analysis 规格三条：
- 草稿格式 MUST 与爬虫实际消费的格式一致，即能通过 1.5 的配置校验；
- 最高候选命中率低于置信度阈值的字段，旁边 MUST 带提示注释；
- MUST NOT 静默覆盖已存在的配置，需写入新路径并告知用户两个路径。
"""

import pytest
import yaml

from kbwb.acquire.draft import (
    MANUAL_PLACEHOLDER,
    build_draft,
    write_draft,
)
from kbwb.acquire.selectors import FieldInference, InferenceResult, SelectorCandidate

CONTACT = "mailto:admin@example.edu.cn"
ENTRIES = ("https://jwc.example.edu.cn/tzgg.htm",)


def _inference(spec, samples=5):
    """spec: {字段: (选择器, 命中数)}；未列出的字段视为无候选。"""
    fields = {}
    for name in ("body", "title", "date", "department", "attachment"):
        if name in spec:
            selector, hits = spec[name]
            candidates = (SelectorCandidate(selector=selector, hits=hits, samples=samples),)
        else:
            candidates = ()
        fields[name] = FieldInference(field=name, candidates=candidates)
    return InferenceResult(samples=samples, fields=fields)


FULL = {
    "body": ("div.article", 5),
    "title": ("h1.title", 5),
    "date": ("span.date", 5),
    "department": ("span.dept", 5),
    "attachment": ('div.attach a[href$=".pdf"]', 5),
}


def _draft(spec=None, **kwargs):
    return build_draft(
        name="jwc",
        entry_urls=ENTRIES,
        contact=CONTACT,
        inference=_inference(spec if spec is not None else FULL),
        **kwargs,
    )


class TestDraftValidity:
    def test_draft_is_valid_yaml(self):
        assert isinstance(yaml.safe_load(_draft().yaml_text), dict)

    def test_draft_passes_site_config_validation(self):
        from kbwb.config.site import parse_site_config

        config = parse_site_config(yaml.safe_load(_draft().yaml_text))
        assert config.name == "jwc"
        assert config.selectors.body == "div.article"

    def test_draft_written_to_disk_loads_back(self, tmp_path):
        from kbwb.config.site import load_site_config

        outcome = write_draft(_draft(), tmp_path / "site.yaml")
        assert load_site_config(outcome.path).selectors.title == "h1.title"

    def test_entry_urls_and_contact_preserved(self):
        data = yaml.safe_load(_draft().yaml_text)
        assert data["entry_urls"] == list(ENTRIES)
        assert data["contact"] == CONTACT

    def test_selector_with_quotes_survives_round_trip(self):
        data = yaml.safe_load(_draft().yaml_text)
        assert data["selectors"]["attachment"] == 'div.attach a[href$=".pdf"]'


class TestConfidenceComments:
    def test_low_confidence_field_gets_a_comment(self):
        draft = _draft({**FULL, "date": ("span.date", 1)}, confidence_threshold=0.6)
        line = next(l for l in draft.yaml_text.splitlines() if "span.date" in l)
        assert "#" in line and "20%" in line

    def test_high_confidence_field_has_no_warning_comment(self):
        draft = _draft(confidence_threshold=0.6)
        line = next(l for l in draft.yaml_text.splitlines() if "h1.title" in l)
        assert "命中率偏低" not in line

    def test_low_confidence_fields_are_listed(self):
        draft = _draft({**FULL, "date": ("span.date", 2)}, confidence_threshold=0.6)
        assert "date" in draft.low_confidence_fields

    def test_comment_does_not_break_parsing(self):
        draft = _draft({**FULL, "date": ("span.date", 1)})
        assert yaml.safe_load(draft.yaml_text)["selectors"]["date"] == "span.date"


class TestManualFields:
    def test_optional_field_without_candidate_is_null(self):
        draft = _draft({k: v for k, v in FULL.items() if k != "department"})
        assert yaml.safe_load(draft.yaml_text)["selectors"]["department"] is None

    def test_manual_field_carries_a_comment(self):
        draft = _draft({k: v for k, v in FULL.items() if k != "department"})
        line = next(l for l in draft.yaml_text.splitlines() if "department" in l)
        assert "需人工指定" in line

    def test_required_field_without_candidate_uses_a_visible_placeholder(self):
        """body 是必填，不能留空；用醒目的占位符而非编造一个选择器。"""
        draft = _draft({k: v for k, v in FULL.items() if k != "body"})
        assert yaml.safe_load(draft.yaml_text)["selectors"]["body"] == MANUAL_PLACEHOLDER

    def test_draft_with_placeholder_still_validates(self):
        from kbwb.config.site import parse_site_config

        draft = _draft({k: v for k, v in FULL.items() if k != "body"})
        parse_site_config(yaml.safe_load(draft.yaml_text))

    def test_manual_fields_are_listed(self):
        draft = _draft({"title": ("h1", 5)})
        assert set(draft.manual_fields) >= {"body", "date", "department", "attachment"}


class TestNonDestructiveWrite:
    def test_writes_to_the_requested_path_when_free(self, tmp_path):
        target = tmp_path / "site.yaml"
        outcome = write_draft(_draft(), target)
        assert outcome.path == target
        assert outcome.renamed is False
        assert outcome.existing_path is None

    def test_existing_file_is_not_overwritten(self, tmp_path):
        target = tmp_path / "site.yaml"
        target.write_text("原有配置", encoding="utf-8")
        write_draft(_draft(), target)
        assert target.read_text(encoding="utf-8") == "原有配置"

    def test_draft_goes_to_a_new_path(self, tmp_path):
        target = tmp_path / "site.yaml"
        target.write_text("原有配置", encoding="utf-8")
        outcome = write_draft(_draft(), target)
        assert outcome.path != target
        assert outcome.path.exists()
        assert outcome.renamed is True

    def test_outcome_names_both_paths(self, tmp_path):
        target = tmp_path / "site.yaml"
        target.write_text("原有配置", encoding="utf-8")
        outcome = write_draft(_draft(), target)
        assert outcome.existing_path == target
        assert str(target) in outcome.message and str(outcome.path) in outcome.message

    def test_repeated_runs_keep_finding_new_paths(self, tmp_path):
        target = tmp_path / "site.yaml"
        target.write_text("原有配置", encoding="utf-8")
        first = write_draft(_draft(), target)
        second = write_draft(_draft(), target)
        assert first.path != second.path
        assert first.path.exists() and second.path.exists()

    def test_parent_directory_is_created(self, tmp_path):
        outcome = write_draft(_draft(), tmp_path / "nested" / "deep" / "site.yaml")
        assert outcome.path.exists()

    def test_outcome_is_immutable(self, tmp_path):
        outcome = write_draft(_draft(), tmp_path / "site.yaml")
        with pytest.raises(Exception):
            outcome.renamed = False
