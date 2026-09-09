"""站点配置 schema 与 YAML 加载校验。

web-crawling 规格要求：配置非法时，系统在发起任何网络请求之前中止，
并报告**具体的字段路径**与错误原因。因此本组用例的核心断言是错误信息
中出现点分字段路径（如 ``rate_limit.max_concurrency``）。
"""

import pytest

from kbwb.config.site import SiteConfig, SiteConfigError, load_site_config, parse_site_config

MINIMAL = {
    "name": "fjnu-jwc",
    "entry_urls": ["https://jwc.fjnu.edu.cn/tzgg.htm"],
    "contact": "mailto:admin@example.edu.cn",
    "selectors": {"body": "div.content", "title": "h1"},
}


def _write(tmp_path, text):
    path = tmp_path / "site.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestValidConfig:
    def test_minimal_config_parses(self):
        config = parse_site_config(MINIMAL)
        assert isinstance(config, SiteConfig)
        assert config.name == "fjnu-jwc"
        assert config.selectors.body == "div.content"

    def test_defaults_are_conservative(self):
        config = parse_site_config(MINIMAL)
        # 合规约束的默认值必须是保守的：低并发、有间隔、有采样上限
        assert config.rate_limit.max_concurrency >= 1
        assert config.rate_limit.min_interval_seconds > 0
        assert config.sampling.max_samples > 0
        assert config.attachments.max_bytes > 0

    def test_optional_selectors_may_be_unspecified(self):
        # site-analysis 规格：无可用候选的字段标记为"需人工指定"，而非填零命中率的选择器
        config = parse_site_config({**MINIMAL, "selectors": {"body": "div.c", "title": "h1", "department": None}})
        assert config.selectors.department is None

    def test_loads_from_yaml_file(self, tmp_path):
        path = _write(tmp_path, """
name: fjnu-jwc
entry_urls:
  - https://jwc.fjnu.edu.cn/tzgg.htm
contact: mailto:admin@example.edu.cn
selectors:
  body: div.content
  title: h1.title
rate_limit:
  min_interval_seconds: 2.0
  max_concurrency: 1
""")
        config = load_site_config(path)
        assert config.rate_limit.min_interval_seconds == 2.0
        assert config.rate_limit.max_concurrency == 1


class TestFieldPathReporting:
    def test_missing_required_field_reports_path(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({k: v for k, v in MINIMAL.items() if k != "entry_urls"})
        assert "entry_urls" in str(exc.value)

    def test_missing_nested_required_field_reports_dotted_path(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "selectors": {"title": "h1"}})
        assert "selectors.body" in str(exc.value)

    def test_nested_type_error_reports_dotted_path(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "rate_limit": {"max_concurrency": "many"}})
        assert "rate_limit.max_concurrency" in str(exc.value)

    def test_list_item_error_reports_index_in_path(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "entry_urls": ["https://ok.example.cn/", "ftp://bad.example.cn/"]})
        assert "entry_urls.1" in str(exc.value)

    def test_unknown_field_is_rejected_with_its_path(self):
        # 拼错的字段若被静默忽略，用户会以为配置生效了
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "selectorz": {}})
        assert "selectorz" in str(exc.value)

    def test_all_problems_reported_together(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({"name": "x"})
        message = str(exc.value)
        assert "entry_urls" in message and "contact" in message and "selectors" in message


class TestComplianceConstraints:
    def test_zero_interval_rejected(self):
        # 间隔为 0 等于取消限速，与合规约束冲突
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "rate_limit": {"min_interval_seconds": 0}})
        assert "rate_limit.min_interval_seconds" in str(exc.value)

    def test_non_positive_concurrency_rejected(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "rate_limit": {"max_concurrency": 0}})
        assert "rate_limit.max_concurrency" in str(exc.value)

    def test_contact_is_required_for_identifiable_user_agent(self):
        # 规格要求 User-Agent 含可联系的 URL 或邮箱
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({k: v for k, v in MINIMAL.items() if k != "contact"})
        assert "contact" in str(exc.value)

    def test_contact_must_be_url_or_mailto(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "contact": "找小王"})
        assert "contact" in str(exc.value)

    def test_non_http_entry_url_rejected(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "entry_urls": ["file:///etc/passwd"]})
        assert "entry_urls" in str(exc.value)

    def test_empty_entry_urls_rejected(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "entry_urls": []})
        assert "entry_urls" in str(exc.value)


class TestPatternValidation:
    def test_invalid_regex_reports_path(self):
        with pytest.raises(SiteConfigError) as exc:
            parse_site_config({**MINIMAL, "allow_patterns": ["^https://ok", "[unclosed"]})
        assert "allow_patterns.1" in str(exc.value)

    def test_valid_patterns_accepted(self):
        config = parse_site_config({**MINIMAL, "allow_patterns": [r"^https://jwc\.fjnu\.edu\.cn/"], "deny_patterns": [r"\.jpg$"]})
        assert len(config.allow_patterns) == 1
        assert len(config.deny_patterns) == 1


class TestYamlLoading:
    def test_malformed_yaml_reports_clearly(self, tmp_path):
        path = _write(tmp_path, "name: [unclosed\n")
        with pytest.raises(SiteConfigError) as exc:
            load_site_config(path)
        assert str(path) in str(exc.value)

    def test_non_mapping_document_rejected(self, tmp_path):
        path = _write(tmp_path, "- just\n- a list\n")
        with pytest.raises(SiteConfigError):
            load_site_config(path)

    def test_empty_document_rejected(self, tmp_path):
        path = _write(tmp_path, "")
        with pytest.raises(SiteConfigError):
            load_site_config(path)

    def test_missing_file_reports_path(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        with pytest.raises(SiteConfigError) as exc:
            load_site_config(missing)
        assert str(missing) in str(exc.value)


class TestImmutability:
    def test_config_is_frozen(self):
        config = parse_site_config(MINIMAL)
        with pytest.raises(Exception):
            config.name = "other"

    def test_nested_model_is_frozen(self):
        config = parse_site_config(MINIMAL)
        with pytest.raises(Exception):
            config.rate_limit.max_concurrency = 99
