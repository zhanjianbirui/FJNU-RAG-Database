"""配置模型的边界校验：必填项缺失必须指明环境变量名，而非内部字段名。

宿主环境的隔离由 tests/conftest.py 的 autouse fixture 统一负责。
"""

import pytest
from pydantic import ValidationError

from kbwb.config.settings import ConfigError, RetrievalMode, Settings, load_settings

_REQUIRED = {"CHAT_MODEL": "gpt-4o-mini", "EMBEDDING_MODEL": "text-embedding-3-small"}


def _load(**env):
    return load_settings(_env_file=None, **env)


class TestRequiredFields:
    def test_missing_chat_model_names_the_env_var(self):
        with pytest.raises(ConfigError) as exc:
            _load(embedding_model="text-embedding-3-small")
        message = str(exc.value)
        assert "CHAT_MODEL" in message
        # 报的必须是环境变量名，而不是内部字段名
        assert "chat_model" not in message

    def test_missing_embedding_model_names_the_env_var(self):
        with pytest.raises(ConfigError) as exc:
            _load(chat_model="gpt-4o-mini")
        assert "EMBEDDING_MODEL" in str(exc.value)

    def test_all_missing_required_vars_reported_together(self):
        with pytest.raises(ConfigError) as exc:
            _load()
        message = str(exc.value)
        assert "CHAT_MODEL" in message and "EMBEDDING_MODEL" in message

    def test_loads_when_required_present(self):
        settings = _load(**_REQUIRED)
        assert settings.chat_model == "gpt-4o-mini"
        assert settings.embedding_model == "text-embedding-3-small"


class TestDefaults:
    def test_server_binds_loopback_by_default(self):
        # design.md 要求默认只绑回环，对外监听必须显式配置
        assert _load(**_REQUIRED).host == "127.0.0.1"

    def test_hybrid_retrieval_is_the_default(self):
        # design.md：混合检索是默认而非可选优化
        assert _load(**_REQUIRED).retrieval_mode is RetrievalMode.HYBRID

    def test_rerank_disabled_by_default(self):
        # 精排需下载约 1.1GB 权重，不应默认开启
        assert _load(**_REQUIRED).rerank_enabled is False

    def test_chunking_and_threshold_defaults_present(self):
        settings = _load(**_REQUIRED)
        assert settings.chunk_max_chars > 0
        assert settings.chunk_overlap_chars >= 0
        assert 0 < settings.scanned_ratio_threshold < 1


class TestEnvParsing:
    def test_reads_values_from_environment(self, monkeypatch):
        monkeypatch.setenv("CHAT_MODEL", "deepseek-chat")
        monkeypatch.setenv("EMBEDDING_MODEL", "bge-small-zh")
        monkeypatch.setenv("PORT", "9001")
        monkeypatch.setenv("RETRIEVAL_MODE", "keyword")
        monkeypatch.setenv("RERANK_ENABLED", "true")
        settings = load_settings(_env_file=None)
        assert settings.port == 9001
        assert settings.retrieval_mode is RetrievalMode.KEYWORD
        assert settings.rerank_enabled is True

    def test_unknown_env_vars_are_ignored(self, monkeypatch):
        # 旧的 VECTOR_STORE_PATH 已被 DATA_ROOT 取代，其残留不应导致启动失败
        monkeypatch.setenv("VECTOR_STORE_PATH", "vector_store")
        assert _load(**_REQUIRED).data_root is not None

    def test_invalid_retrieval_mode_names_the_env_var(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, retrieval_mode="graph")
        assert "RETRIEVAL_MODE" in str(exc.value)


class TestCrossFieldValidation:
    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, chunk_max_chars=500, chunk_overlap_chars=500)
        assert "CHUNK_OVERLAP_CHARS" in str(exc.value)

    def test_top_k_must_not_exceed_candidate_k(self):
        # 精排从候选集中截断，top_k 大于候选数说明配置自相矛盾
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, retrieval_top_k=50, retrieval_candidate_k=20)
        assert "RETRIEVAL_TOP_K" in str(exc.value)

    def test_port_out_of_range_rejected(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, port=70000)
        assert "PORT" in str(exc.value)

    def test_threshold_outside_unit_interval_rejected(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, scanned_ratio_threshold=1.5)
        assert "SCANNED_RATIO_THRESHOLD" in str(exc.value)


class TestImmutability:
    def test_settings_are_frozen(self):
        settings = _load(**_REQUIRED)
        with pytest.raises(ValidationError):
            settings.port = 1234


class TestProbeSettings:
    """站点结构探测的配置项。

    site-structure 规格要求探测在可配置的请求总数上限内完成，模板聚类的
    相似度阈值与分区置信度阈值同样需要可调——它们都是启发式判据，写死会
    让不同结构的站点无法适配。
    """

    def test_request_budget_default_is_modest(self):
        # 探测的意义在于比全站爬取便宜一个数量级，默认预算应体现这一点
        assert 0 < _load(**_REQUIRED).probe_request_budget <= 200

    def test_request_budget_must_be_positive(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, probe_request_budget=0)
        assert "PROBE_REQUEST_BUDGET" in str(exc.value)

    def test_samples_per_template_default(self):
        assert _load(**_REQUIRED).samples_per_template >= 1

    def test_samples_per_template_must_be_positive(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, samples_per_template=0)
        assert "SAMPLES_PER_TEMPLATE" in str(exc.value)

    def test_template_similarity_threshold_in_unit_interval(self):
        assert 0 < _load(**_REQUIRED).template_similarity_threshold <= 1

    def test_template_similarity_threshold_rejects_out_of_range(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, template_similarity_threshold=1.5)
        assert "TEMPLATE_SIMILARITY_THRESHOLD" in str(exc.value)

    def test_region_confidence_threshold_in_unit_interval(self):
        assert 0 < _load(**_REQUIRED).region_confidence_threshold <= 1

    def test_region_confidence_threshold_rejects_zero(self):
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, region_confidence_threshold=0)
        assert "REGION_CONFIDENCE_THRESHOLD" in str(exc.value)

    def test_budget_must_allow_at_least_one_sample_round(self):
        """预算小于一轮模板采样所需时，探测必然半途而废——在加载期就拦下。"""
        with pytest.raises(ConfigError) as exc:
            _load(**_REQUIRED, probe_request_budget=2, samples_per_template=5)
        message = str(exc.value)
        assert "PROBE_REQUEST_BUDGET" in message and "SAMPLES_PER_TEMPLATE" in message

    def test_probe_values_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("CHAT_MODEL", "m")
        monkeypatch.setenv("EMBEDDING_MODEL", "e")
        monkeypatch.setenv("PROBE_REQUEST_BUDGET", "45")
        monkeypatch.setenv("TEMPLATE_SIMILARITY_THRESHOLD", "0.82")
        settings = load_settings(_env_file=None)
        assert settings.probe_request_budget == 45
        assert settings.template_similarity_threshold == 0.82
