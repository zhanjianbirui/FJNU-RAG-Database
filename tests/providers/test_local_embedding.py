"""本地 Sentence-Transformers 嵌入 provider。

model-providers 规格：配置 embedding 为本地模型且无外网连接时，索引构建与
检索仍可正常完成。因此关键用例在**禁用网络**的前提下断言嵌入成功。

sentence-transformers 属可选依赖组 local-models，默认不安装；用例通过注入
模型工厂验证 provider 自身逻辑，不依赖该包是否就位。
"""

import socket

import pytest

from kbwb.config.settings import load_settings
from kbwb.providers.base import EmbeddingProvider, ProviderError, ProviderResponseError
from kbwb.providers.local_embedding import (
    LocalEmbeddingProvider,
    MissingLocalModelDependency,
    build_local_embedding_provider,
)

_BASE = {"chat_model": "m-chat", "embedding_model": "m-embed"}


def _settings(**overrides):
    return load_settings(_env_file=None, **_BASE, **overrides)


class FakeSentenceTransformer:
    """替身模型：确定性输出，不加载权重、不联网。"""

    def __init__(self, dimension=4):
        self._dimension = dimension
        self.encoded = []

    def encode(self, texts, **kwargs):
        self.encoded.append(list(texts))
        return [[float(len(t)), 0.5, 0.25, 0.125][: self._dimension] for t in texts]

    def get_sentence_embedding_dimension(self):
        return self._dimension


@pytest.fixture
def no_network(monkeypatch):
    """禁用一切出站连接，确保本地嵌入路径确实不触网。"""

    def _blocked(*args, **kwargs):
        raise AssertionError("本地嵌入不应发起网络连接")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def _provider(model=None, settings=None):
    model = model or FakeSentenceTransformer()
    return build_local_embedding_provider(
        settings or _settings(), model_factory=lambda name: model
    )


class TestOfflineEmbedding:
    def test_embeds_without_network(self, no_network):
        provider = _provider()
        vectors = provider.embed(["办事指南", "学籍证明"])
        assert len(vectors) == 2
        assert all(len(v) == 4 for v in vectors)

    def test_is_an_embedding_provider(self):
        assert isinstance(_provider(), EmbeddingProvider)
        assert isinstance(_provider(), LocalEmbeddingProvider)

    def test_dimension_reported_from_model(self):
        assert _provider(FakeSentenceTransformer(dimension=3)).dimension == 3

    def test_uses_configured_model_name(self):
        seen = {}

        def factory(name):
            seen["name"] = name
            return FakeSentenceTransformer()

        settings = _settings(local_embedding_model="BAAI/bge-small-zh-v1.5")
        build_local_embedding_provider(settings, model_factory=factory)
        assert seen["name"] == "BAAI/bge-small-zh-v1.5"

    def test_provider_name_is_local(self):
        assert _provider().name == "local"


class TestBatching:
    def test_preserves_input_order(self, no_network):
        model = FakeSentenceTransformer()
        provider = _provider(model)
        provider.embed(["甲", "乙丙"])
        assert model.encoded[0] == ["甲", "乙丙"]

    def test_empty_batch_skips_the_model(self):
        model = FakeSentenceTransformer()
        assert _provider(model).embed([]) == []
        assert model.encoded == []

    def test_blank_input_rejected(self):
        with pytest.raises(ValueError):
            _provider().embed(["正常", "  "])


class TestFailureHandling:
    def test_model_returning_wrong_count_is_an_error(self):
        class Broken(FakeSentenceTransformer):
            def encode(self, texts, **kwargs):
                return [[0.1, 0.2, 0.3, 0.4]]

        with pytest.raises(ProviderResponseError):
            _provider(Broken()).embed(["甲", "乙"])

    def test_model_returning_empty_vector_is_an_error(self):
        class Broken(FakeSentenceTransformer):
            def encode(self, texts, **kwargs):
                return [[] for _ in texts]

        with pytest.raises(ProviderResponseError):
            _provider(Broken()).embed(["甲"])

    def test_model_load_failure_is_a_provider_error(self):
        def failing_factory(name):
            raise OSError("权重文件损坏")

        with pytest.raises(ProviderError):
            build_local_embedding_provider(_settings(), model_factory=failing_factory)


class TestOptionalDependency:
    def test_missing_dependency_names_the_extra_group(self, monkeypatch):
        import kbwb.providers.local_embedding as module

        def _raise(name):
            raise ImportError("No module named 'sentence_transformers'")

        monkeypatch.setattr(module, "_load_sentence_transformer", _raise)
        with pytest.raises(MissingLocalModelDependency) as exc:
            build_local_embedding_provider(_settings())
        message = str(exc.value)
        assert "local-models" in message
        assert "sentence-transformers" in message

    def test_missing_dependency_error_is_a_provider_error(self):
        assert issubclass(MissingLocalModelDependency, ProviderError)
