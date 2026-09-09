"""Provider 契约与注册表。

model-providers 规格的两条核心要求：

1. 更换 provider 不需改动调用方代码；
2. 未知 provider 名称在**启动时**报错并列出所有可用名称，而非首次调用才失败。
"""

import pytest

from kbwb.config.settings import load_settings
from kbwb.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    Message,
    ProviderError,
)
from kbwb.providers.registry import (
    UnknownProviderError,
    available_chat_providers,
    available_embedding_providers,
    build_providers,
    create_chat_provider,
    create_embedding_provider,
    register_chat_provider,
    register_embedding_provider,
)

_REQUIRED = {"chat_model": "m-chat", "embedding_model": "m-embed"}


def _settings(**overrides):
    return load_settings(_env_file=None, **_REQUIRED, **overrides)


class FakeChat(ChatProvider):
    name = "fake-chat"

    def complete(self, messages, *, max_tokens=None):
        return f"echo:{messages[-1].content}"


class FakeEmbedding(EmbeddingProvider):
    name = "fake-embedding"
    dimension = 3

    def embed(self, texts):
        return [[float(len(text)), 0.0, 1.0] for text in texts]


@pytest.fixture
def registered():
    """在隔离的注册表上注册假实现，避免污染内置注册项。"""
    register_chat_provider("fake-chat", lambda settings: FakeChat())
    register_embedding_provider("fake-embedding", lambda settings: FakeEmbedding())
    yield
    # 注册表按名覆盖，重复注册无副作用；此处无需清理内置项。


class TestContractShape:
    def test_chat_provider_is_abstract(self):
        with pytest.raises(TypeError):
            ChatProvider()

    def test_embedding_provider_is_abstract(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_incomplete_chat_implementation_cannot_be_instantiated(self):
        class Incomplete(ChatProvider):
            name = "incomplete"

        with pytest.raises(TypeError):
            Incomplete()

    def test_message_is_immutable(self):
        message = Message(role="user", content="你好")
        with pytest.raises(Exception):
            message.content = "改了"

    def test_message_rejects_unknown_role(self):
        with pytest.raises(ValueError):
            Message(role="wizard", content="x")


class TestSwappingProviders:
    def test_caller_uses_chat_through_the_contract(self, registered):
        provider = create_chat_provider("fake-chat", _settings())
        assert isinstance(provider, ChatProvider)
        assert provider.complete([Message(role="user", content="在吗")]) == "echo:在吗"

    def test_embedding_returns_one_vector_per_input(self, registered):
        provider = create_embedding_provider("fake-embedding", _settings())
        vectors = provider.embed(["a", "bb", "ccc"])
        assert len(vectors) == 3
        assert {len(v) for v in vectors} == {provider.dimension}

    def test_embedding_of_empty_batch_is_empty(self, registered):
        provider = create_embedding_provider("fake-embedding", _settings())
        assert provider.embed([]) == []


class TestUnknownProviderName:
    def test_unknown_chat_name_raises(self, registered):
        with pytest.raises(UnknownProviderError):
            create_chat_provider("nope", _settings())

    def test_error_lists_available_names(self, registered):
        with pytest.raises(UnknownProviderError) as exc:
            create_chat_provider("nope", _settings())
        message = str(exc.value)
        for name in available_chat_providers():
            assert name in message

    def test_error_names_the_rejected_value(self, registered):
        with pytest.raises(UnknownProviderError) as exc:
            create_embedding_provider("nope", _settings())
        assert "nope" in str(exc.value)

    def test_unknown_embedding_name_lists_embedding_names_only(self, registered):
        with pytest.raises(UnknownProviderError) as exc:
            create_embedding_provider("nope", _settings())
        message = str(exc.value)
        for name in available_embedding_providers():
            assert name in message

    def test_unknown_provider_error_is_a_provider_error(self, registered):
        assert issubclass(UnknownProviderError, ProviderError)


class TestStartupValidation:
    def test_build_providers_fails_at_startup_not_first_call(self, registered):
        """规格要求：未知名称在启动阶段暴露，而不是等到首次调用。"""
        with pytest.raises(UnknownProviderError):
            build_providers(_settings(chat_provider="nope"))

    def test_build_providers_reports_unknown_embedding_name(self, registered):
        with pytest.raises(UnknownProviderError):
            build_providers(_settings(embedding_provider="nope"))

    def test_build_providers_returns_both_providers(self, registered):
        chat, embedding = build_providers(
            _settings(chat_provider="fake-chat", embedding_provider="fake-embedding")
        )
        assert isinstance(chat, ChatProvider)
        assert isinstance(embedding, EmbeddingProvider)

    def test_chat_and_embedding_are_independently_configurable(self, registered):
        """混合搭配：chat 与 embedding 可来自不同供应方。"""
        chat, embedding = build_providers(
            _settings(chat_provider="fake-chat", embedding_provider="fake-embedding")
        )
        assert chat.name != embedding.name


class TestBuiltinRegistration:
    def test_builtin_names_are_registered(self):
        assert "openai" in available_chat_providers()
        assert "deepseek" in available_chat_providers()
        assert "openai" in available_embedding_providers()
        assert "deepseek" in available_embedding_providers()
        assert "local" in available_embedding_providers()

    def test_local_is_embedding_only(self):
        # 本地模型只提供嵌入，不提供对话生成
        assert "local" not in available_chat_providers()

    def test_available_names_are_sorted_and_stable(self):
        assert list(available_chat_providers()) == sorted(available_chat_providers())
