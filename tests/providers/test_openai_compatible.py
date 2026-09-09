"""OpenAI 与 DeepSeek 的共用实现。

design.md：两者共用一套基于 OpenAI SDK 的实现，差别只是 base_url、模型名与
凭据变量。因此本组用例重点验证「同一实现按 profile 路由到不同端点与凭据」，
以及响应解析在异常结构下不产生冒充成功的空结果。
"""

import pytest

from kbwb.config.settings import load_settings
from kbwb.providers.base import Message, ProviderResponseError
from kbwb.providers.openai_compatible import (
    ClientConfig,
    build_chat_provider,
    build_embedding_provider,
)

from .fixtures import (
    CHAT_RESPONSE,
    CHAT_RESPONSE_EMPTY_CONTENT,
    CHAT_RESPONSE_NO_CHOICES,
    EMBEDDING_RESPONSE,
    EMBEDDING_RESPONSE_OUT_OF_ORDER,
    EMBEDDING_RESPONSE_SHORT,
    as_object,
)

_BASE = {"chat_model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small"}


def _settings(**overrides):
    return load_settings(_env_file=None, **_BASE, **overrides)


class RecordingClient:
    """替身 SDK 客户端：返回录制响应，并记录收到的调用参数。"""

    def __init__(self, chat_payload=CHAT_RESPONSE, embedding_payload=EMBEDDING_RESPONSE):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return as_object(chat_payload)

        class _Embeddings:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return as_object(embedding_payload)

        self.chat = type("_Chat", (), {"completions": _Completions()})()
        self.embeddings = _Embeddings()


@pytest.fixture
def captured():
    """捕获传给客户端工厂的连接配置，用于断言 profile 路由。"""
    seen = {}

    def factory(config: ClientConfig):
        seen["config"] = config
        return RecordingClient()

    factory.seen = seen
    return factory


class TestProfileRouting:
    def test_openai_uses_its_own_credential(self, captured):
        build_chat_provider(
            "openai", _settings(openai_api_key="sk-openai-x"), client_factory=captured
        )
        assert captured.seen["config"].api_key == "sk-openai-x"

    def test_deepseek_uses_its_own_credential_and_base_url(self, captured):
        settings = _settings(deepseek_api_key="sk-deepseek-y")
        build_chat_provider("deepseek", settings, client_factory=captured)
        config = captured.seen["config"]
        assert config.api_key == "sk-deepseek-y"
        assert config.base_url == settings.deepseek_base_url

    def test_openai_base_url_defaults_to_sdk_default(self, captured):
        build_chat_provider("openai", _settings(openai_api_key="sk-x"), client_factory=captured)
        assert captured.seen["config"].base_url is None

    def test_openai_base_url_can_be_overridden(self, captured):
        settings = _settings(openai_api_key="sk-x", openai_base_url="https://proxy.example.cn/v1")
        build_chat_provider("openai", settings, client_factory=captured)
        assert captured.seen["config"].base_url == "https://proxy.example.cn/v1"

    def test_unknown_profile_rejected(self):
        with pytest.raises(ValueError):
            build_chat_provider("anthropic", _settings(openai_api_key="sk-x"))

    def test_provider_name_reflects_the_profile(self, captured):
        provider = build_chat_provider("deepseek", _settings(deepseek_api_key="k"), client_factory=captured)
        assert provider.name == "deepseek"


class TestChatCompletion:
    def _provider(self, payload=CHAT_RESPONSE):
        client = RecordingClient(chat_payload=payload)
        provider = build_chat_provider(
            "openai", _settings(openai_api_key="sk-x"), client_factory=lambda _: client
        )
        return provider, client

    def test_returns_recorded_content(self):
        provider, _ = self._provider()
        answer = provider.complete([Message(role="user", content="学籍证明怎么办")])
        assert answer == "办理学籍证明需携带身份证原件及复印件各一份。"

    def test_sends_configured_model(self):
        provider, client = self._provider()
        provider.complete([Message(role="user", content="x")])
        assert client.calls[0]["model"] == "gpt-4o-mini"

    def test_serialises_messages_as_role_content_pairs(self):
        provider, client = self._provider()
        provider.complete(
            [Message(role="system", content="只依据资料作答"), Message(role="user", content="问题")]
        )
        assert client.calls[0]["messages"] == [
            {"role": "system", "content": "只依据资料作答"},
            {"role": "user", "content": "问题"},
        ]

    def test_max_tokens_passed_only_when_given(self):
        provider, client = self._provider()
        provider.complete([Message(role="user", content="x")])
        assert "max_tokens" not in client.calls[0]
        provider.complete([Message(role="user", content="x")], max_tokens=128)
        assert client.calls[1]["max_tokens"] == 128

    def test_empty_message_list_rejected_before_calling(self):
        provider, client = self._provider()
        with pytest.raises(ValueError):
            provider.complete([])
        assert client.calls == []

    def test_response_without_choices_is_an_error(self):
        provider, _ = self._provider(CHAT_RESPONSE_NO_CHOICES)
        with pytest.raises(ProviderResponseError):
            provider.complete([Message(role="user", content="x")])

    def test_empty_content_is_an_error_not_a_result(self):
        # 规格：不得返回空字符串冒充成功
        provider, _ = self._provider(CHAT_RESPONSE_EMPTY_CONTENT)
        with pytest.raises(ProviderResponseError):
            provider.complete([Message(role="user", content="x")])


class TestEmbedding:
    def _provider(self, payload=EMBEDDING_RESPONSE):
        client = RecordingClient(embedding_payload=payload)
        provider = build_embedding_provider(
            "openai", _settings(openai_api_key="sk-x"), client_factory=lambda _: client
        )
        return provider, client

    def test_returns_vectors_for_each_input(self):
        provider, _ = self._provider()
        vectors = provider.embed(["甲", "乙"])
        assert vectors == [[0.11, 0.22, 0.33, 0.44], [0.55, 0.66, 0.77, 0.88]]

    def test_reorders_by_index(self):
        provider, _ = self._provider(EMBEDDING_RESPONSE_OUT_OF_ORDER)
        vectors = provider.embed(["甲", "乙"])
        assert vectors[0] == [0.11, 0.22, 0.33, 0.44]

    def test_sends_configured_model(self):
        provider, client = self._provider()
        provider.embed(["甲", "乙"])
        assert client.calls[0]["model"] == "text-embedding-3-small"

    def test_empty_batch_skips_the_call(self):
        provider, client = self._provider()
        assert provider.embed([]) == []
        assert client.calls == []

    def test_dimension_discovered_from_first_response(self):
        provider, _ = self._provider()
        provider.embed(["甲", "乙"])
        assert provider.dimension == 4

    def test_count_mismatch_is_an_error(self):
        # 少返回一条时若静默补零，会污染整个索引
        provider, _ = self._provider(EMBEDDING_RESPONSE_SHORT)
        with pytest.raises(ProviderResponseError):
            provider.embed(["甲", "乙"])

    def test_blank_input_rejected_before_calling(self):
        provider, client = self._provider()
        with pytest.raises(ValueError):
            provider.embed(["甲", "   "])
        assert client.calls == []
