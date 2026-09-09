"""OpenAI 与 DeepSeek 的共用实现。

两家都提供 OpenAI 兼容接口，差别只有三处：``base_url``、模型名与凭据变量。
因此这里用一套代码 + 一张 profile 表覆盖，而不是写两个近乎相同的类
（见 design.md 的 provider 抽象决策）。

抽象层真正的价值在于容纳协议不同的实现——本地嵌入在 ``local_embedding``
模块中单独实现。
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kbwb.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    Message,
    ProviderResponseError,
)
from kbwb.providers.credentials import collect_secrets, redact
from kbwb.providers.retry import RetryPolicy, call_with_retry

if TYPE_CHECKING:  # pragma: no cover
    from kbwb.config.settings import Settings

__all__ = [
    "ClientConfig",
    "OpenAICompatibleChatProvider",
    "OpenAICompatibleEmbeddingProvider",
    "PROFILES",
    "ProviderProfile",
    "build_chat_provider",
    "build_embedding_provider",
]


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """一家 OpenAI 兼容供应方的接入参数来源。"""

    name: str
    api_key_field: str
    api_key_env: str
    base_url_field: str | None


PROFILES: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        name="openai",
        api_key_field="openai_api_key",
        api_key_env="OPENAI_API_KEY",
        # 留空则使用 SDK 默认端点
        base_url_field="openai_base_url",
    ),
    "deepseek": ProviderProfile(
        name="deepseek",
        api_key_field="deepseek_api_key",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_field="deepseek_base_url",
    ),
}


@dataclass(frozen=True, slots=True)
class ClientConfig:
    """构造 SDK 客户端所需的连接信息。"""

    profile: str
    api_key: str | None
    base_url: str | None


ClientFactory = Callable[[ClientConfig], Any]


def _default_client_factory(config: ClientConfig) -> Any:
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return OpenAI(**kwargs)


def _resolve_profile(name: str) -> ProviderProfile:
    profile = PROFILES.get(name)
    if profile is None:
        raise ValueError(
            f"未知的 OpenAI 兼容 profile {name!r}，可用：{', '.join(sorted(PROFILES))}"
        )
    return profile


def _client_config(profile: ProviderProfile, settings: "Settings") -> ClientConfig:
    base_url = getattr(settings, profile.base_url_field, None) if profile.base_url_field else None
    return ClientConfig(
        profile=profile.name,
        api_key=getattr(settings, profile.api_key_field, None),
        base_url=base_url or None,
    )


def _call_redacted(operation, policy, sleep, secrets):
    """执行调用；错误向上传递前先脱敏，保持原有错误类型不变。

    SDK 的异常文本常回显 Authorization 头，不脱敏就会把凭据写进日志。
    """
    try:
        return call_with_retry(operation, policy, sleep=sleep)
    except Exception as exc:
        cleaned = redact(str(exc), secrets)
        if cleaned == str(exc):
            raise
        raise type(exc)(cleaned) from None


class OpenAICompatibleChatProvider(ChatProvider):
    def __init__(
        self,
        *,
        name: str,
        client: Any,
        model: str,
        policy: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._client = client
        self._model = model
        self._policy = policy
        self._sleep = sleep
        self._secrets = secrets

    def complete(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> str:
        if not messages:
            raise ValueError("messages 不能为空")
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        # 只有网络调用进入重试；响应解析失败不可重试，留在外面
        response = _call_redacted(
            lambda: self._client.chat.completions.create(**payload),
            self._policy,
            self._sleep,
            self._secrets,
        )
        return self._extract(response)

    def _extract(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ProviderResponseError(
                f"{self.name} 返回的响应不含任何 choices，无法取得回复"
            )
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not content:
            # 空回复不是有效结果；静默返回空串会让上层误以为模型确实这么答
            raise ProviderResponseError(f"{self.name} 返回了空的回复内容")
        return content


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        name: str,
        client: Any,
        model: str,
        policy: RetryPolicy,
        sleep: Callable[[float], None] = time.sleep,
        secrets: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self._client = client
        self._model = model
        self._policy = policy
        self._sleep = sleep
        self._secrets = secrets
        self.dimension = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        for index, text in enumerate(items):
            if not text or not text.strip():
                raise ValueError(f"待嵌入文本第 {index} 条为空")
        response = _call_redacted(
            lambda: self._client.embeddings.create(model=self._model, input=items),
            self._policy,
            self._sleep,
            self._secrets,
        )
        vectors = self._extract(response, expected=len(items))
        self.dimension = len(vectors[0])
        return vectors

    def _extract(self, response: Any, *, expected: int) -> list[list[float]]:
        data = getattr(response, "data", None) or []
        if len(data) != expected:
            # 数量对不上时补零会把污染悄悄写进索引，宁可失败得明显
            raise ProviderResponseError(
                f"{self.name} 返回了 {len(data)} 条嵌入，与请求的 {expected} 条不符"
            )
        # 服务端不保证顺序，按 index 归位
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        vectors = [list(getattr(item, "embedding", []) or []) for item in ordered]
        if any(not vector for vector in vectors):
            raise ProviderResponseError(f"{self.name} 返回了空向量")
        if len({len(vector) for vector in vectors}) != 1:
            raise ProviderResponseError(f"{self.name} 返回的向量维度不一致")
        return vectors


def build_chat_provider(
    profile_name: str,
    settings: "Settings",
    *,
    client_factory: ClientFactory | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> OpenAICompatibleChatProvider:
    profile = _resolve_profile(profile_name)
    factory = client_factory or _default_client_factory
    client = factory(_client_config(profile, settings))
    return OpenAICompatibleChatProvider(
        name=profile.name,
        client=client,
        model=settings.chat_model,
        policy=RetryPolicy.from_settings(settings),
        sleep=sleep,
        secrets=collect_secrets(settings),
    )


def build_embedding_provider(
    profile_name: str,
    settings: "Settings",
    *,
    client_factory: ClientFactory | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> OpenAICompatibleEmbeddingProvider:
    profile = _resolve_profile(profile_name)
    factory = client_factory or _default_client_factory
    client = factory(_client_config(profile, settings))
    return OpenAICompatibleEmbeddingProvider(
        name=profile.name,
        client=client,
        model=settings.embedding_model,
        policy=RetryPolicy.from_settings(settings),
        sleep=sleep,
        secrets=collect_secrets(settings),
    )
