"""按名注册的 provider 注册表。

规格要求未知 provider 名称在**启动时**报错并列出所有可用名称，而不是等到
首次调用才失败——因此 :func:`build_providers` 在启动阶段就构造两个 provider。

内置名称以惰性工厂注册：注册表本身不导入具体实现，可用名称的列举因而不受
可选依赖（如本地嵌入需要 sentence-transformers）是否安装的影响。
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from kbwb.providers.base import ChatProvider, EmbeddingProvider, ProviderError
from kbwb.providers.credentials import require_credentials

if TYPE_CHECKING:  # pragma: no cover - 仅为类型标注，避免运行期循环导入
    from kbwb.config.settings import Settings

__all__ = [
    "UnknownProviderError",
    "available_chat_providers",
    "available_embedding_providers",
    "build_providers",
    "create_chat_provider",
    "create_embedding_provider",
    "register_chat_provider",
    "register_embedding_provider",
]

ChatFactory = Callable[["Settings"], ChatProvider]
EmbeddingFactory = Callable[["Settings"], EmbeddingProvider]

_CHAT_PROVIDERS: dict[str, ChatFactory] = {}
_EMBEDDING_PROVIDERS: dict[str, EmbeddingFactory] = {}


class UnknownProviderError(ProviderError):
    """配置指定了未注册的 provider 名称。"""


def register_chat_provider(name: str, factory: ChatFactory) -> None:
    _CHAT_PROVIDERS[name] = factory


def register_embedding_provider(name: str, factory: EmbeddingFactory) -> None:
    _EMBEDDING_PROVIDERS[name] = factory


def available_chat_providers() -> tuple[str, ...]:
    return tuple(sorted(_CHAT_PROVIDERS))


def available_embedding_providers() -> tuple[str, ...]:
    return tuple(sorted(_EMBEDDING_PROVIDERS))


def _require_known(registry: dict, name: str, *, kind: str, env_var: str) -> None:
    """只校验名称是否注册，不构造实例。"""
    if name not in registry:
        raise UnknownProviderError(
            f"{env_var} 指定了未知的{kind} provider {name!r}，"
            f"可用名称：{', '.join(sorted(registry)) or '（无）'}"
        )


def _create(registry: dict, name: str, settings: "Settings", *, kind: str, env_var: str):
    _require_known(registry, name, kind=kind, env_var=env_var)
    return registry[name](settings)


def create_chat_provider(name: str, settings: "Settings") -> ChatProvider:
    return _create(
        _CHAT_PROVIDERS, name, settings, kind="chat", env_var="CHAT_PROVIDER"
    )


def create_embedding_provider(name: str, settings: "Settings") -> EmbeddingProvider:
    return _create(
        _EMBEDDING_PROVIDERS,
        name,
        settings,
        kind="embedding",
        env_var="EMBEDDING_PROVIDER",
    )


def build_providers(settings: "Settings") -> tuple[ChatProvider, EmbeddingProvider]:
    """在启动阶段构造两个 provider，使配置错误立即暴露。

    两个名称都先校验再构造：否则一个未知的 embedding 名称要等 chat 构造成功
    之后才报出，而 chat 的构造可能因缺凭据或缺可选依赖先失败，掩盖真正的
    配置笔误。
    """
    _require_known(
        _CHAT_PROVIDERS, settings.chat_provider, kind="chat", env_var="CHAT_PROVIDER"
    )
    _require_known(
        _EMBEDDING_PROVIDERS,
        settings.embedding_provider,
        kind="embedding",
        env_var="EMBEDDING_PROVIDER",
    )
    require_credentials(settings)
    chat = create_chat_provider(settings.chat_provider, settings)
    embedding = create_embedding_provider(settings.embedding_provider, settings)
    return chat, embedding


# --- 内置实现的惰性注册 -------------------------------------------------------
# OpenAI 与 DeepSeek 共用一套基于 OpenAI SDK 的实现，差别只是 base_url、
# 模型名与凭据变量（见 design.md 的 provider 抽象决策）。


def _openai_chat(settings: "Settings") -> ChatProvider:
    from kbwb.providers.openai_compatible import build_chat_provider

    return build_chat_provider("openai", settings)


def _deepseek_chat(settings: "Settings") -> ChatProvider:
    from kbwb.providers.openai_compatible import build_chat_provider

    return build_chat_provider("deepseek", settings)


def _openai_embedding(settings: "Settings") -> EmbeddingProvider:
    from kbwb.providers.openai_compatible import build_embedding_provider

    return build_embedding_provider("openai", settings)


def _deepseek_embedding(settings: "Settings") -> EmbeddingProvider:
    from kbwb.providers.openai_compatible import build_embedding_provider

    return build_embedding_provider("deepseek", settings)


def _local_embedding(settings: "Settings") -> EmbeddingProvider:
    from kbwb.providers.local_embedding import build_local_embedding_provider

    return build_local_embedding_provider(settings)


register_chat_provider("openai", _openai_chat)
register_chat_provider("deepseek", _deepseek_chat)
register_embedding_provider("openai", _openai_embedding)
register_embedding_provider("deepseek", _deepseek_embedding)
# 本地模型只提供嵌入，不提供对话生成
register_embedding_provider("local", _local_embedding)
