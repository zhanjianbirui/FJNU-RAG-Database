"""Chat 与 Embedding 的 provider 契约。

契约刻意保持窄：只描述调用方真正需要的行为，供应方特有的参数（base_url、
超时、组织 ID 等）经配置传给具体实现，不进入签名。这样切换供应方时调用方
代码无需改动（model-providers 规格）。
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ChatProvider",
    "EmbeddingProvider",
    "Message",
    "ProviderAuthError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderUnavailableError",
    "ROLES",
]

ROLES = frozenset({"system", "user", "assistant"})


class ProviderError(RuntimeError):
    """provider 调用失败。调用方据此与自身逻辑错误区分开。"""


class ProviderAuthError(ProviderError):
    """凭据缺失或无效。信息中不得包含凭据本身的任何字符。"""


class ProviderRateLimitError(ProviderError):
    """被限流。可重试。"""


class ProviderUnavailableError(ProviderError):
    """超时或服务端 5xx。可重试。"""


class ProviderResponseError(ProviderError):
    """响应格式不符合预期，无法从中取到结果。不可重试。"""


@dataclass(frozen=True, slots=True)
class Message:
    """一条对话消息。不可变，便于安全地在重试之间复用。"""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(
                f"未知的消息角色 {self.role!r}，可用值：{', '.join(sorted(ROLES))}"
            )


class ChatProvider(ABC):
    """对话生成。"""

    #: 注册表中的 provider 名称，用于日志与界面展示。
    name: str = "unnamed"

    @abstractmethod
    def complete(
        self, messages: Sequence[Message], *, max_tokens: int | None = None
    ) -> str:
        """生成回复文本。

        失败时 MUST 抛出 :class:`ProviderError` 的子类，
        MUST NOT 返回空字符串冒充成功。
        """


class EmbeddingProvider(ABC):
    """文本嵌入。"""

    name: str = "unnamed"

    #: 向量维度。建索引时用于校验，避免维度不一致的向量混入同一集合。
    dimension: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量嵌入。

        返回值与输入等长且顺序一致。失败时 MUST 抛出 :class:`ProviderError`
        的子类，MUST NOT 返回空向量冒充成功。
        """
