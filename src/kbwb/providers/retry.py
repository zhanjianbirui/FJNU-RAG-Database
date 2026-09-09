"""错误分类与退避重试。

规格把可重试（限流、超时、5xx）与不可重试（凭据无效、请求本身有误）分开，
并要求重试耗尽后返回**可区分**的错误类型——调用方据此决定是提示用户改配置
还是稍后再试。任何情况下都不得返回空结果冒充成功。

分类按鸭子类型进行（看 ``status_code`` 与异常类名），因此不与某个 SDK 的
异常类层次耦合，替换底层库时无需改动。
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from kbwb.providers.base import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:  # pragma: no cover
    from kbwb.config.settings import Settings

__all__ = ["RetryPolicy", "call_with_retry", "classify_exception"]

T = TypeVar("T")

#: 类名中出现这些片段的异常按可重试的网络问题处理。
_RETRYABLE_NAME_HINTS = ("timeout", "connection", "unavailable")

_AUTH_STATUSES = frozenset({401, 403})
_RATE_LIMIT_STATUSES = frozenset({429})

#: 重试时的退避倍率。
_BACKOFF_FACTOR = 2.0


def classify_exception(exc: BaseException) -> ProviderError:
    """把底层异常归类为 :class:`ProviderError` 的某个子类。"""
    if isinstance(exc, ProviderError):
        return exc

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return _classify_status(status, exc)

    name = type(exc).__name__.lower()
    if any(hint in name for hint in _RETRYABLE_NAME_HINTS):
        return ProviderUnavailableError(f"{type(exc).__name__}: {exc}")
    return ProviderError(f"{type(exc).__name__}: {exc}")


def _classify_status(status: int, exc: BaseException) -> ProviderError:
    if status in _AUTH_STATUSES:
        # 信息中只带状态码，不回显请求内容，避免凭据经由异常泄漏
        return ProviderAuthError(f"认证失败（HTTP {status}）")
    if status in _RATE_LIMIT_STATUSES:
        return ProviderRateLimitError(f"被限流（HTTP {status}）")
    if status >= 500:
        return ProviderUnavailableError(f"服务端错误（HTTP {status}）")
    return ProviderResponseError(f"请求被拒绝（HTTP {status}）：{exc}")


def _is_retryable(error: ProviderError) -> bool:
    return isinstance(error, (ProviderRateLimitError, ProviderUnavailableError))


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_seconds: float = 1.0

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RetryPolicy":
        return cls(
            max_retries=settings.provider_max_retries,
            backoff_seconds=settings.provider_backoff_seconds,
        )

    def delay_for(self, attempt: int) -> float:
        """第 ``attempt`` 次重试前的等待时长（指数退避）。"""
        return self.backoff_seconds * (_BACKOFF_FACTOR**attempt)


def call_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """执行 ``operation``，对可重试错误按策略退避重试。

    重试耗尽后抛出最后一次的分类错误，绝不返回替代结果。
    """
    last_error: ProviderError | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return operation()
        except BaseException as exc:  # noqa: BLE001 - 统一分类后再决定是否重试
            error = classify_exception(exc)
            if not _is_retryable(error) or attempt == policy.max_retries:
                raise error from exc
            last_error = error
            sleep(policy.delay_for(attempt))
    raise last_error or ProviderError("重试循环异常退出")  # pragma: no cover
