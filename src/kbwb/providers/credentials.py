"""凭据校验与脱敏。

两项规格要求在此实现：

- **启动时校验**所选 provider 的凭据是否齐备，缺失时指出环境变量名。等到
  首次调用才失败，意味着用户可能已经跑完一轮爬取才发现建不了库。
- **凭据不得出现在任何日志、错误信息或 API 响应中**。上游 SDK 的异常文本
  常回显请求头，因此在错误向上传递之前统一做一次脱敏。
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kbwb.providers.base import ProviderAuthError

if TYPE_CHECKING:  # pragma: no cover
    from kbwb.config.settings import Settings

__all__ = [
    "CREDENTIAL_REQUIREMENTS",
    "MissingCredentialError",
    "collect_secrets",
    "redact",
    "require_credentials",
]

MASK = "***"


@dataclass(frozen=True, slots=True)
class CredentialRequirement:
    settings_field: str
    env_var: str


#: provider 名称 → 其所需凭据。不在表中的 provider（如本地模型）无需凭据。
CREDENTIAL_REQUIREMENTS: dict[str, CredentialRequirement] = {
    "openai": CredentialRequirement("openai_api_key", "OPENAI_API_KEY"),
    "deepseek": CredentialRequirement("deepseek_api_key", "DEEPSEEK_API_KEY"),
}


class MissingCredentialError(ProviderAuthError):
    """所选 provider 缺少必需凭据。信息只含变量名，不含任何取值。"""


def _is_blank(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def _missing_for(provider_name: str, settings: "Settings") -> str | None:
    requirement = CREDENTIAL_REQUIREMENTS.get(provider_name)
    if requirement is None:
        return None
    value = getattr(settings, requirement.settings_field, None)
    return requirement.env_var if _is_blank(value) else None


def require_credentials(settings: "Settings") -> None:
    """校验 chat 与 embedding 两侧所选 provider 的凭据。一次列全缺失项。"""
    missing = {
        env_var
        for provider_name in (settings.chat_provider, settings.embedding_provider)
        if (env_var := _missing_for(provider_name, settings)) is not None
    }
    if missing:
        names = "、".join(sorted(missing))
        raise MissingCredentialError(
            f"缺少所选 provider 需要的凭据：{names}。请在 .env 中设置后重试。"
        )


def collect_secrets(settings: "Settings") -> tuple[str, ...]:
    """收集配置中所有属于凭据的取值，供脱敏使用。"""
    values = (
        getattr(settings, requirement.settings_field, None)
        for requirement in CREDENTIAL_REQUIREMENTS.values()
    )
    return tuple(value for value in values if isinstance(value, str) and value.strip())


def redact(text: str, secrets: Iterable[str | None]) -> str:
    """把文本中出现的凭据替换为掩码。

    对短取值同样处理——宁可多掩一些，也不让凭据经由错误信息外泄。
    """
    result = text
    for secret in secrets:
        if isinstance(secret, str) and secret.strip():
            result = result.replace(secret, MASK)
    return result
