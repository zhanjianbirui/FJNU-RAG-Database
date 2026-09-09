"""凭据校验与脱敏。

model-providers 规格两条：

- 启动时校验所选 provider 所需凭据是否齐备，缺失时指出**环境变量名**；
- 凭据 MUST NOT 出现在任何日志、错误信息或 API 响应中。
"""

import pytest

from kbwb.config.settings import load_settings
from kbwb.providers.base import Message, ProviderAuthError
from kbwb.providers.credentials import (
    MissingCredentialError,
    redact,
    require_credentials,
)
from kbwb.providers.openai_compatible import build_chat_provider
from kbwb.providers.registry import build_providers

SECRET = "sk-proj-8Ab3ZqW9kLmN0pQrStUvWxYz1234567890"

_BASE = {"chat_model": "m-chat", "embedding_model": "m-embed"}


def _settings(**overrides):
    return load_settings(_env_file=None, **_BASE, **overrides)


def _leaks(text: str, secret: str, *, window: int = 8) -> bool:
    """凭据本身或其任一足够长的片段出现在文本中即视为泄漏。"""
    if secret in text:
        return True
    return any(
        secret[i : i + window] in text for i in range(len(secret) - window + 1)
    )


class TestCredentialValidation:
    def test_missing_openai_key_names_the_env_var(self):
        with pytest.raises(MissingCredentialError) as exc:
            require_credentials(_settings(chat_provider="openai", embedding_provider="local"))
        assert "OPENAI_API_KEY" in str(exc.value)

    def test_missing_deepseek_key_names_the_env_var(self):
        with pytest.raises(MissingCredentialError) as exc:
            require_credentials(
                _settings(chat_provider="deepseek", embedding_provider="local")
            )
        assert "DEEPSEEK_API_KEY" in str(exc.value)

    def test_reports_both_missing_credentials(self):
        with pytest.raises(MissingCredentialError) as exc:
            require_credentials(
                _settings(chat_provider="openai", embedding_provider="deepseek")
            )
        message = str(exc.value)
        assert "OPENAI_API_KEY" in message and "DEEPSEEK_API_KEY" in message

    def test_local_embedding_needs_no_credential(self):
        require_credentials(
            _settings(chat_provider="openai", embedding_provider="local", openai_api_key=SECRET)
        )

    def test_blank_credential_counts_as_missing(self):
        with pytest.raises(MissingCredentialError):
            require_credentials(
                _settings(chat_provider="openai", embedding_provider="local", openai_api_key="   ")
            )

    def test_complete_credentials_pass(self):
        require_credentials(
            _settings(
                chat_provider="deepseek",
                embedding_provider="openai",
                deepseek_api_key=SECRET,
                openai_api_key=SECRET,
            )
        )

    def test_error_is_an_auth_error(self):
        assert issubclass(MissingCredentialError, ProviderAuthError)

    def test_missing_credential_message_carries_no_value(self):
        with pytest.raises(MissingCredentialError) as exc:
            require_credentials(_settings(chat_provider="openai", embedding_provider="local"))
        assert not _leaks(str(exc.value), SECRET)


class TestRedaction:
    def test_masks_the_secret(self):
        assert not _leaks(redact(f"Authorization: Bearer {SECRET}", [SECRET]), SECRET)

    def test_keeps_surrounding_context(self):
        assert "Authorization" in redact(f"Authorization: Bearer {SECRET}", [SECRET])

    def test_masks_every_occurrence(self):
        text = f"{SECRET} 与 {SECRET}"
        assert not _leaks(redact(text, [SECRET]), SECRET)

    def test_ignores_empty_secrets(self):
        assert redact("原样保留", ["", None]) == "原样保留"

    def test_no_secret_means_unchanged(self):
        assert redact("普通信息", []) == "普通信息"


class TestProviderErrorRedaction:
    def _provider(self, exc):
        class Failing:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs):
                        raise exc

        return build_chat_provider(
            "openai",
            _settings(openai_api_key=SECRET, provider_max_retries=0),
            client_factory=lambda _: Failing(),
            sleep=lambda _: None,
        )

    def test_auth_failure_message_excludes_the_credential(self):
        """SDK 的异常文本常回显请求头，凭据不得随之外泄。"""

        class AuthFailed(Exception):
            status_code = 401

        provider = self._provider(
            AuthFailed(f"401 Unauthorized for Authorization: Bearer {SECRET}")
        )
        with pytest.raises(ProviderAuthError) as exc:
            provider.complete([Message(role="user", content="x")])
        assert not _leaks(str(exc.value), SECRET)

    def test_generic_failure_message_excludes_the_credential(self):
        provider = self._provider(RuntimeError(f"boom key={SECRET}"))
        with pytest.raises(Exception) as exc:
            provider.complete([Message(role="user", content="x")])
        assert not _leaks(str(exc.value), SECRET)

    def test_redaction_preserves_the_error_type(self):
        class AuthFailed(Exception):
            status_code = 401

        provider = self._provider(AuthFailed(f"bad {SECRET}"))
        with pytest.raises(ProviderAuthError):
            provider.complete([Message(role="user", content="x")])


class TestStartupIntegration:
    def test_build_providers_validates_credentials(self):
        with pytest.raises(MissingCredentialError):
            build_providers(_settings(chat_provider="openai", embedding_provider="local"))

    def test_build_providers_passes_with_credentials(self, monkeypatch):
        import kbwb.providers.local_embedding as local

        monkeypatch.setattr(
            local,
            "_load_sentence_transformer",
            lambda name: type(
                "M", (), {"encode": lambda self, t, **k: [[0.1]] * len(t),
                          "get_sentence_embedding_dimension": lambda self: 1}
            )(),
        )
        chat, embedding = build_providers(
            _settings(chat_provider="openai", embedding_provider="local", openai_api_key=SECRET)
        )
        assert chat.name == "openai" and embedding.name == "local"
