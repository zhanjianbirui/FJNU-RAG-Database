"""退避重试与错误分类。

model-providers 规格：对限流、超时、5xx 按配置退避重试；重试耗尽后必须向
调用方返回**可区分**的错误类型，MUST NOT 静默返回空字符串或空向量冒充成功。
"""

import pytest

from kbwb.config.settings import load_settings
from kbwb.providers.base import (
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from kbwb.providers.openai_compatible import build_chat_provider, build_embedding_provider
from kbwb.providers.retry import RetryPolicy, classify_exception, call_with_retry

from .fixtures import CHAT_RESPONSE, EMBEDDING_RESPONSE, as_object

_BASE = {"chat_model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small"}


def _settings(**overrides):
    return load_settings(_env_file=None, **_BASE, openai_api_key="sk-x", **overrides)


class HttpError(Exception):
    """带状态码的替身异常，模拟 SDK 抛出的 APIStatusError 家族。"""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class Timeout(Exception):
    pass


Timeout.__name__ = "APITimeoutError"


class Recorder:
    """记录每次 sleep 的时长，避免测试真的等待。"""

    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


class TestClassification:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, ProviderAuthError),
            (403, ProviderAuthError),
            (429, ProviderRateLimitError),
            (500, ProviderUnavailableError),
            (502, ProviderUnavailableError),
            (503, ProviderUnavailableError),
            (400, ProviderResponseError),
            (404, ProviderResponseError),
        ],
    )
    def test_status_code_maps_to_error_type(self, status, expected):
        assert isinstance(classify_exception(HttpError(status)), expected)

    def test_timeout_is_retryable_unavailable(self):
        assert isinstance(classify_exception(Timeout()), ProviderUnavailableError)

    def test_unknown_exception_becomes_provider_error(self):
        assert isinstance(classify_exception(RuntimeError("怪事")), ProviderError)

    def test_existing_provider_error_passes_through(self):
        original = ProviderAuthError("已分类")
        assert classify_exception(original) is original


class TestRetryLoop:
    def _policy(self, **overrides):
        return RetryPolicy(**{"max_retries": 3, "backoff_seconds": 1.0, **overrides})

    def test_rate_limit_then_success(self):
        attempts = {"n": 0}
        sleep = Recorder()

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise HttpError(429)
            return "成功"

        assert call_with_retry(flaky, self._policy(), sleep=sleep) == "成功"
        assert attempts["n"] == 3
        assert len(sleep.delays) == 2

    def test_exhausted_retries_raise_distinguishable_error(self):
        sleep = Recorder()

        def always_limited():
            raise HttpError(429)

        with pytest.raises(ProviderRateLimitError):
            call_with_retry(always_limited, self._policy(), sleep=sleep)

    def test_exhausted_retries_do_not_return_a_fake_result(self):
        """规格重点：耗尽后必须抛错，而不是返回空结果冒充成功。"""
        def always_down():
            raise HttpError(503)

        with pytest.raises(ProviderUnavailableError):
            call_with_retry(always_down, self._policy(), sleep=Recorder())

    def test_attempt_count_equals_retries_plus_one(self):
        attempts = {"n": 0}

        def always_limited():
            attempts["n"] += 1
            raise HttpError(429)

        with pytest.raises(ProviderRateLimitError):
            call_with_retry(always_limited, self._policy(max_retries=2), sleep=Recorder())
        assert attempts["n"] == 3

    def test_auth_error_is_not_retried(self):
        attempts = {"n": 0}

        def unauthorised():
            attempts["n"] += 1
            raise HttpError(401)

        with pytest.raises(ProviderAuthError):
            call_with_retry(unauthorised, self._policy(), sleep=Recorder())
        assert attempts["n"] == 1

    def test_client_error_is_not_retried(self):
        attempts = {"n": 0}

        def bad_request():
            attempts["n"] += 1
            raise HttpError(400)

        with pytest.raises(ProviderResponseError):
            call_with_retry(bad_request, self._policy(), sleep=Recorder())
        assert attempts["n"] == 1

    def test_backoff_grows(self):
        sleep = Recorder()

        def always_limited():
            raise HttpError(429)

        with pytest.raises(ProviderRateLimitError):
            call_with_retry(always_limited, self._policy(max_retries=3), sleep=sleep)
        assert sleep.delays == sorted(sleep.delays)
        assert sleep.delays[0] < sleep.delays[-1]

    def test_zero_retries_means_single_attempt(self):
        attempts = {"n": 0}
        sleep = Recorder()

        def always_limited():
            attempts["n"] += 1
            raise HttpError(429)

        with pytest.raises(ProviderRateLimitError):
            call_with_retry(always_limited, self._policy(max_retries=0), sleep=sleep)
        assert attempts["n"] == 1
        assert sleep.delays == []

    def test_policy_from_settings(self):
        policy = RetryPolicy.from_settings(
            _settings(provider_max_retries=5, provider_backoff_seconds=2.5)
        )
        assert policy.max_retries == 5
        assert policy.backoff_seconds == 2.5


class FlakyClient:
    """前 ``failures`` 次调用返回限流，之后返回录制响应。"""

    def __init__(self, failures, chat_payload=CHAT_RESPONSE, embedding_payload=EMBEDDING_RESPONSE):
        self.attempts = 0
        outer = self

        def _fail_then(payload):
            def create(**kwargs):
                outer.attempts += 1
                if outer.attempts <= failures:
                    raise HttpError(429)
                return as_object(payload)

            return create

        self.chat = type(
            "_Chat", (), {"completions": type("_C", (), {"create": staticmethod(_fail_then(chat_payload))})()}
        )()
        self.embeddings = type("_E", (), {"create": staticmethod(_fail_then(embedding_payload))})()


class TestProviderIntegration:
    def test_chat_recovers_after_rate_limit(self):
        client = FlakyClient(failures=2)
        provider = build_chat_provider(
            "openai", _settings(), client_factory=lambda _: client, sleep=Recorder()
        )
        answer = provider.complete([Message(role="user", content="学籍证明")])
        assert answer.startswith("办理学籍证明")
        assert client.attempts == 3

    def test_chat_raises_after_exhausting_retries(self):
        client = FlakyClient(failures=99)
        provider = build_chat_provider(
            "openai",
            _settings(provider_max_retries=2),
            client_factory=lambda _: client,
            sleep=Recorder(),
        )
        with pytest.raises(ProviderRateLimitError):
            provider.complete([Message(role="user", content="x")])

    def test_embedding_recovers_after_rate_limit(self):
        client = FlakyClient(failures=1)
        provider = build_embedding_provider(
            "openai", _settings(), client_factory=lambda _: client, sleep=Recorder()
        )
        assert len(provider.embed(["甲", "乙"])) == 2

    def test_embedding_never_returns_empty_vectors_on_failure(self):
        client = FlakyClient(failures=99)
        provider = build_embedding_provider(
            "openai",
            _settings(provider_max_retries=1),
            client_factory=lambda _: client,
            sleep=Recorder(),
        )
        with pytest.raises(ProviderError):
            provider.embed(["甲", "乙"])
