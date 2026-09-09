"""本地服务的启动约束。

workbench-ui 规格：
- 服务 MUST 默认只监听回环地址，MUST NOT 在未显式配置时对外暴露；
- 端口被占用时报告冲突并提示如何指定其他端口，而非静默失败。

qa-service 规格另要求对外监听时提示服务不含身份认证。
"""

import socket

import pytest

from kbwb.config.settings import load_settings
from kbwb.server.runtime import (
    NO_AUTH_WARNING,
    PortInUseError,
    ensure_port_available,
    resolve_binding,
    startup_notices,
)

_BASE = {"chat_model": "m", "embedding_model": "e"}


def _settings(**overrides):
    return load_settings(_env_file=None, **_BASE, **overrides)


class TestBinding:
    def test_defaults_to_loopback(self):
        binding = resolve_binding(_settings())
        assert binding.host == "127.0.0.1"
        assert binding.is_loopback is True

    def test_ipv6_loopback_recognised(self):
        assert resolve_binding(_settings(host="::1")).is_loopback is True

    def test_localhost_recognised_as_loopback(self):
        assert resolve_binding(_settings(host="localhost")).is_loopback is True

    def test_wildcard_is_not_loopback(self):
        assert resolve_binding(_settings(host="0.0.0.0")).is_loopback is False

    def test_explicit_lan_address_is_not_loopback(self):
        assert resolve_binding(_settings(host="192.168.1.10")).is_loopback is False

    def test_url_uses_localhost_for_display(self):
        assert resolve_binding(_settings(port=9000)).url == "http://127.0.0.1:9000"

    def test_binding_is_immutable(self):
        binding = resolve_binding(_settings())
        with pytest.raises(Exception):
            binding.port = 1


class TestStartupNotices:
    def test_loopback_start_has_no_auth_warning(self):
        notices = startup_notices(resolve_binding(_settings()))
        assert not any(NO_AUTH_WARNING in n for n in notices)

    def test_external_binding_warns_about_missing_auth(self):
        notices = startup_notices(resolve_binding(_settings(host="0.0.0.0")))
        assert any(NO_AUTH_WARNING in n for n in notices)

    def test_notices_always_include_the_url(self):
        binding = resolve_binding(_settings(port=8123))
        assert any(binding.url in n for n in startup_notices(binding))


class TestPortConflict:
    def test_free_port_passes(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]
        ensure_port_available("127.0.0.1", free_port)

    def test_occupied_port_raises(self):
        with socket.socket() as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", 0))
            occupied.listen(1)
            port = occupied.getsockname()[1]
            with pytest.raises(PortInUseError) as exc:
                ensure_port_available("127.0.0.1", port)
        message = str(exc.value)
        assert str(port) in message
        assert "PORT" in message  # 提示如何指定其他端口
