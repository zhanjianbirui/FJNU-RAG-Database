"""本地服务的启动约束。

两条规格要求落在这里：默认只绑回环地址，以及端口被占用时明确报错而非静默
失败。对外监听时额外提示服务不含身份认证——该提示是可验收要求，不是客套。
"""

import socket
from dataclasses import dataclass
from ipaddress import ip_address
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from kbwb.config.settings import Settings

__all__ = [
    "NO_AUTH_WARNING",
    "PortInUseError",
    "ServerBinding",
    "ensure_port_available",
    "resolve_binding",
    "startup_notices",
]

NO_AUTH_WARNING = "本服务不含身份认证"

_LOOPBACK_ALIASES = frozenset({"localhost", ""})

#: 展示用地址。绑 :: 或 0.0.0.0 时给用户一个能直接点开的本机地址。
_DISPLAY_FALLBACK = "127.0.0.1"


class PortInUseError(RuntimeError):
    """监听端口已被占用。"""


@dataclass(frozen=True, slots=True)
class ServerBinding:
    host: str
    port: int

    @property
    def is_loopback(self) -> bool:
        if self.host.lower() in _LOOPBACK_ALIASES:
            return True
        try:
            return ip_address(self.host).is_loopback
        except ValueError:
            return False

    @property
    def display_host(self) -> str:
        return self.host if self.is_loopback and ":" not in self.host else _DISPLAY_FALLBACK

    @property
    def url(self) -> str:
        return f"http://{self.display_host}:{self.port}"


def resolve_binding(settings: "Settings") -> ServerBinding:
    return ServerBinding(host=settings.host, port=settings.port)


def startup_notices(binding: ServerBinding) -> tuple[str, ...]:
    """启动时应当打印的信息。对外监听的告警在此产生。"""
    notices = [f"工作台已启动：{binding.url}"]
    if not binding.is_loopback:
        notices.append(
            f"警告：正在监听 {binding.host}，同网段的其他机器可以访问。"
            f"{NO_AUTH_WARNING}，请自行在前置代理上做鉴权。"
        )
    return tuple(notices)


def ensure_port_available(host: str, port: int) -> None:
    """端口被占用时报错并说明如何改端口，而不是让启动静默失败。"""
    probe = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    try:
        probe.bind((host, port))
    except OSError as exc:
        raise PortInUseError(
            f"端口 {port} 已被占用（{exc.strerror or exc}）。"
            f"请在 .env 中设置 PORT 为其他值，或先停止占用该端口的程序。"
        ) from exc
    finally:
        probe.close()
