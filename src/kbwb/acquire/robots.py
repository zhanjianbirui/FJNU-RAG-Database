"""robots.txt 解析与判定。

自行实现而非直接用 ``urllib.robotparser``，原因是规格要求向用户**报告命中的
那条规则**——标准库只给布尔值，无法说明"为什么不抓"。

匹配遵循通行约定：先按 User-agent 选组（具名组优先于 ``*``），组内按路径
最长匹配决定，长度相同时 Allow 优先；支持 ``*`` 通配与 ``$`` 结尾锚定。
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from kbwb.acquire.fetching import FetchError, Fetcher, origin_of

__all__ = ["RobotsDecision", "RobotsGate", "RobotsRules"]

WILDCARD_AGENT = "*"


@dataclass(frozen=True, slots=True)
class RobotsDecision:
    """一次判定的结果。被禁时 ``rule`` 给出命中的指令原文。"""

    allowed: bool
    rule: str | None = None
    robots_url: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Directive:
    allow: bool
    path: str
    pattern: re.Pattern

    @property
    def text(self) -> str:
        return f"{'Allow' if self.allow else 'Disallow'}: {self.path}"


@dataclass
class _Group:
    directives: list[_Directive] = field(default_factory=list)
    crawl_delay: float | None = None


def _compile(path: str) -> re.Pattern:
    """把 robots 路径模式编译为正则：``*`` 任意串，``$`` 锚定结尾。"""
    anchored = path.endswith("$")
    body = path[:-1] if anchored else path
    escaped = "".join(".*" if char == "*" else re.escape(char) for char in body)
    return re.compile("^" + escaped + ("$" if anchored else ""))


class RobotsRules:
    """一份 robots.txt 的解析结果。"""

    def __init__(self, groups: dict[str, _Group]) -> None:
        self._groups = groups

    @classmethod
    def parse(cls, text: str) -> "RobotsRules":
        groups: dict[str, _Group] = {}
        current: list[str] = []
        starting_group = True
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "user-agent":
                if not starting_group:
                    current = []
                    starting_group = True
                current.append(value.lower())
                groups.setdefault(value.lower(), _Group())
            elif current:
                starting_group = False
                cls._apply(groups, current, key, value)
        return cls(groups)

    @staticmethod
    def _apply(groups: dict[str, _Group], agents: list[str], key: str, value: str) -> None:
        for agent in agents:
            group = groups[agent]
            if key in ("allow", "disallow"):
                if not value:
                    # 空的 Disallow 表示不禁止任何内容，直接跳过
                    continue
                group.directives.append(
                    _Directive(allow=key == "allow", path=value, pattern=_compile(value))
                )
            elif key == "crawl-delay":
                try:
                    group.crawl_delay = float(value)
                except ValueError:
                    pass  # 非法值忽略，不因一行畸形配置整体失效

    def _select(self, user_agent: str) -> _Group | None:
        """具名组优先；同时匹配多个具名组时取名字最长的那个。"""
        lowered = user_agent.lower()
        named = [
            agent
            for agent in self._groups
            if agent != WILDCARD_AGENT and agent and agent in lowered
        ]
        if named:
            return self._groups[max(named, key=len)]
        return self._groups.get(WILDCARD_AGENT)

    def evaluate(self, path: str, user_agent: str) -> RobotsDecision:
        group = self._select(user_agent)
        if group is None:
            return RobotsDecision(allowed=True)
        matched = [d for d in group.directives if d.pattern.match(path)]
        if not matched:
            return RobotsDecision(allowed=True)
        # 最长匹配优先；长度相同时 Allow 胜出
        best = max(matched, key=lambda d: (len(d.path), d.allow))
        if best.allow:
            return RobotsDecision(allowed=True)
        return RobotsDecision(allowed=False, rule=best.text, reason="robots")

    def crawl_delay(self, user_agent: str) -> float | None:
        group = self._select(user_agent)
        return group.crawl_delay if group else None


class RobotsGate:
    """按来源缓存 robots.txt，并对具体 URL 作出判定。

    取不到 robots.txt 时的取向是**失败即关闭**：4xx 视为没有限制（站点确实
    未提供），5xx 与网络错误视为禁止——无法确认许可时宁可少抓。
    """

    def __init__(self, fetcher: Fetcher, *, user_agent: str) -> None:
        self._fetcher = fetcher
        self._user_agent = user_agent
        self._cache: dict[str, RobotsRules | None] = {}

    def _robots_url(self, url: str) -> str:
        return f"{origin_of(url)}/robots.txt"

    def _rules_for(self, url: str) -> RobotsRules | None:
        """``None`` 表示无法确认许可，调用方应按禁止处理。"""
        origin = origin_of(url)
        if origin not in self._cache:
            self._cache[origin] = self._load(f"{origin}/robots.txt")
        return self._cache[origin]

    def _load(self, robots_url: str) -> RobotsRules | None:
        try:
            response = self._fetcher.get(robots_url)
        except FetchError:
            return None
        if response.status >= 500:
            return None
        if response.status >= 400:
            return RobotsRules.parse("")  # 站点未提供 robots.txt
        return RobotsRules.parse(response.text)

    def check(self, url: str) -> RobotsDecision:
        robots_url = self._robots_url(url)
        rules = self._rules_for(url)
        if rules is None:
            return RobotsDecision(
                allowed=False,
                rule=None,
                robots_url=robots_url,
                reason="无法获取 robots.txt，按禁止处理",
            )
        parts = urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        decision = rules.evaluate(path, self._user_agent)
        return RobotsDecision(
            allowed=decision.allowed,
            rule=decision.rule,
            robots_url=robots_url,
            reason=decision.reason,
        )

    def crawl_delay(self, url: str) -> float | None:
        rules = self._rules_for(url)
        return rules.crawl_delay(self._user_agent) if rules else None
