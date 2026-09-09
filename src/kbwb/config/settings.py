"""环境配置模型。

设计要点：

- **报错用环境变量名**。pydantic 的校验错误以内部字段名定位，对照 `.env` 排查
  不便，故在边界处统一转换为大写的环境变量名后再抛出。
- **不可变**。配置在启动时一次读入，之后全程只读，避免运行中被局部修改导致
  不同模块看到不同配置。
- **默认值体现设计决策**：默认绑回环、默认混合检索、默认不启用精排。
"""

from enum import StrEnum
from pathlib import Path

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["ConfigError", "RetrievalMode", "Settings", "load_settings"]

DEFAULT_ENV_FILE = ".env"


class ConfigError(RuntimeError):
    """配置缺失或非法。错误信息以环境变量名定位问题。"""


class RetrievalMode(StrEnum):
    """检索方式。图谱方式未在当前版本启用，故不在此列举。"""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """工作台的全部环境配置。字段名对应同名大写的环境变量。"""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # 容忍历史遗留变量，避免升级后启动失败
        frozen=True,
    )

    # --- provider 选型与凭据 -------------------------------------------------
    chat_provider: str = "openai"
    embedding_provider: str = "openai"
    chat_model: str
    embedding_model: str
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    local_embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # --- 存储 ----------------------------------------------------------------
    data_root: Path = Path("data")

    # --- 本地服务监听 ---------------------------------------------------------
    # 默认只绑回环；对外监听须显式配置，且启动时告警（见 workbench-ui 规格）
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    # --- 切分默认值 -----------------------------------------------------------
    chunk_max_chars: int = Field(default=1000, ge=1)
    chunk_overlap_chars: int = Field(default=120, ge=0)
    chunk_min_chars: int = Field(default=80, ge=0)

    # --- 检索默认值 -----------------------------------------------------------
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_candidate_k: int = Field(default=20, ge=1)
    retrieval_score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-base"

    # --- 画像与推荐阈值 -------------------------------------------------------
    scanned_ratio_threshold: float = Field(default=0.2, gt=0.0, lt=1.0)
    duplicate_template_ratio_threshold: float = Field(default=0.3, gt=0.0, lt=1.0)
    doc_id_density_threshold: float = Field(default=0.1, gt=0.0, lt=1.0)
    min_body_chars: int = Field(default=50, ge=1)

    # --- 运行时限制 -----------------------------------------------------------
    max_input_chars: int = Field(default=4000, ge=1)
    rate_limit_per_minute: int = Field(default=30, ge=1)

    # --- provider 失败处理 ----------------------------------------------------
    provider_max_retries: int = Field(default=3, ge=0)
    provider_backoff_seconds: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        """跨字段约束。信息中直接写出环境变量名，便于对照 `.env` 修改。"""
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError(
                "CHUNK_OVERLAP_CHARS 必须小于 CHUNK_MAX_CHARS，"
                f"当前分别为 {self.chunk_overlap_chars} 与 {self.chunk_max_chars}"
            )
        if self.chunk_min_chars > self.chunk_max_chars:
            raise ValueError(
                "CHUNK_MIN_CHARS 不得大于 CHUNK_MAX_CHARS，"
                f"当前分别为 {self.chunk_min_chars} 与 {self.chunk_max_chars}"
            )
        if self.retrieval_top_k > self.retrieval_candidate_k:
            raise ValueError(
                "RETRIEVAL_TOP_K 不得大于 RETRIEVAL_CANDIDATE_K"
                "（最终结果从候选集中截断），"
                f"当前分别为 {self.retrieval_top_k} 与 {self.retrieval_candidate_k}"
            )
        return self


def _env_name(error: dict) -> str | None:
    """把 pydantic 的错误定位转换为环境变量名；模型级错误无对应变量。"""
    location = error.get("loc") or ()
    return str(location[0]).upper() if location else None


def _describe(error: dict) -> str:
    name = _env_name(error)
    message = error.get("msg", "配置非法")
    return f"{name}: {message}" if name else message


def _format(exc: ValidationError) -> str:
    lines = sorted(_describe(error) for error in exc.errors())
    return "配置校验失败：\n" + "\n".join(f"  - {line}" for line in lines)


def load_settings(**overrides) -> Settings:
    """加载配置。

    校验失败时抛出 :class:`ConfigError`，信息以环境变量名指明每一处问题，
    而非暴露内部字段名。测试可传 ``_env_file=None`` 以跳过 `.env` 读取。
    """
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        raise ConfigError(_format(exc)) from exc
