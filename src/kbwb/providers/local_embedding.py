"""本地 Sentence-Transformers 嵌入。

design.md 的默认取向是走云端 API，本地模型作为可选依赖按需启用，避免让所有
使用者为一项可选能力付出安装体积（中文嵌入权重约 96MB，加上 torch 更大）。

因此这里做两件事：把可选依赖缺失转成一条能直接照做的错误信息，以及在权重
就位后完成不触网的嵌入。
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from kbwb.providers.base import EmbeddingProvider, ProviderError, ProviderResponseError

if TYPE_CHECKING:  # pragma: no cover
    from kbwb.config.settings import Settings

__all__ = [
    "LocalEmbeddingProvider",
    "MissingLocalModelDependency",
    "build_local_embedding_provider",
]

PROVIDER_NAME = "local"
EXTRA_GROUP = "local-models"


class MissingLocalModelDependency(ProviderError):
    """选择了本地嵌入但未安装对应的可选依赖组。"""


def _load_sentence_transformer(model_name: str) -> Any:
    """导入并加载模型。独立成函数，便于测试替换。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _as_vectors(raw: Any) -> list[list[float]]:
    """兼容 numpy 数组与纯 Python 列表两种返回形态。"""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [[float(value) for value in vector] for vector in raw]


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, model: Any, model_name: str) -> None:
        self.name = PROVIDER_NAME
        self._model = model
        self._model_name = model_name
        self.dimension = int(model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        for index, text in enumerate(items):
            if not text or not text.strip():
                raise ValueError(f"待嵌入文本第 {index} 条为空")
        try:
            raw = self._model.encode(items, convert_to_numpy=False)
        except Exception as exc:  # 模型内部失败一律转为 provider 错误
            raise ProviderError(f"本地模型 {self._model_name} 嵌入失败：{exc}") from exc
        vectors = _as_vectors(raw)
        self._check(vectors, expected=len(items))
        return vectors

    def _check(self, vectors: list[list[float]], *, expected: int) -> None:
        if len(vectors) != expected:
            raise ProviderResponseError(
                f"本地模型 {self._model_name} 返回了 {len(vectors)} 条嵌入，"
                f"与请求的 {expected} 条不符"
            )
        if any(not vector for vector in vectors):
            raise ProviderResponseError(f"本地模型 {self._model_name} 返回了空向量")


def build_local_embedding_provider(
    settings: "Settings", *, model_factory: Callable[[str], Any] | None = None
) -> LocalEmbeddingProvider:
    model_name = settings.local_embedding_model
    factory = model_factory or _load_sentence_transformer
    try:
        model = factory(model_name)
    except ImportError as exc:
        raise MissingLocalModelDependency(
            "EMBEDDING_PROVIDER=local 需要 sentence-transformers，"
            f"请先安装可选依赖组：uv sync --extra {EXTRA_GROUP}"
        ) from exc
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"加载本地嵌入模型 {model_name} 失败：{exc}") from exc
    return LocalEmbeddingProvider(model=model, model_name=model_name)
