"""知识库存储布局。

每个知识库占一个目录，删除即删目录，无需数据库服务（见 design.md）：

    <data_root>/<kb_id>/
    ├── raw/            原件：页面快照、导入文件副本、附件。权威，不改动
    ├── md/             统一 Markdown 中间表示。派生产物，可删除重建
    ├── profile.json    语料画像
    ├── config.yaml     该库的建库配置（含人工覆盖标记）
    └── index/
        ├── current     指向生效版本
        └── <version>/  向量库 + 关键词索引

``kb_id`` 与索引版本标识都来自外部输入（HTTP 接口、命令行），因此在拼接路径
前必须校验，防止借 ``..`` 或分隔符跳出 ``data_root``。
"""

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "INDEX_CURRENT_POINTER",
    "InvalidKnowledgeBaseId",
    "KnowledgeBaseLayout",
    "list_knowledge_bases",
    "resolve_layout",
]

INDEX_CURRENT_POINTER = "current"

MAX_SEGMENT_LENGTH = 64

# 仅允许 ASCII 字母数字与连字符、下划线，且必须以字母或数字开头。
# 这一并排除了 ``.``、``..``、路径分隔符、空白、通配符与前导点的隐藏目录。
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class InvalidKnowledgeBaseId(ValueError):
    """知识库标识或索引版本标识不安全，拒绝用于拼接路径。"""


def _validate_segment(value: object, *, kind: str) -> str:
    """校验单个路径片段，返回原值；不合法则抛出并回显原始输入。"""
    if not isinstance(value, str):
        raise InvalidKnowledgeBaseId(f"{kind}必须是字符串，收到 {type(value).__name__}")
    if len(value) > MAX_SEGMENT_LENGTH:
        raise InvalidKnowledgeBaseId(
            f"{kind}长度不得超过 {MAX_SEGMENT_LENGTH} 个字符：{value!r}"
        )
    if not _SAFE_SEGMENT.match(value):
        raise InvalidKnowledgeBaseId(
            f"{kind}只允许字母、数字、连字符与下划线，且须以字母或数字开头：{value}"
        )
    return value


def _is_safe_segment(value: str) -> bool:
    return len(value) <= MAX_SEGMENT_LENGTH and bool(_SAFE_SEGMENT.match(value))


def _contain(parent: Path, child: Path, *, label: str) -> Path:
    """纵深防御：即便片段通过了字符校验，也确认结果仍落在父目录内。"""
    if not child.resolve().is_relative_to(parent.resolve()):
        raise InvalidKnowledgeBaseId(f"{label} 解析后落在 {parent} 之外")
    return child


@dataclass(frozen=True, slots=True)
class KnowledgeBaseLayout:
    """单个知识库的目录布局。值对象，构造后不可变。"""

    data_root: Path
    kb_id: str

    @property
    def root(self) -> Path:
        return self.data_root / self.kb_id

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def md(self) -> Path:
        return self.root / "md"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def profile_file(self) -> Path:
        return self.root / "profile.json"

    @property
    def config_file(self) -> Path:
        return self.root / "config.yaml"

    @property
    def index_pointer(self) -> Path:
        """指向当前生效索引版本的指针文件。"""
        return self.index / INDEX_CURRENT_POINTER

    def index_version(self, version: str) -> Path:
        """某个索引版本的目录。``version`` 是外部输入，同样需要校验。"""
        _validate_segment(version, kind="索引版本标识")
        return _contain(self.index, self.index / version, label="索引版本标识")

    def ensure(self) -> "KnowledgeBaseLayout":
        """创建目录结构。已存在的目录与其中内容不受影响。"""
        for path in (self.root, self.raw, self.md, self.index):
            path.mkdir(parents=True, exist_ok=True)
        return self


def resolve_layout(data_root: Path | str, kb_id: str) -> KnowledgeBaseLayout:
    """解析知识库布局。不接触文件系统，仅校验并拼接路径。"""
    root = Path(data_root)
    _validate_segment(kb_id, kind="知识库标识")
    layout = KnowledgeBaseLayout(data_root=root, kb_id=kb_id)
    _contain(root, layout.root, label="知识库标识")
    return layout


def list_knowledge_bases(data_root: Path | str) -> list[str]:
    """列出 ``data_root`` 下的知识库标识，按名称升序。

    忽略普通文件与名称不合法的目录——后者可能是手工创建或其他程序留下的，
    不应被当作知识库对待。
    """
    root = Path(data_root)
    if not root.is_dir():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir() and _is_safe_segment(entry.name)
    )
