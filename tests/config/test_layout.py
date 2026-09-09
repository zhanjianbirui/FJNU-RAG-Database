"""知识库存储布局：目录创建与路径穿越拒绝。

布局对应 design.md：

    <data_root>/<kb_id>/
    ├── raw/            原件，权威，不改动
    ├── md/             Markdown 中间表示，派生产物
    ├── profile.json
    ├── config.yaml
    └── index/
        ├── current     指向生效版本
        └── <version>/
"""

import pytest

from kbwb.config.layout import (
    INDEX_CURRENT_POINTER,
    InvalidKnowledgeBaseId,
    KnowledgeBaseLayout,
    list_knowledge_bases,
    resolve_layout,
)


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


class TestPathResolution:
    def test_directories_follow_the_documented_layout(self, data_root):
        layout = resolve_layout(data_root, "school")
        assert layout.root == data_root / "school"
        assert layout.raw == layout.root / "raw"
        assert layout.md == layout.root / "md"
        assert layout.index == layout.root / "index"
        assert layout.profile_file == layout.root / "profile.json"
        assert layout.config_file == layout.root / "config.yaml"

    def test_index_version_paths(self, data_root):
        layout = resolve_layout(data_root, "school")
        assert layout.index_pointer == layout.index / INDEX_CURRENT_POINTER
        assert layout.index_version("v3") == layout.index / "v3"

    def test_index_version_rejects_traversal(self, data_root):
        # 版本标识同样是外部输入，不能借它跳出 index/
        layout = resolve_layout(data_root, "school")
        with pytest.raises(InvalidKnowledgeBaseId):
            layout.index_version("../../etc")

    def test_resolving_does_not_touch_the_filesystem(self, data_root):
        resolve_layout(data_root, "school")
        assert not data_root.exists()


class TestDirectoryCreation:
    def test_ensure_creates_all_directories(self, data_root):
        layout = resolve_layout(data_root, "school").ensure()
        for path in (layout.root, layout.raw, layout.md, layout.index):
            assert path.is_dir()

    def test_ensure_is_idempotent(self, data_root):
        layout = resolve_layout(data_root, "school")
        layout.ensure()
        (layout.raw / "keep.html").write_text("x", encoding="utf-8")
        layout.ensure()
        assert (layout.raw / "keep.html").read_text(encoding="utf-8") == "x"

    def test_ensure_returns_same_layout_value(self, data_root):
        layout = resolve_layout(data_root, "school")
        assert layout.ensure() == layout


class TestPathTraversalRejection:
    @pytest.mark.parametrize(
        "kb_id",
        [
            "..",
            ".",
            "../escape",
            "../../etc/passwd",
            "a/b",
            "a\\b",
            "/absolute",
            "",
            "   ",
            ".hidden",
            "with\0null",
            "with\nnewline",
            "trailing ",
            "空格 名",
            "kb*glob",
            "~root",
        ],
    )
    def test_rejects_unsafe_identifier(self, data_root, kb_id):
        with pytest.raises(InvalidKnowledgeBaseId):
            resolve_layout(data_root, kb_id)

    def test_error_names_the_offending_identifier(self, data_root):
        with pytest.raises(InvalidKnowledgeBaseId) as exc:
            resolve_layout(data_root, "../escape")
        assert "../escape" in str(exc.value)

    @pytest.mark.parametrize("kb_id", ["school", "kb_1", "KB-2", "a", "fjnu-jwc-2026"])
    def test_accepts_safe_identifier(self, data_root, kb_id):
        assert resolve_layout(data_root, kb_id).kb_id == kb_id

    def test_resolved_root_stays_within_data_root(self, data_root):
        layout = resolve_layout(data_root, "school").ensure()
        assert layout.root.resolve().is_relative_to(data_root.resolve())

    def test_overlong_identifier_rejected(self, data_root):
        with pytest.raises(InvalidKnowledgeBaseId):
            resolve_layout(data_root, "k" * 200)


class TestListing:
    def test_lists_only_existing_knowledge_bases(self, data_root):
        resolve_layout(data_root, "beta").ensure()
        resolve_layout(data_root, "alpha").ensure()
        (data_root / "stray-file.txt").write_text("x", encoding="utf-8")
        assert list_knowledge_bases(data_root) == ["alpha", "beta"]

    def test_missing_data_root_lists_nothing(self, data_root):
        assert list_knowledge_bases(data_root) == []

    def test_ignores_directories_with_unsafe_names(self, data_root):
        data_root.mkdir(parents=True)
        (data_root / ".hidden").mkdir()
        assert list_knowledge_bases(data_root) == []


class TestImmutability:
    def test_layout_is_frozen(self, data_root):
        layout = resolve_layout(data_root, "school")
        with pytest.raises(Exception):
            layout.kb_id = "other"

    def test_layout_is_a_value_object(self, data_root):
        assert resolve_layout(data_root, "school") == resolve_layout(data_root, "school")
        assert isinstance(resolve_layout(data_root, "school"), KnowledgeBaseLayout)
