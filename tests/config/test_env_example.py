"""`.env.example` 与配置模型的一致性。

这组用例防的是漂移：新增配置字段却忘了写进示例文件，使用者照示例填完
仍然启动失败。
"""

from pathlib import Path

import pytest

from kbwb.config.settings import Settings, load_settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"

# 凭据在示例中留空，由使用者自行填写。
_CREDENTIAL_KEYS = {"OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_API_KEY"}


def _parse(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        entries[key.strip()] = value.strip()
    return entries


@pytest.fixture
def entries():
    return _parse(ENV_EXAMPLE)


def test_env_example_exists():
    assert ENV_EXAMPLE.is_file()


def test_covers_every_settings_field(entries):
    expected = {name.upper() for name in Settings.model_fields}
    missing = expected - set(entries)
    assert not missing, f".env.example 缺少这些变量：{sorted(missing)}"


def test_declares_no_unknown_variables(entries):
    known = {name.upper() for name in Settings.model_fields}
    unknown = set(entries) - known
    assert not unknown, f".env.example 含配置模型中不存在的变量：{sorted(unknown)}"


def test_credentials_are_left_blank(entries):
    for key in _CREDENTIAL_KEYS:
        assert entries.get(key) == "", f"{key} 不应在示例中带值"


def test_no_plausible_secret_committed(entries):
    for key, value in entries.items():
        assert not value.startswith("sk-"), f"{key} 疑似包含真实凭据"


def test_settings_load_after_filling_the_example(entries, monkeypatch):
    """按示例填写后（凭据补上占位值），1.3 的配置加载应当通过。"""
    for name in _CREDENTIAL_KEYS:
        monkeypatch.delenv(name, raising=False)
    filled = {k: (v or "placeholder") for k, v in entries.items()}
    settings = load_settings(_env_file=None, **{k.lower(): v for k, v in filled.items()})
    assert settings.host == "127.0.0.1"
    assert settings.chat_model
    assert settings.embedding_model


def test_example_values_match_model_defaults(entries):
    """示例中的非凭据值不应与模型默认值冲突，否则示例本身就在改变行为。"""
    defaults = {
        name.upper(): field.default
        for name, field in Settings.model_fields.items()
        if field.default is not None and not field.is_required()
    }
    mismatched = [
        key
        for key, value in entries.items()
        if key in defaults and value and str(defaults[key]).lower() != value.lower()
    ]
    assert not mismatched, f"示例值与模型默认值不一致：{mismatched}"
