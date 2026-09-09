"""全局测试隔离。

配置读取的优先级是「进程环境变量 > .env 文件」，而 `_env_file=None` 只关掉
后者。仓库根目录的 `.env` 与 pytest 插件（deepeval 会把它保存的判分模型配置
注入环境）都可能在会话期间往 `os.environ` 里写入同名变量，使配置用例的断言
依赖运行机器的状态。

因此在每个用例前统一清空全部配置变量，让用例只看见自己显式传入的值。
"""

import pytest

from kbwb.config.settings import Settings

_CONFIG_ENV_VARS = tuple(name.upper() for name in Settings.model_fields)


@pytest.fixture(autouse=True)
def isolated_settings_env(monkeypatch):
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # 已被 DATA_ROOT 取代的历史变量，一并清除以免干扰"未知变量被忽略"的用例
    monkeypatch.delenv("VECTOR_STORE_PATH", raising=False)
