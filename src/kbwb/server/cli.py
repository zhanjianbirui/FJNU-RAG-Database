"""命令行入口。"""

import typer
import uvicorn

from kbwb.config.settings import load_settings
from kbwb.server.app import create_app
from kbwb.server.runtime import ensure_port_available, resolve_binding, startup_notices

app = typer.Typer(help="kbwb 知识库工作台")


@app.callback()
def main() -> None:
    """保留子命令形式：单命令时 typer 会把它折叠成顶层选项，
    后续还要加 crawl / build 等命令，形式先固定下来。"""


@app.command()
def serve(
    host: str | None = typer.Option(None, help="监听地址。默认只绑回环。"),
    port: int | None = typer.Option(None, help="监听端口。"),
) -> None:
    """启动本地工作台。"""
    settings = load_settings()
    binding = resolve_binding(settings)
    if host or port:
        binding = type(binding)(host=host or binding.host, port=port or binding.port)
    ensure_port_available(binding.host, binding.port)
    for line in startup_notices(binding):
        typer.echo(line)
    uvicorn.run(create_app(), host=binding.host, port=binding.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    app()
