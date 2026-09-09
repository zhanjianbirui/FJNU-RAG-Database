"""工作台的 HTTP 接口与静态页面托管。

当前是骨架：只接入已完成的站点分析能力，其余端点随对应任务组补齐。
前端为单文件静态页，尚未引入构建链——框架选型按 design.md 留待阶段 5。
"""

from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from kbwb.server.analysis import DEFAULT_MAX_SAMPLES, SiteAnalysisService

__all__ = ["create_app"]

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

MAX_SAMPLES_CEILING = 30


def _http_url(value: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("入口地址必须是 http/https 链接")
    return value


def _contact(value: str) -> str:
    from urllib.parse import urlsplit

    if urlsplit(value).scheme not in ("http", "https", "mailto"):
        raise ValueError("联系方式必须是 http(s) 地址或 mailto: 邮箱")
    return value


class SiteAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_url: Annotated[str, AfterValidator(_http_url)]
    # User-Agent 需带可联系的地址，这是合规要求而非可选项
    contact: Annotated[str, AfterValidator(_contact)]
    max_samples: int = Field(default=DEFAULT_MAX_SAMPLES, ge=1, le=MAX_SAMPLES_CEILING)


def create_app(*, analysis_service: Any = None) -> FastAPI:
    service = analysis_service or SiteAnalysisService()
    app = FastAPI(title="kbwb 知识库工作台", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(INDEX_FILE, media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/api/site-analysis")
    def site_analysis(request: Annotated[SiteAnalysisRequest, Body()]) -> dict:
        try:
            return service.analyze(
                entry_url=request.entry_url,
                contact=request.contact,
                max_samples=request.max_samples,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            # 目标站点侧的问题不是本服务的内部错误，用 502 区分开
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
