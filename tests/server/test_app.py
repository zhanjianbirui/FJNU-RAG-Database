"""工作台 HTTP 接口。

骨架阶段只接入已完成的站点分析能力：输入一个栏目入口 URL，返回采样结果、
候选选择器与配置草稿。爬取、画像、建库等端点随对应任务组接入。
"""

import pytest
from fastapi.testclient import TestClient

from kbwb.server.app import create_app


class FakeAnalysis:
    """替身分析服务，避免测试触网。"""

    def __init__(self, payload=None, error=None):
        self.payload = payload or {
            "entry_url": "https://x.edu.cn/list.htm",
            "sampled": [{"url": "https://x.edu.cn/1.htm", "bytes": 100}],
            "skipped": [],
            "failures": [],
            "fields": [
                {"field": "body", "needs_manual": False, "candidates": [
                    {"selector": "div.article", "hit_rate": 1.0, "hits": 1, "samples": 1}
                ]}
            ],
            "draft_yaml": "name: 'x'\n",
        }
        self.error = error
        self.calls = []

    def analyze(self, *, entry_url, contact, max_samples):
        self.calls.append((entry_url, contact, max_samples))
        if self.error:
            raise self.error
        return self.payload


@pytest.fixture
def service():
    return FakeAnalysis()


@pytest.fixture
def client(service):
    return TestClient(create_app(analysis_service=service))


class TestStaticUi:
    def test_root_serves_the_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_page_mentions_the_product(self, client):
        assert "知识库" in client.get("/").text

    def test_health_endpoint(self, client):
        assert client.get("/api/health").json()["status"] == "ok"


class TestSiteAnalysisEndpoint:
    def test_returns_the_analysis(self, client):
        response = client.post(
            "/api/site-analysis",
            json={"entry_url": "https://x.edu.cn/list.htm", "contact": "mailto:a@b.cn"},
        )
        assert response.status_code == 200
        assert response.json()["fields"][0]["field"] == "body"

    def test_passes_arguments_through(self, client, service):
        client.post(
            "/api/site-analysis",
            json={
                "entry_url": "https://x.edu.cn/list.htm",
                "contact": "mailto:a@b.cn",
                "max_samples": 3,
            },
        )
        assert service.calls == [("https://x.edu.cn/list.htm", "mailto:a@b.cn", 3)]

    def test_rejects_non_http_url_before_calling_the_service(self, client, service):
        response = client.post(
            "/api/site-analysis",
            json={"entry_url": "file:///etc/passwd", "contact": "mailto:a@b.cn"},
        )
        assert response.status_code == 422
        assert service.calls == []

    def test_rejects_missing_contact(self, client, service):
        response = client.post(
            "/api/site-analysis", json={"entry_url": "https://x.edu.cn/list.htm"}
        )
        assert response.status_code == 422
        assert service.calls == []

    def test_rejects_absurd_sample_count(self, client, service):
        response = client.post(
            "/api/site-analysis",
            json={
                "entry_url": "https://x.edu.cn/list.htm",
                "contact": "mailto:a@b.cn",
                "max_samples": 5000,
            },
        )
        assert response.status_code == 422
        assert service.calls == []

    def test_service_failure_becomes_a_readable_error(self):
        service = FakeAnalysis(error=RuntimeError("目标站点不可达"))
        client = TestClient(create_app(analysis_service=service))
        response = client.post(
            "/api/site-analysis",
            json={"entry_url": "https://x.edu.cn/list.htm", "contact": "mailto:a@b.cn"},
        )
        assert response.status_code == 502
        assert "目标站点不可达" in response.json()["detail"]

    def test_no_write_endpoints_exposed(self, client):
        """骨架阶段只读：不应存在任何落盘或建库端点。"""
        paths = {route.path for route in client.app.routes}
        assert not any(p.startswith("/api/") and "build" in p for p in paths)
