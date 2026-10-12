import httpx
import anyio

from seo_audit_agent import tools
from seo_audit_agent.mcp_server import mcp


def fake_get(html, status=200):
    def _get(url, **kwargs):
        return httpx.Response(status, text=html, request=httpx.Request("GET", url))
    return _get


def test_returns_title(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", fake_get("<title> Hello SEO </title>"))
    out = tools.fetch_title("https://example.com")
    assert out["title"] == "Hello SEO"
    assert out["length"] == 9


def test_missing_title(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", fake_get("<html><body>hi</body></html>"))
    assert tools.fetch_title("https://example.com")["title"] is None


def test_http_error_returned_as_data(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", fake_get("nope", status=404))
    assert "error" in tools.fetch_title("https://example.com/missing")


def test_mcp_server_exposes_fetch_title():
    tools = anyio.run(mcp.list_tools)
    assert "fetch_title" in [t.name for t in tools]