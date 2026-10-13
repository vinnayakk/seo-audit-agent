import httpx
import respx

from seo_audit_agent.tools import check_speed
from seo_audit_agent.tools.speed import PSI_ENDPOINT

FAKE_PSI = {
    "lighthouseResult": {
        "categories": {"performance": {"score": 0.72, "auditRefs": [
            {"id": "largest-contentful-paint", "group": "metrics"},
            {"id": "render-blocking-insight", "group": "insights"},
            {"id": "image-delivery-insight", "group": "insights"},
            {"id": "interactive", "group": "hidden"},
            {"id": "unused-javascript", "group": "diagnostics"},
        ]}},
        "audits": {
            "largest-contentful-paint": {"title": "Largest Contentful Paint", "displayValue": "3.1 s", "score": 0.5},
            "render-blocking-insight": {"title": "Render-blocking requests", "displayValue": "Est savings of 170 ms", "score": 0},
            "image-delivery-insight": {"title": "Improve image delivery", "displayValue": "Est savings of 6,652 KiB", "score": 0.5},
            "interactive": {"title": "Time to Interactive", "displayValue": "7.5 s", "score": 0.48},
            "unused-javascript": {"title": "Reduce unused JavaScript", "score": 0.95},
            "total-byte-weight": {"numericValue": 7555870, "details": {"items": [
                {"url": "https://shop.test/cover.gif", "totalBytes": 5572621}]}},
            "lcp-breakdown-insight": {"details": {"items": [
                {"type": "table"}, {"type": "node", "nodeLabel": "Hero paragraph"}]}},
        },
    },
    "loadingExperience": {"metrics": {
        "LARGEST_CONTENTFUL_PAINT_MS": {"percentile": 2900, "category": "AVERAGE"}}},
}


@respx.mock
def test_check_speed_summarises_response(monkeypatch):
    monkeypatch.setenv("PAGESPEED_API_KEY", "test-key")
    route = respx.get(url__startswith=PSI_ENDPOINT).mock(return_value=httpx.Response(200, json=FAKE_PSI))

    out = check_speed("https://shop.test/")

    request = route.calls.last.request
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key" not in request.url.params            # key never in the URL
    assert request.url.params["strategy"] == "mobile"
    assert out["performance_score"] == 72
    assert out["lab_metrics"]["LCP"] == "3.1 s"
    assert out["lcp_element"] == "Hero paragraph"
    assert out["page_weight_kb"] == 7379
    assert out["heaviest_resources"][0]["kb"] == 5442
    titles = [i["title"] for i in out["top_issues"]]
    assert titles == ["Render-blocking requests", "Improve image delivery"]  # hidden + passing excluded
    assert out["field_data"]["LARGEST_CONTENTFUL_PAINT_MS"]["rating"] == "AVERAGE"


@respx.mock
def test_check_speed_api_error_is_data():
    respx.get(url__startswith=PSI_ENDPOINT).mock(return_value=httpx.Response(429, text="quota"))
    assert "error" in check_speed("https://shop.test/")
