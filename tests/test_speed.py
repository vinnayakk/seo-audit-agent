import httpx
import respx

from seo_audit_agent.tools import check_speed
from seo_audit_agent.tools.speed import PSI_ENDPOINT

FAKE_PSI = {
    "lighthouseResult": {
        "categories": {"performance": {"score": 0.72, "auditRefs": [
            {"id": "largest-contentful-paint", "group": "metrics"},
            {"id": "render-blocking-resources"},
            {"id": "unused-javascript"},
        ]}},
        "audits": {
            "largest-contentful-paint": {"title": "Largest Contentful Paint", "displayValue": "3.1 s", "score": 0.5},
            "render-blocking-resources": {"title": "Eliminate render-blocking resources",
                                          "displayValue": "Potential savings of 450 ms", "score": 0.3},
            "unused-javascript": {"title": "Reduce unused JavaScript", "score": 0.95},
        },
    },
    "loadingExperience": {"metrics": {
        "LARGEST_CONTENTFUL_PAINT_MS": {"percentile": 2900, "category": "AVERAGE"}}},
}


@respx.mock
def test_check_speed_summarises_response(monkeypatch):
    monkeypatch.delenv("PAGESPEED_API_KEY", raising=False)
    route = respx.get(url__startswith=PSI_ENDPOINT).mock(return_value=httpx.Response(200, json=FAKE_PSI))

    out = check_speed("https://shop.test/")

    assert route.calls.last.request.url.params["strategy"] == "mobile"
    assert out["performance_score"] == 72
    assert out["lab_metrics"]["LCP"] == "3.1 s"
    assert [i["title"] for i in out["top_issues"]] == ["Eliminate render-blocking resources"]
    assert out["field_data"]["LARGEST_CONTENTFUL_PAINT_MS"]["rating"] == "AVERAGE"


@respx.mock
def test_check_speed_api_error_is_data():
    respx.get(url__startswith=PSI_ENDPOINT).mock(return_value=httpx.Response(429, text="quota"))
    assert "error" in check_speed("https://shop.test/")