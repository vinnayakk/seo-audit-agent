import os
from typing import Literal

import httpx

from seo_audit_agent.net import get_with_retry

PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
LAB_METRICS = {
    "first-contentful-paint": "FCP",
    "largest-contentful-paint": "LCP",
    "total-blocking-time": "TBT",
    "cumulative-layout-shift": "CLS",
    "speed-index": "Speed Index",
}
ISSUE_GROUPS = {"insights", "diagnostics"}  # skip "metrics" (reported above) and "hidden"


def _lcp_element(audits: dict) -> str | None:
    items = audits.get("lcp-breakdown-insight", {}).get("details", {}).get("items", [])
    for item in items:
        if item.get("type") == "node":
            return item.get("nodeLabel") or item.get("snippet")
    return None


def check_speed(url: str, strategy: Literal["mobile", "desktop"] = "mobile") -> dict:
    """Measure page speed with Google PageSpeed Insights (Lighthouse). Returns a 0-100
    performance score, lab metrics (FCP, LCP, TBT, CLS, Speed Index), the LCP element,
    total page weight, the heaviest resources, real-user Core Web Vitals when Google has
    enough traffic data, and the top failing audits.
    Lab results vary between runs (the score can swing 20+ points), so treat one run as
    an estimate. Slow: takes 20-60 seconds. Use for speed, performance or Core Web Vitals."""
    params = {"url": url, "strategy": strategy, "category": "performance"}
    headers = {}
    if key := os.getenv("PAGESPEED_API_KEY"):
        headers["X-goog-api-key"] = key  # header, not URL, so the key never appears in logs

    try:
        # 2 attempts, not 3: each PageSpeed run can take up to 90 s
        r = get_with_retry(PSI_ENDPOINT, params=params, headers=headers, timeout=90, attempts=2)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return {"url": url, "error": f"PageSpeed API {e.response.status_code}: {e.response.text[:200]}"}
    except httpx.HTTPError as e:
        return {"url": url, "error": f"{type(e).__name__}: {e}"}

    data = r.json()
    lighthouse = data.get("lighthouseResult", {})
    audits = lighthouse.get("audits", {})
    perf = lighthouse.get("categories", {}).get("performance", {})
    score = perf.get("score")

    failing = []
    for ref in perf.get("auditRefs", []):
        audit = audits.get(ref.get("id"), {})
        if ref.get("group") in ISSUE_GROUPS and audit.get("score") is not None and audit["score"] < 0.9:
            failing.append(audit)
    failing.sort(key=lambda a: a["score"])

    weight = audits.get("total-byte-weight", {})
    heaviest = [{"url": i.get("url"), "kb": round(i.get("totalBytes", 0) / 1024)}
                for i in weight.get("details", {}).get("items", [])[:3]]

    field = data.get("loadingExperience", {}).get("metrics", {})
    return {
        "url": url,
        "strategy": strategy,
        "performance_score": round(score * 100) if score is not None else None,
        "lab_metrics": {name: audits.get(aid, {}).get("displayValue") for aid, name in LAB_METRICS.items()},
        "lcp_element": _lcp_element(audits),
        "page_weight_kb": round(weight["numericValue"] / 1024) if "numericValue" in weight else None,
        "heaviest_resources": heaviest,
        "field_data": {k: {"p75": v.get("percentile"), "rating": v.get("category")}
                       for k, v in field.items()} or None,
        "top_issues": [{"title": a.get("title"), "detail": a.get("displayValue")} for a in failing[:8]],
    }
