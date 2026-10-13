import os
from typing import Literal

import httpx

PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
LAB_METRICS = {
    "first-contentful-paint": "FCP",
    "largest-contentful-paint": "LCP",
    "total-blocking-time": "TBT",
    "cumulative-layout-shift": "CLS",
    "speed-index": "Speed Index",
}


def check_speed(url: str, strategy: Literal["mobile", "desktop"] = "mobile") -> dict:
    """Measure page speed with Google PageSpeed Insights (Lighthouse). Returns a 0-100
    performance score, lab metrics (FCP, LCP, TBT, CLS, Speed Index), real-user Core Web
    Vitals when Google has enough traffic data, and the top issues slowing the page.
    Slow: takes 20-60 seconds. Use when asked about speed, performance or Core Web Vitals."""
    params = {"url": url, "strategy": strategy, "category": "performance"}
    if key := os.getenv("PAGESPEED_API_KEY"):
        params["key"] = key

    try:
        r = httpx.get(PSI_ENDPOINT, params=params, timeout=90)
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

    # Top issues: failing performance audits that aren't the headline metrics.
    failing = []
    for ref in perf.get("auditRefs", []):
        audit = audits.get(ref.get("id"), {})
        if ref.get("group") != "metrics" and audit.get("score") is not None and audit["score"] < 0.9:
            failing.append(audit)
    failing.sort(key=lambda a: a["score"])

    field = data.get("loadingExperience", {}).get("metrics", {})
    return {
        "url": url,
        "strategy": strategy,
        "performance_score": round(score * 100) if score is not None else None,
        "lab_metrics": {name: audits.get(aid, {}).get("displayValue") for aid, name in LAB_METRICS.items()},
        "field_data": {k: {"p75": v.get("percentile"), "rating": v.get("category")}
                       for k, v in field.items()} or None,
        "top_issues": [{"title": a.get("title"), "detail": a.get("displayValue")} for a in failing[:5]],
    }