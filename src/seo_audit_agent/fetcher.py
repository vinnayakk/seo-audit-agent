import time
from dataclasses import dataclass, field
from functools import lru_cache

import httpx

USER_AGENT = "seo-audit-agent/0.1"
TIMEOUT = 15


@dataclass(frozen=True)
class Page:
    requested_url: str
    final_url: str | None = None
    status: int | None = None
    headers: dict = field(default_factory=dict)
    html: str = ""
    elapsed_ms: int | None = None
    redirects: tuple = ()
    error: str | None = None


@lru_cache(maxsize=32)
def get_page(url: str) -> Page:
    """Download a page once; repeat calls for the same URL come from the cache."""
    start = time.perf_counter()
    try:
        r = httpx.get(url, follow_redirects=True, timeout=TIMEOUT,
                      headers={"User-Agent": USER_AGENT})
    except httpx.HTTPError as e:
        return Page(requested_url=url, error=f"{type(e).__name__}: {e}")

    is_html = "html" in r.headers.get("content-type", "")
    return Page(
        requested_url=url,
        final_url=str(r.url),
        status=r.status_code,
        headers=dict(r.headers),
        html=r.text if is_html else "",
        elapsed_ms=round((time.perf_counter() - start) * 1000),
        redirects=tuple(f"{h.status_code} {h.url}" for h in r.history),
    )