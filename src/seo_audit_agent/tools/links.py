from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from seo_audit_agent.fetcher import USER_AGENT, get_page


def extract_links(page) -> list[dict]:
    soup = BeautifulSoup(page.html, "html.parser")
    host = urlparse(page.final_url).netloc
    seen, links = set(), []
    for a in soup.find_all("a", href=True):
        absolute, _ = urldefrag(urljoin(page.final_url, a["href"].strip()))
        if urlparse(absolute).scheme not in ("http", "https") or absolute in seen:
            continue  # skips mailto:, tel:, javascript:, duplicates
        if absolute.rstrip("/") == page.final_url.rstrip("/"):
            continue  # skips #anchors and self-links
        seen.add(absolute)
        links.append({
            "url": absolute,
            "anchor": a.get_text(" ", strip=True)[:80],
            "internal": urlparse(absolute).netloc == host,
            "nofollow": "nofollow" in (a.get("rel") or []),
        })
    return links


def _status(client: httpx.Client, url: str) -> dict:
    try:
        r = client.head(url)
        if r.status_code in (403, 405, 501):  # some servers refuse HEAD; retry with GET
            r = client.get(url)
        return {"status": r.status_code, "final_url": str(r.url), "redirected": bool(r.history)}
    except httpx.HTTPError as e:
        return {"status": None, "error": type(e).__name__}


def check_links(url: str, max_links: int = 50) -> dict:
    """Find broken and redirected links on a page. Extracts every <a href>, then checks up to
    `max_links` of them. Returns link counts (internal, external, nofollow) and lists of
    broken links (4xx/5xx/unreachable) and redirected links. Takes 5-15 seconds.
    Use this when asked about broken links, 404s or link health."""
    page = get_page(url)
    if page.error or not page.html:
        return {"url": url, "error": page.error or f"No HTML returned (status {page.status})"}

    links = extract_links(page)
    to_check = links[:max_links]
    with httpx.Client(follow_redirects=True, timeout=10,
                      headers={"User-Agent": USER_AGENT}) as client, \
         ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda link: {**link, **_status(client, link["url"])}, to_check))

    broken = [{"url": r["url"], "anchor": r["anchor"], "status": r.get("status"), "error": r.get("error")}
              for r in results if r.get("status") is None or r["status"] >= 400]
    redirected = [{"url": r["url"], "final_url": r["final_url"]}
                  for r in results if r.get("redirected")]

    return {
        "url": page.final_url,
        "total_links": len(links),
        "checked": len(results),
        "skipped": len(links) - len(results),
        "internal": sum(link["internal"] for link in links),
        "external": sum(not link["internal"] for link in links),
        "nofollow": sum(link["nofollow"] for link in links),
        "broken": broken,
        "redirected": redirected,
    }