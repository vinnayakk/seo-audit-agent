from bs4 import BeautifulSoup

from seo_audit_agent.fetcher import get_page

# Common heuristics, not official Google limits (Google truncates by pixel width).
TITLE_RANGE = (30, 60)
DESC_RANGE = (70, 160)


def _meta(soup, **attrs):
    tag = soup.find("meta", attrs=attrs)
    return tag.get("content", "").strip() if tag else None


def fetch_title(url: str) -> dict:
    """Fetch a web page and return its <title> text, its length, the final URL and HTTP status."""
    page = get_page(url)
    if page.error:
        return {"url": url, "error": page.error}
    if page.status >= 400:
        return {"url": page.final_url, "status": page.status, "error": f"HTTP {page.status}"}
    soup = BeautifulSoup(page.html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    return {"url": page.final_url, "status": page.status,
            "title": title, "length": len(title) if title else 0}


def fetch_page(url: str) -> dict:
    """Check whether a page is reachable. Returns HTTP status, redirect chain, final URL,
    response time, HTML size, visible word count, HTTPS use and any X-Robots-Tag header.
    Use this first in any audit, and whenever the user asks if a page works or loads.
    response_time_ms is measured from the machine running this agent and includes DNS and
    TLS setup. It is not the server's processing time; use check_speed for real performance."""
    page = get_page(url)
    if page.error:
        return {"url": url, "error": page.error}

    soup = BeautifulSoup(page.html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    words = soup.get_text(" ", strip=True).split()

    return {
        "requested_url": url,
        "final_url": page.final_url,
        "status": page.status,
        "redirects": list(page.redirects),
        "response_time_ms": page.elapsed_ms,
        "content_type": page.headers.get("content-type"),
        "html_size_kb": round(len(page.html.encode()) / 1024, 1),
        "word_count": len(words),
        "is_https": page.final_url.startswith("https://"),
        "x_robots_tag": page.headers.get("x-robots-tag"),
    }


def check_meta_tags(url: str) -> dict:
    """Audit a page's on-page SEO tags: <title>, meta description, canonical link,
    meta robots, H1 headings, viewport, html lang and Open Graph tags.
    Returns each value plus an `issues` list of problems found.
    Use this when asked about titles, descriptions, indexing or on-page SEO."""
    page = get_page(url)
    if page.error or not page.html:
        return {"url": url, "error": page.error or f"No HTML returned (status {page.status})"}

    soup = BeautifulSoup(page.html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    description = _meta(soup, name="description")
    canonical_tag = soup.find("link", rel="canonical")
    canonical = canonical_tag.get("href") if canonical_tag else None
    robots = _meta(soup, name="robots")
    x_robots = page.headers.get("x-robots-tag")
    h1s = [h.get_text(" ", strip=True) for h in soup.find_all("h1")]
    lang = soup.html.get("lang") if soup.html else None
    viewport = _meta(soup, name="viewport")
    og = {p: _meta(soup, property=f"og:{p}") for p in ("title", "description", "image")}

    issues = []
    if not title:
        issues.append("Missing <title>")
    elif not TITLE_RANGE[0] <= len(title) <= TITLE_RANGE[1]:
        issues.append(f"Title is {len(title)} chars (aim for ~{TITLE_RANGE[0]}-{TITLE_RANGE[1]})")
    if not description:
        issues.append("Missing meta description")
    elif not DESC_RANGE[0] <= len(description) <= DESC_RANGE[1]:
        issues.append(f"Meta description is {len(description)} chars (aim for ~{DESC_RANGE[0]}-{DESC_RANGE[1]})")
    if not canonical:
        issues.append("No canonical link")
    elif canonical.rstrip("/") != page.final_url.rstrip("/"):
        issues.append(f"Canonical points to a different URL: {canonical}")
    if "noindex" in (robots or "").lower() or "noindex" in (x_robots or "").lower():
        issues.append("Page is set to noindex (it won't appear in search)")
    if len(h1s) != 1:
        issues.append(f"Found {len(h1s)} <h1> tags (expected 1)")
    if not viewport:
        issues.append("No viewport meta tag (may hurt mobile)")
    if not lang:
        issues.append("No lang attribute on <html>")
    if not og["title"] or not og["image"]:
        issues.append("Missing og:title or og:image (poor social sharing previews)")

    return {
        "url": page.final_url,
        "title": title, "title_length": len(title) if title else 0,
        "description": description, "description_length": len(description) if description else 0,
        "canonical": canonical, "robots": robots, "x_robots_tag": x_robots,
        "h1": h1s, "lang": lang, "viewport": viewport, "open_graph": og,
        "issues": issues,
    }