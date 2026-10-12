import httpx
from bs4 import BeautifulSoup


def fetch_title(url: str) -> dict:
    """Fetch a page and return its <title> plus basic facts."""
    try:
        r = httpx.get(url, follow_redirects=True, timeout=10,
                      headers={"User-Agent": "seo-audit-agent/0.1"})
        r.raise_for_status()
    except httpx.HTTPError as e:
        return {"url": url, "error": str(e)}   # errors go back as data, not crashes
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    return {"url": str(r.url), "status": r.status_code,
            "title": title, "length": len(title) if title else 0}