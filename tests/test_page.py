import httpx
import respx

from seo_audit_agent.tools import check_meta_tags, fetch_page, fetch_title

GOOD_HTML = """<html lang="en"><head>
<title>Handmade Leather Wallets | Ridgeway Goods</title>
<meta name="description" content="Slim, hand-stitched leather wallets made in small batches. Free shipping on orders over $50 and a lifetime repair promise.">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="canonical" href="https://shop.test/">
<meta property="og:title" content="Ridgeway Goods">
<meta property="og:image" content="https://shop.test/og.png">
</head><body><h1>Handmade wallets</h1><p>Some words here.</p></body></html>"""

BAD_HTML = """<html><head><title>Home</title>
<meta name="robots" content="noindex"></head>
<body><h1>A</h1><h1>B</h1></body></html>"""


@respx.mock
def test_fetch_title():
    respx.get("https://shop.test/").mock(return_value=httpx.Response(200, html=GOOD_HTML))
    out = fetch_title("https://shop.test/")
    assert out["title"] == "Handmade Leather Wallets | Ridgeway Goods"
    assert out["length"] == 41


@respx.mock
def test_fetch_title_404_is_error_data():
    respx.get("https://shop.test/missing").mock(return_value=httpx.Response(404, html="nope"))
    assert "error" in fetch_title("https://shop.test/missing")


@respx.mock
def test_fetch_page_follows_redirect():
    respx.get("http://shop.test/").mock(
        return_value=httpx.Response(301, headers={"Location": "https://shop.test/"}))
    respx.get("https://shop.test/").mock(return_value=httpx.Response(200, html=GOOD_HTML))
    out = fetch_page("http://shop.test/")
    assert out["final_url"] == "https://shop.test/"
    assert len(out["redirects"]) == 1
    assert out["is_https"] is True


@respx.mock
def test_meta_tags_good_page_has_no_issues():
    respx.get("https://shop.test/").mock(return_value=httpx.Response(200, html=GOOD_HTML))
    assert check_meta_tags("https://shop.test/")["issues"] == []


@respx.mock
def test_meta_tags_bad_page_flags_problems():
    respx.get("https://shop.test/").mock(return_value=httpx.Response(200, html=BAD_HTML))
    issues = " ".join(check_meta_tags("https://shop.test/")["issues"])
    assert "noindex" in issues
    assert "Missing meta description" in issues
    assert "2 <h1>" in issues

@respx.mock
def test_fetch_page_retries_a_flaky_server():
    respx.get("https://shop.test/").mock(
        side_effect=[httpx.Response(502), httpx.Response(200, html=GOOD_HTML)])
    assert fetch_page("https://shop.test/")["status"] == 200
