import httpx
import respx

from seo_audit_agent.tools import check_links

PAGE = """<html><body>
<a href="/about">About</a>
<a href="/about#team">About again</a>
<a href="https://other.test/x">External</a>
<a href="/gone">Gone</a>
<a href="mailto:hi@shop.test">Mail</a>
<a href="#top">Top</a>
</body></html>"""


@respx.mock
def test_check_links_finds_broken():
    respx.get("https://shop.test/").mock(return_value=httpx.Response(200, html=PAGE))
    respx.head("https://shop.test/about").mock(return_value=httpx.Response(200))
    respx.head("https://other.test/x").mock(return_value=httpx.Response(405))  # refuses HEAD
    respx.get("https://other.test/x").mock(return_value=httpx.Response(200))  # GET works
    respx.head("https://shop.test/gone").mock(return_value=httpx.Response(404))

    out = check_links("https://shop.test/")

    assert out["total_links"] == 3          # duplicates, mailto and #top dropped
    assert out["internal"] == 2
    assert out["external"] == 1
    assert [b["url"] for b in out["broken"]] == ["https://shop.test/gone"]