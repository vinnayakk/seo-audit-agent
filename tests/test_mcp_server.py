import anyio

from seo_audit_agent.mcp_server import mcp


def test_server_exposes_all_tools():
    names = {t.name for t in anyio.run(mcp.list_tools)}
    assert names == {"fetch_title", "fetch_page", "check_meta_tags", "check_links"}