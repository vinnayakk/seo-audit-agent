from mcp.server.mcpserver import MCPServer

from seo_audit_agent.tools import fetch_title as _fetch_title

mcp = MCPServer("seo-audit")


@mcp.tool()
def fetch_title(url: str) -> dict:
    """Fetch a web page and return its <title>, length, final URL and HTTP status."""
    return _fetch_title(url)


if __name__ == "__main__":
    mcp.run(transport="stdio")