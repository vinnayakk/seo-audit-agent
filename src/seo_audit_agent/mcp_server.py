from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

from seo_audit_agent.tools import ALL_TOOLS

load_dotenv()  # MCP SDK v2 no longer reads .env itself; check_speed needs your API key

mcp = MCPServer("seo-audit")
for fn in ALL_TOOLS:
    mcp.tool()(fn)   # same as putting @mcp.tool() above each function

if __name__ == "__main__":
    mcp.run(transport="stdio")