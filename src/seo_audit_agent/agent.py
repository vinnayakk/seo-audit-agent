import json
import sys

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv

from seo_audit_agent.mcp_server import mcp
from seo_audit_agent.tools import ALL_TOOLS

load_dotenv()
client = Anthropic()
MODEL = "claude-sonnet-5-5"
SYSTEM = ("You are an SEO auditor. Use the tools to inspect pages before answering. "
          "Report concrete findings, then prioritised fixes.")

# Single source of truth: names, descriptions and schemas come from the MCP server.
TOOLS = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
         for t in anyio.run(mcp.list_tools)]
TOOL_FUNCS = {fn.__name__: fn for fn in ALL_TOOLS}


def run_agent(question: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        resp = client.messages.create(model=MODEL, max_tokens=2048, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"→ {block.name}({block.input})")
                output = TOOL_FUNCS[block.name](**block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(output)})
        messages.append({"role": "user", "content": results})
    return "Stopped: hit max_turns."


if __name__ == "__main__":
    print(run_agent(" ".join(sys.argv[1:]) or "Audit the meta tags of https://www.vinnayakk.com"))