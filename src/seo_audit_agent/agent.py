import argparse
import json
import logging
import sys

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv

from seo_audit_agent.mcp_server import mcp
from seo_audit_agent.tools import ALL_TOOLS

load_dotenv()
logging.getLogger("httpx").setLevel(logging.WARNING)  # stop logging every request URL
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


FULL_AUDIT = ("Do a full SEO audit of {url}: reachability, meta tags, broken links "
              "and mobile speed. Give me a prioritised fix list.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="seo-audit", description="AI SEO audit agent")
    parser.add_argument("url", nargs="?", help="Page to audit, e.g. https://example.com")
    parser.add_argument("-q", "--question", help="Ask something specific instead of a full audit")
    args = parser.parse_args()

    if args.question:
        question = f"{args.question}\nURL: {args.url}" if args.url else args.question
    else:
        url = args.url or input("URL to audit: ").strip()
        if not url:
            sys.exit("No URL given.")
        question = FULL_AUDIT.format(url=url)

    print(run_agent(question))


if __name__ == "__main__":
    main()
