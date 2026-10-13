import argparse
import json
import logging
import sys

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes

from seo_audit_agent.mcp_server import mcp
from seo_audit_agent.tools import ALL_TOOLS

load_dotenv()
for noisy in ("httpx", "httpx2"):  # stop logging every request URL
    logging.getLogger(noisy).setLevel(logging.WARNING)

client = Anthropic()
langfuse = get_client()  # reads LANGFUSE_* from .env; without keys it is a silent no-op
MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4096
SYSTEM = ("You are an SEO auditor. Use the tools to inspect pages before answering. "
          "Report concrete findings, then prioritised fixes.")

# Single source of truth: names, descriptions and schemas come from the MCP server.
TOOLS = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
         for t in anyio.run(mcp.list_tools)]
TOOL_FUNCS = {fn.__name__: fn for fn in ALL_TOOLS}


def call_tool(name: str, args: dict) -> dict:
    """Run one tool inside its own Langfuse 'tool' observation (timed automatically)."""
    with langfuse.start_as_current_observation(name=name, as_type="tool", input=args) as span:
        try:
            output = TOOL_FUNCS[name](**args)
        except Exception as e:  # a crashing tool should not kill the whole audit
            output = {"error": f"{type(e).__name__}: {e}"}
        span.update(output=output, level="ERROR" if "error" in output else "DEFAULT")
        return output


@observe(name="seo-audit", as_type="agent")
def run_agent(question: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": question}]
    for turn in range(1, max_turns + 1):
        with langfuse.start_as_current_observation(
            name=f"claude-turn-{turn}", as_type="generation", model=MODEL,
            input=messages, model_parameters={"max_tokens": MAX_TOKENS},
        ) as gen:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                                          tools=TOOLS, messages=messages)
            content = [block.model_dump(exclude_none=True) for block in resp.content]
            gen.update(output=content, usage_details={"input": resp.usage.input_tokens,
                                                      "output": resp.usage.output_tokens})
        messages.append({"role": "assistant", "content": content})
        if resp.stop_reason == "max_tokens":
            print(f"⚠ Reply was cut off at {MAX_TOKENS} tokens; raise MAX_TOKENS.", file=sys.stderr)

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"→ {block.name}({block.input})")
                output = call_tool(block.name, block.input)
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
        args.url = url
        question = FULL_AUDIT.format(url=url)

    # Everything inside this block lands in one trace, tagged and searchable in Langfuse.
    with propagate_attributes(trace_name="seo-audit", tags=["cli"],
                              metadata={"url": args.url or "", "mode": "question" if args.question else "full-audit"}):
        print(run_agent(question))
    langfuse.flush()  # short-lived script: send buffered traces before exiting


if __name__ == "__main__":
    main()
