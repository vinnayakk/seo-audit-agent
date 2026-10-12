import json
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

from seo_audit_agent.tools import fetch_title

load_dotenv()
client = Anthropic()
MODEL = "claude-sonnet-5-5"

TOOLS = [{
    "name": "fetch_title",
    "description": ("Fetch a web page and return its HTML <title> text, its character "
                    "length, the final URL after redirects, and the HTTP status. "
                    "Use this whenever the user asks about a page's title."),
    "input_schema": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Full URL including https://"}},
        "required": ["url"],
    },
}]
TOOL_FUNCS = {"fetch_title": fetch_title}


def run_agent(question: str, max_turns: int = 5) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        resp = client.messages.create(model=MODEL, max_tokens=1024,
                                      tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":          # model is done
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
    q = " ".join(sys.argv[1:]) or "What's the title of https://example.com and is it a good length for SEO?"
    print(run_agent(q))