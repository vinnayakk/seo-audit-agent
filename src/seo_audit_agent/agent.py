import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import anyio
from anthropic import Anthropic
from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from pydantic import ValidationError

from seo_audit_agent.mcp_server import mcp
from seo_audit_agent.report import SUBMIT_TOOL, AuditReport, AuditRun, render_markdown
from seo_audit_agent.tools import ALL_TOOLS

load_dotenv()
for noisy in ("httpx", "httpx2"):  # stop logging every request URL
    logging.getLogger(noisy).setLevel(logging.WARNING)

client = Anthropic(timeout=120, max_retries=3)  # SDK retries 429/5xx/overloaded itself
langfuse = get_client()  # reads LANGFUSE_* from .env; without keys it is a silent no-op
MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 8192  # structured reports carry evidence, so they're longer than prose
MAX_TURNS = 10

SYSTEM = """You are an SEO auditor. Use the tools to inspect pages before answering.
Rules for audits:
- Only state facts that appear in a tool output. Cite them as evidence (tool, field, exact value),
  pointing at one value (e.g. top_issues[2].detail), never a whole object.
- Name a specific file, script or image as the cause of a problem only if a tool output lists it
  for that problem (e.g. in top_issues[n].urls). Otherwise describe the fix without naming a culprit.
- Each fix must be one concrete change, not a target like "get LCP under 2.5 s".
- Don't recommend something the tool data shows the site already does.
- If a check failed or wasn't run, put it in not_checked instead of guessing.
- Keep suggested titles under 60 characters and meta descriptions between 70 and 160.
  Give exactly one suggestion per fix; don't revise it mid-sentence."""

STEP_LIMIT_NOTE = ("Step limit reached. Do not call more tools. Finish now using only the data "
                   "you already have, and list any checks you could not complete.")
SUBMIT_NUDGE = "Now write the final audit report."

# Single source of truth: names, descriptions and schemas come from the MCP server.
TOOLS = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
         for t in anyio.run(mcp.list_tools)]
TOOL_FUNCS = {fn.__name__: fn for fn in ALL_TOOLS}


class ReportError(RuntimeError):
    """Claude did not produce a valid report within the step limit."""


def call_tool(name: str, args: dict) -> dict:
    """Run one tool inside its own Langfuse 'tool' observation (timed automatically)."""
    with langfuse.start_as_current_observation(name=name, as_type="tool", input=args) as span:
        try:
            output = TOOL_FUNCS[name](**args)
        except Exception as e:  # a crashing tool should not kill the whole audit
            output = {"error": f"{type(e).__name__}: {e}"}
        span.update(output=output, level="ERROR" if "error" in output else "DEFAULT")
        return output


def validate_report(args: dict) -> AuditReport:
    """Pydantic check of what Claude submitted, inside its own observation so failures are visible."""
    with langfuse.start_as_current_observation(name="validate_report", input=args) as span:
        try:
            report = AuditReport.model_validate(args)
        except ValidationError as e:
            span.update(output=str(e), level="WARNING")
            raise
        span.update(output={"findings": len(report.findings)})
        return report


def final_report(messages: list, tools: list, turn: int) -> AuditReport:
    """Make Claude answer NOW, in the AuditReport shape, using structured outputs.

    Opus 5.5, Sonnet 5.5 and Fable 5.1 reject forced tool calls (tool_choice "tool"/"any" -> 400),
    so instead of forcing submit_report we forbid tools (tool_choice "none", which every model
    supports) and constrain the reply itself to the report's JSON schema.
    """
    with langfuse.start_as_current_observation(
        name=f"claude-turn-{turn}", as_type="generation", model=MODEL, input=messages,
        model_parameters={"max_tokens": MAX_TOKENS, "tool_choice": "none", "output_format": "AuditReport"},
    ) as gen:
        resp = client.messages.parse(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM, tools=tools,
                                     tool_choice={"type": "none"}, messages=messages, output_format=AuditReport)
        gen.update(output=resp.parsed_output.model_dump() if resp.parsed_output else None,
                   usage_details={"input": resp.usage.input_tokens, "output": resp.usage.output_tokens})
    if resp.parsed_output is None:
        raise ReportError("Claude returned no report in the structured-output turn.")
    return validate_report(resp.parsed_output.model_dump())  # also checks min_length etc. and logs it


@observe(name="seo-audit", as_type="agent")
def run_agent(question: str, max_turns: int = MAX_TURNS, structured: bool = False) -> AuditRun:
    """Tool-calling loop. structured=True ends with a validated AuditReport (via submit_report, or a
    structured-output turn when Claude must answer now); structured=False ends with free text."""
    run = AuditRun(question=question, trace_id=langfuse.get_current_trace_id())
    tools = TOOLS + [SUBMIT_TOOL] if structured else TOOLS
    messages = [{"role": "user", "content": question}]
    must_report = False  # set when Claude wrote prose or submitted an invalid report

    for turn in range(1, max_turns + 1):
        last_turn = turn == max_turns
        if last_turn and turn > 1 and isinstance(messages[-1]["content"], list):
            messages[-1]["content"].append({"type": "text", "text": STEP_LIMIT_NOTE})

        # Rules live in code, not just in the prompt.
        if structured and (last_turn or must_report):
            try:
                run.report = final_report(messages, tools, turn)   # report now, schema-guaranteed
                return run
            except ValidationError as e:
                raise ReportError(f"Final report failed validation: {e}") from e
        tool_choice = {"type": "none"} if last_turn else {"type": "auto"}

        with langfuse.start_as_current_observation(
            name=f"claude-turn-{turn}", as_type="generation", model=MODEL, input=messages,
            model_parameters={"max_tokens": MAX_TOKENS, "tool_choice": tool_choice["type"]},
        ) as gen:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM,
                                          tools=tools, tool_choice=tool_choice, messages=messages)
            content = [block.model_dump(exclude_none=True) for block in resp.content]
            gen.update(output=content, usage_details={"input": resp.usage.input_tokens,
                                                      "output": resp.usage.output_tokens})
        messages.append({"role": "assistant", "content": content})
        if resp.stop_reason == "max_tokens":
            print(f"⚠ Reply was cut off at {MAX_TOKENS} tokens; raise MAX_TOKENS.", file=sys.stderr)

        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text")
            if not structured:
                run.text = text
                return run
            # Claude wrote prose instead of submitting: next turn asks for the report directly.
            messages.append({"role": "user", "content": SUBMIT_NUDGE})
            must_report = True
            continue

        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name == "submit_report":
                try:
                    run.report = validate_report(block.input)
                    return run
                except ValidationError as e:
                    # Send the exact errors back so Claude can correct itself on the next turn.
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": f"Report rejected, fix these errors and resubmit:\n{e}"})
                    must_report = True
                    continue
            print(f"→ {block.name}({block.input})")
            output = call_tool(block.name, block.input)
            run.tool_outputs.append({"tool": block.name, "input": block.input, "output": output})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(output)})
        messages.append({"role": "user", "content": results})

    if structured:
        raise ReportError(f"No valid report after {max_turns} turns.")
    run.text = "Stopped: the step limit was reached before an answer was written."
    return run


FULL_AUDIT = ("Do a full SEO audit of {url}: reachability, meta tags, broken links "
              "and mobile speed. Then submit a prioritised report.")


def save_run(run: AuditRun, folder: str) -> Path:
    """Write the run (report + tool outputs) as JSON, ready to become an eval case."""
    url = run.report.url if run.report else "question"
    slug = re.sub(r"[^a-z0-9]+", "-", re.sub(r"^https?://(www\.)?", "", url.lower())).strip("-")
    path = Path(folder) / f"{slug}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(prog="seo-audit", description="AI SEO audit agent")
    parser.add_argument("url", nargs="?", help="Page to audit, e.g. https://example.com")
    parser.add_argument("-q", "--question", help="Ask something specific instead of a full audit")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS,
                        help=f"Maximum Claude turns before forcing a report (default {MAX_TURNS})")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON instead of Markdown")
    parser.add_argument("--save", metavar="FOLDER", help="Also save the run (report + tool outputs) as JSON here")
    args = parser.parse_args()

    if args.question:
        question = f"{args.question}\nURL: {args.url}" if args.url else args.question
    else:
        url = args.url or input("URL to audit: ").strip()
        if not url:
            sys.exit("No URL given.")
        args.url = url
        question = FULL_AUDIT.format(url=url)
    structured = not args.question

    # Everything inside this block lands in one trace, tagged and searchable in Langfuse.
    with propagate_attributes(trace_name="seo-audit", tags=["cli"],
                              metadata={"url": args.url or "", "mode": "full-audit" if structured else "question"}):
        try:
            run = run_agent(question, max_turns=args.max_turns, structured=structured)
        except ReportError as e:
            langfuse.flush()
            sys.exit(f"✗ {e}")
    langfuse.flush()  # short-lived script: send buffered traces before exiting

    if run.report:
        print(run.report.model_dump_json(indent=2) if args.json else render_markdown(run.report))
    else:
        print(run.text)
    if args.save:
        print(f"\nSaved run to {save_run(run, args.save)}", file=sys.stderr)


if __name__ == "__main__":
    main()
