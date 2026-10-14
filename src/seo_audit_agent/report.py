"""The structured audit report: what Claude must return, how it's prioritised, how it's shown."""
from typing import Literal

from pydantic import BaseModel, Field

ToolName = Literal["fetch_page", "check_meta_tags", "check_links", "check_speed"]
Level = Literal["high", "medium", "low"]


class Evidence(BaseModel):
    tool: ToolName = Field(description="The tool whose output proves this finding")
    field: str = Field(description="Path to the field in that tool's output, e.g. "
                                   "'heaviest_resources[0].kb' or 'open_graph.image'")
    value: str = Field(description="The value copied exactly as the tool returned it, e.g. '5442' or 'null'")


class Finding(BaseModel):
    category: Literal["reachability", "meta_tags", "links", "speed", "content"]
    issue: str = Field(description="What is wrong, in one sentence")
    evidence: list[Evidence] = Field(min_length=1, description="At least one tool output that proves the issue")
    fix: str = Field(description="One concrete change to make (an action, not a target metric)")
    impact: Level = Field(description="How much fixing this improves search visibility, speed or sharing")
    effort: Level = Field(description="How much work the fix takes: low = minutes, medium = hours, high = days")


class AuditReport(BaseModel):
    url: str
    summary: str = Field(description="Two or three sentences on overall health and the biggest problem")
    findings: list[Finding]
    not_checked: list[str] = Field(default_factory=list,
                                   description="Checks that were not run or failed, so the reader knows the gaps")


class AuditRun(BaseModel):
    """Everything one run produced: the report plus the raw tool outputs it was based on.
    Saved runs become eval cases: the judge compares report claims against tool_outputs."""
    question: str
    trace_id: str | None = None         # Langfuse trace, so judge scores can be attached to it later
    report: AuditReport | None = None   # structured audits
    text: str = ""                      # free-text answers (-q mode)
    tool_outputs: list[dict] = Field(default_factory=list)


# ---------- Priority is computed in code, not chosen by the model ----------

WEIGHT = {"high": 3, "medium": 2, "low": 1}


def priority_score(f: Finding) -> int:
    """Impact counts double; effort subtracts. high/low = 5 ... low/high = -1."""
    return 2 * WEIGHT[f.impact] - WEIGHT[f.effort]


def priority_label(f: Finding) -> str:
    score = priority_score(f)
    return "P1" if score >= 4 else "P2" if score >= 2 else "P3"


def prioritised(report: AuditReport) -> list[Finding]:
    return sorted(report.findings, key=lambda f: (-priority_score(f), -WEIGHT[f.impact]))


def render_markdown(report: AuditReport) -> str:
    lines = [f"# SEO audit: {report.url}", "", report.summary, "",
             "## Prioritised fixes", "",
             "| # | Priority | Issue | Fix | Impact | Effort |",
             "|---|---|---|---|---|---|"]
    ordered = prioritised(report)
    for i, f in enumerate(ordered, 1):
        cells = [str(i), priority_label(f), f.issue, f.fix, f.impact, f.effort]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")

    lines += ["", "## Evidence", ""]
    for i, f in enumerate(ordered, 1):
        cited = "; ".join(f"`{e.tool}.{e.field}` = {e.value}" for e in f.evidence)
        lines.append(f"{i}. {cited}")

    if report.not_checked:
        lines += ["", "## Not checked", ""] + [f"- {item}" for item in report.not_checked]
    return "\n".join(lines)


SUBMIT_TOOL = {
    "name": "submit_report",
    "description": ("Submit the final SEO audit. Call this once, after running the checks. "
                    "Every finding must cite evidence copied from tool outputs."),
    "input_schema": AuditReport.model_json_schema(),
}
