"""Deterministic checks on saved audit runs: the part of the judge that needs no LLM.

    uv run seo-eval evals/cases            # check every saved run
    uv run seo-eval evals/cases/foo.json   # check one

Each check returns a list of failures (empty = pass). Code checks are cheap, exact and
never drift, so anything that *can* be checked this way should be. The LLM judge (next step)
only gets what code can't decide: is the reasoning sound, is the priority sensible?
"""
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from seo_audit_agent.report import AuditRun, Finding

TITLE_MAX = 60
DESC_RANGE = (70, 160)
QUOTED = re.compile(r"'([^']{8,})'|\"([^\"]{8,})\"|‘([^’]{8,})’|“([^”]{8,})”")


@dataclass
class CheckResult:
    name: str
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


# ---------- helpers ----------

def resolve(obj, path: str):
    """Follow a path like 'heaviest_resources[0].kb' or 'open_graph.image' into a dict."""
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        obj = obj[int(part[1:-1])] if part.startswith("[") else obj[part]
    return obj


def normalise(value) -> str:
    """Compare values the way a human would: None == 'null', non-breaking spaces ignored."""
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).replace("\xa0", " ").strip()


def quoted_suggestions(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in QUOTED.finditer(text)]


def cites(finding: Finding, *fields: str) -> bool:
    return any(e.field.split(".")[0].split("[")[0] in fields for e in finding.evidence)


# ---------- checks ----------

def check_evidence_grounded(run: AuditRun) -> CheckResult:
    """Every cited value must exist at that field in that tool's output."""
    result = CheckResult("evidence_grounded")
    outputs = {t["tool"]: t["output"] for t in run.tool_outputs}
    for i, f in enumerate(run.report.findings, 1):
        for e in f.evidence:
            where = f"finding {i}: {e.tool}.{e.field}"
            if e.tool not in outputs:
                result.failures.append(f"{where} cites a tool that was never run")
                continue
            try:
                actual = normalise(resolve(outputs[e.tool], e.field))
            except (KeyError, IndexError, TypeError, ValueError):
                result.failures.append(f"{where} does not exist in the tool output")
                continue
            if actual != normalise(e.value):
                result.failures.append(f"{where} cited {e.value!r} but the tool returned {actual!r}")
    return result


def check_suggestion_lengths(run: AuditRun) -> CheckResult:
    """Suggested titles must be <= 60 chars; suggested descriptions 70-160 (quoted examples only)."""
    result = CheckResult("suggestion_lengths")
    for i, f in enumerate(run.report.findings, 1):
        for text in quoted_suggestions(f.fix):
            n = len(text)
            if cites(f, "title", "title_length") and n > TITLE_MAX:
                result.failures.append(f"finding {i}: suggested title is {n} chars (max {TITLE_MAX}): {text!r}")
            if cites(f, "description", "description_length") and not DESC_RANGE[0] <= n <= DESC_RANGE[1]:
                result.failures.append(f"finding {i}: suggested description is {n} chars "
                                       f"(want {DESC_RANGE[0]}-{DESC_RANGE[1]}): {text!r}")
    return result


def check_failed_tools_disclosed(run: AuditRun) -> CheckResult:
    """If a tool returned an error, the report must admit a gap in not_checked."""
    result = CheckResult("failed_tools_disclosed")
    failed = [t["tool"] for t in run.tool_outputs if "error" in t["output"]]
    if failed and not run.report.not_checked:
        result.failures.append(f"{', '.join(failed)} failed but not_checked is empty")
    return result


def check_critical_issues_reported(run: AuditRun) -> CheckResult:
    """Problems so serious that any audit missing them is wrong, whatever else it says."""
    result = CheckResult("critical_issues_reported")
    outputs = {t["tool"]: t["output"] for t in run.tool_outputs}
    categories = {f.category for f in run.report.findings}
    page = outputs.get("fetch_page", {})
    meta = outputs.get("check_meta_tags", {})
    links = outputs.get("check_links", {})

    if (page.get("status") or 0) >= 400 and "reachability" not in categories:
        result.failures.append(f"page returned HTTP {page['status']} but there is no reachability finding")
    robots = f"{meta.get('robots') or ''} {meta.get('x_robots_tag') or ''}".lower()
    if "noindex" in robots and not any("noindex" in (f.issue + f.fix).lower() for f in run.report.findings):
        result.failures.append("page is noindex but no finding mentions it")
    if "title" in meta and not meta.get("title") and not any(cites(f, "title", "title_length")
                                                             for f in run.report.findings):
        result.failures.append("page has no <title> but no finding cites it")
    if links.get("broken") and "links" not in categories:
        result.failures.append(f"{len(links['broken'])} broken link(s) but no links finding")
    return result


CHECKS = [check_evidence_grounded, check_suggestion_lengths,
          check_failed_tools_disclosed, check_critical_issues_reported]


def evaluate(run: AuditRun) -> list[CheckResult]:
    if run.report is None:
        return [CheckResult("has_report", ["run has no structured report"])]
    return [check(run) for check in CHECKS]


# ---------- CLI ----------

def load_runs(target: str) -> list[tuple[Path, AuditRun]]:
    path = Path(target)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    return [(f, AuditRun.model_validate_json(f.read_text())) for f in files]


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "evals/cases"
    runs = load_runs(target)
    if not runs:
        sys.exit(f"No saved runs found in {target}")

    total_checks = total_passed = 0
    for path, run in runs:
        results = evaluate(run)
        passed = sum(r.passed for r in results)
        total_checks += len(results)
        total_passed += passed
        print(f"\n{path.name}  ({passed}/{len(results)} checks passed)")
        for r in results:
            print(f"  {'✅' if r.passed else '❌'} {r.name}")
            for failure in r.failures:
                print(f"      - {failure}")

    print(f"\nOverall: {total_passed}/{total_checks} checks passed across {len(runs)} run(s)")
    sys.exit(0 if total_passed == total_checks else 1)  # non-zero exit lets CI fail on regressions


if __name__ == "__main__":
    main()
