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
    """Compare values the way a human would: None == 'null', True == 'true', non-breaking spaces ignored."""
    if value is None:
        return "null"
    if isinstance(value, bool):  # Python prints True; JSON (and Claude) say true
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("\xa0", " ").strip()


def scalars(obj) -> list[str]:
    """All non-null leaf values inside a dict/list, normalised."""
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in scalars(x)]
    if isinstance(obj, list):
        return [v for x in obj for v in scalars(x)]
    return [] if obj is None else [normalise(obj)]


def quoted_suggestions(text: str) -> list[str]:
    return [next(g for g in m.groups() if g) for m in QUOTED.finditer(text)]


def cites(finding: Finding, *fields: str) -> bool:
    return any(e.field.split(".")[0].split("[")[0] in fields for e in finding.evidence)


# ---------- checks ----------

def check_evidence(run: AuditRun) -> list[CheckResult]:
    """Every cited value must exist at that field in that tool's output.

    Two separate results, because they mean different things:
    - evidence_grounded: the value is wrong, the field is missing, or the tool never ran (a real error)
    - evidence_precise: the citation points at a whole object (e.g. top_issues[2]) instead of one
      value (top_issues[2].detail). Every value in it matches, so it isn't wrong, just imprecise.
    """
    grounded, precise = CheckResult("evidence_grounded"), CheckResult("evidence_precise")
    outputs = {t["tool"]: t["output"] for t in run.tool_outputs}
    for i, f in enumerate(run.report.findings, 1):
        for e in f.evidence:
            where = f"finding {i}: {e.tool}.{e.field}"
            if e.tool not in outputs:
                grounded.failures.append(f"{where} cites a tool that was never run")
                continue
            try:
                actual = resolve(outputs[e.tool], e.field)
            except (KeyError, IndexError, TypeError, ValueError):
                grounded.failures.append(f"{where} does not exist in the tool output")
                continue
            cited = normalise(e.value)
            if normalise(actual) == cited:
                continue
            if isinstance(actual, dict) and all(v in cited for v in scalars(actual)):
                precise.failures.append(f"{where} cites a whole object; cite one field inside it instead")
            else:
                grounded.failures.append(f"{where} cited {e.value!r} but the tool returned {normalise(actual)!r}")
    return [grounded, precise]


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
    """Problems so serious that any audit missing them is wrong, whatever else it says.
    Mentioning them anywhere counts (summary, findings or not_checked): on a deliberate 404 page,
    "this 404 is correct" in the summary is the right answer, not a missing finding."""
    result = CheckResult("critical_issues_reported")
    outputs = {t["tool"]: t["output"] for t in run.tool_outputs}
    report = run.report
    text = " ".join([report.summary, *report.not_checked,
                     *(f"{f.issue} {f.fix}" for f in report.findings)]).lower()
    categories = {f.category for f in report.findings}
    page = outputs.get("fetch_page", {})
    meta = outputs.get("check_meta_tags", {})
    links = outputs.get("check_links", {})

    status = page.get("status") or 0
    if status >= 400 and "reachability" not in categories and str(status) not in text:
        result.failures.append(f"page returned HTTP {status} but the report never mentions it")
    robots = f"{meta.get('robots') or ''} {meta.get('x_robots_tag') or ''}".lower()
    if "noindex" in robots and "noindex" not in text:
        result.failures.append("page is noindex but the report never mentions it")
    if "title" in meta and not meta.get("title") and not any(cites(f, "title", "title_length")
                                                             for f in report.findings):
        result.failures.append("page has no <title> but no finding cites it")
    if links.get("broken") and "links" not in categories:
        result.failures.append(f"{len(links['broken'])} broken link(s) but no links finding")
    return result


CHECKS = [check_suggestion_lengths, check_failed_tools_disclosed, check_critical_issues_reported]


def evaluate(run: AuditRun) -> list[CheckResult]:
    if run.report is None:
        return [CheckResult("has_report", ["run has no structured report"])]
    return check_evidence(run) + [check(run) for check in CHECKS]


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
