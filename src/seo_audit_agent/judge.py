"""LLM-as-judge for audit reports, plus the hand-labelling and agreement workflow.

    uv run seo-judge label   # 1. list every finding in evals/labels.json for you to label
    uv run seo-judge run     # 2. the LLM judge labels the same findings (cached per case)
    uv run seo-judge agree   # 3. how often does the judge agree with you?

The judge only rules on what code can't check (evals.py handles grounding and lengths):
  - reasoning_sound: does the issue follow from the evidence, and would the fix resolve it?
  - priority: is the impact rating right for this site, relative to its other findings?
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from seo_audit_agent.evals import load_runs
from seo_audit_agent.report import AuditRun, priority_label

JUDGE_MODEL = "claude-opus-5-5"
CASES = "evals/cases"
LABELS = Path("evals/labels.json")
JUDGEMENTS = Path("evals/judgements")


# ---------- What the judge must return (critique BEFORE verdict: reason first, then decide) ----------

class FindingVerdict(BaseModel):
    finding: int = Field(description="The finding number, as given")
    reasoning_critique: str = Field(description="Check the issue against the cited evidence and the full tool "
                                                "outputs, then check whether the fix would resolve that issue")
    reasoning_sound: bool
    priority_critique: str = Field(description="Weigh this finding's impact rating against the tool data "
                                               "(e.g. real-user field data) and the other findings")
    priority: Literal["ok", "too_high", "too_low"]


class Judgement(BaseModel):
    verdicts: list[FindingVerdict]


JUDGE_TOOL = {"name": "submit_verdicts", "description": "Submit one verdict per finding.",
              "input_schema": Judgement.model_json_schema()}

# Examples are invented (shop.test), never taken from the labelled cases: examples copied from
# the test set would inflate agreement, because the judge would just recognise the answers.
RUBRIC = """You review findings from an automated SEO audit. You get the raw tool outputs the audit was
based on and the findings it reported. For EACH finding, decide two things.

1. reasoning_sound (true/false). True only if ALL hold:
   a. The issue follows from the cited evidence and the tool outputs (read the whole output, not just
      the cited field: a cited value can be real but misread).
   b. The fix would plausibly resolve that issue on this site.
   c. The fix doesn't depend on facts the tool outputs don't show. Naming a file, script or image as
      the cause is only sound if a tool output lists it for that problem.
   d. The fix doesn't point the site owner somewhere the tool data shows is broken or already done.
   Example (unsound, c): issue "render-blocking requests cost 900 ms"; fix "defer analytics.js";
   but top_issues lists only app.css under render-blocking. The culprit is invented.
   Example (unsound, d): issue "link to http://shop.test/sale 404s"; fix "point it at
   https://www.shop.test/sale", but check_links shows that URL is the one returning 404.
   Example (sound): issue "page weighs 6 MB, hero.gif is 4.8 MB (heaviest_resources[0])";
   fix "replace hero.gif with an MP4 video".

2. priority ("ok", "too_high", "too_low"): is the impact rating right for THIS site?
   Consider how much fixing it would change search visibility or real users' experience,
   compared with the other findings in the same report. Lab-only speed problems matter less when
   field_data shows real users are all FAST. Broken internal links and noindex mistakes matter more
   than cosmetic tag lengths. Missing og:image matters for sharing, little for ranking.

Be strict and specific. Write the critique first, then decide."""


def render_case(run: AuditRun) -> str:
    """What the judge sees: all tool outputs, then each finding with its rating and evidence."""
    lines = ["TOOL OUTPUTS:"]
    for t in run.tool_outputs:
        lines.append(f"--- {t['tool']} ---\n{json.dumps(t['output'], ensure_ascii=False)}")
    lines += ["", f"AUDITED URL: {run.report.url}", "", "FINDINGS:"]
    for i, f in enumerate(run.report.findings, 1):
        cited = "; ".join(f"{e.tool}.{e.field} = {e.value}" for e in f.evidence)
        lines.append(f"[{i}] ({f.category}) impact={f.impact} effort={f.effort} priority={priority_label(f)}\n"
                     f"    issue: {f.issue}\n    fix: {f.fix}\n    evidence: {cited}")
    return "\n".join(lines)


def judge_run(run: AuditRun, client: Anthropic, model: str = JUDGE_MODEL) -> Judgement:
    resp = client.messages.create(
        model=model, max_tokens=8192, system=RUBRIC, tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_verdicts"},  # structured verdicts, always
        messages=[{"role": "user", "content": render_case(run)}],
    )
    block = next(b for b in resp.content if b.type == "tool_use")
    judgement = Judgement.model_validate(block.input)
    expected = set(range(1, len(run.report.findings) + 1))
    if {v.finding for v in judgement.verdicts} != expected:
        raise ValueError(f"judge returned verdicts for {sorted(v.finding for v in judgement.verdicts)}, "
                         f"expected {sorted(expected)}")
    return judgement


# ---------- 1. label: one entry per finding, for a human ----------

def build_labels(runs: list[tuple[Path, AuditRun]], existing: dict) -> dict:
    """Add new findings; keep labels already given (matched by case + finding + issue text)."""
    labels = {}
    for path, run in runs:
        if run.report is None:
            continue
        old = {(e["finding"], e["issue"]): e for e in existing.get(path.stem, [])}
        entries = []
        for i, f in enumerate(run.report.findings, 1):
            prev = old.get((i, f.issue), {})
            entries.append({
                "finding": i, "priority_label": priority_label(f), "impact": f.impact, "effort": f.effort,
                "issue": f.issue, "fix": f.fix,
                "evidence": [f"{e.tool}.{e.field} = {e.value}" for e in f.evidence],
                "reasoning_sound": prev.get("reasoning_sound"),   # you fill in: true / false
                "priority": prev.get("priority"),                 # you fill in: "ok" / "too_high" / "too_low"
                "note": prev.get("note", ""),
            })
        labels[path.stem] = entries
    return labels


# ---------- 3. agree: judge vs human ----------

def cohen_kappa(pairs: list[tuple]) -> float:
    """Agreement corrected for chance: 1 = perfect, 0 = no better than guessing, <0 = worse."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(h == j for h, j in pairs) / n
    classes = {c for pair in pairs for c in pair}
    expected = sum((sum(h == c for h, _ in pairs) / n) * (sum(j == c for _, j in pairs) / n) for c in classes)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def agreement_report(labels: dict, judgements: dict) -> dict:
    """Compare judge verdicts with human labels on every finding the human has labelled."""
    reasoning, priority, disagreements = [], [], []
    for case, entries in labels.items():
        verdicts = {v["finding"]: v for v in judgements.get(case, {}).get("verdicts", [])}
        for e in entries:
            v = verdicts.get(e["finding"])
            if v is None:
                continue
            if e["reasoning_sound"] is not None:
                reasoning.append((e["reasoning_sound"], v["reasoning_sound"]))
                if e["reasoning_sound"] != v["reasoning_sound"]:
                    disagreements.append((case, e, v, "reasoning"))
            if e["priority"] is not None:
                priority.append((e["priority"], v["priority"]))
                if e["priority"] != v["priority"]:
                    disagreements.append((case, e, v, "priority"))

    # "Catching unsound reasoning" is the job, so measure it directly (unsound = positive class).
    caught = sum(1 for h, j in reasoning if not h and not j)
    flagged = sum(1 for _, j in reasoning if not j)
    actual = sum(1 for h, _ in reasoning if not h)
    return {
        "reasoning_n": len(reasoning),
        "reasoning_accuracy": sum(h == j for h, j in reasoning) / len(reasoning) if reasoning else None,
        "reasoning_kappa": cohen_kappa(reasoning),
        "unsound_precision": caught / flagged if flagged else None,  # when it says unsound, is it right?
        "unsound_recall": caught / actual if actual else None,       # of the truly unsound, how many caught?
        "priority_n": len(priority),
        "priority_accuracy": sum(h == j for h, j in priority) / len(priority) if priority else None,
        "priority_kappa": cohen_kappa(priority),
        "disagreements": disagreements,
    }


def fmt(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.2f}"


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(prog="seo-judge", description="LLM judge for SEO audit reports")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("label", help="Write/refresh evals/labels.json for hand labelling")
    run_p = sub.add_parser("run", help="Run the LLM judge on every case (cached)")
    run_p.add_argument("--force", action="store_true", help="Re-judge cases that already have a judgement")
    run_p.add_argument("--model", default=JUDGE_MODEL)
    run_p.add_argument("--scores", action="store_true", help="Also attach scores to each case's Langfuse trace")
    sub.add_parser("agree", help="Compare judge verdicts with your labels")
    for p in sub.choices.values():
        p.add_argument("--cases", default=CASES)
    args = parser.parse_args()
    runs = load_runs(args.cases)

    if args.command == "label":
        existing = json.loads(LABELS.read_text()) if LABELS.exists() else {}
        labels = build_labels(runs, existing)
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
        total = sum(len(v) for v in labels.values())
        done = sum(e["reasoning_sound"] is not None for v in labels.values() for e in v)
        print(f"{LABELS}: {total} findings across {len(labels)} cases, {done} labelled so far.")

    elif args.command == "run":
        load_dotenv()
        client = Anthropic(timeout=180, max_retries=3)
        JUDGEMENTS.mkdir(parents=True, exist_ok=True)
        langfuse = None
        if args.scores:
            from langfuse import get_client
            langfuse = get_client()
        for path, run in runs:
            if run.report is None:
                continue
            out = JUDGEMENTS / f"{path.stem}.json"
            if out.exists() and not args.force:
                print(f"  cached  {path.stem}")
                judgement = Judgement.model_validate_json(out.read_text())
            else:
                judgement = judge_run(run, client, args.model)
                out.write_text(judgement.model_dump_json(indent=2) + "\n")
                print(f"  judged  {path.stem}")
            sound = sum(v.reasoning_sound for v in judgement.verdicts) / len(judgement.verdicts)
            print(f"          reasoning sound: {sound:.0%}  |  unsound: "
                  f"{[v.finding for v in judgement.verdicts if not v.reasoning_sound] or 'none'}")
            if langfuse and run.trace_id:
                langfuse.create_score(trace_id=run.trace_id, name="judge_reasoning_sound_rate", value=sound,
                                      data_type="NUMERIC", comment=f"{args.model} on {len(judgement.verdicts)} findings")
        if langfuse:
            langfuse.flush()

    elif args.command == "agree":
        if not LABELS.exists():
            sys.exit("No labels yet: run `seo-judge label` and fill in evals/labels.json first.")
        labels = json.loads(LABELS.read_text())
        judgements = {p.stem: json.loads(p.read_text()) for p in JUDGEMENTS.glob("*.json")}
        r = agreement_report(labels, judgements)
        print(f"\nReasoning  (n={r['reasoning_n']}): accuracy {fmt(r['reasoning_accuracy'])}, "
              f"kappa {fmt(r['reasoning_kappa'])}")
        print(f"  catching unsound findings: precision {fmt(r['unsound_precision'])}, "
              f"recall {fmt(r['unsound_recall'])}")
        print(f"Priority   (n={r['priority_n']}): accuracy {fmt(r['priority_accuracy'])}, "
              f"kappa {fmt(r['priority_kappa'])}")
        if r["disagreements"]:
            print(f"\nDisagreements ({len(r['disagreements'])}): read these, then decide who was right")
        for case, e, v, kind in r["disagreements"]:
            human, judge = (e["reasoning_sound"], v["reasoning_sound"]) if kind == "reasoning" \
                else (e["priority"], v["priority"])
            critique = v["reasoning_critique"] if kind == "reasoning" else v["priority_critique"]
            print(f"\n- {case} #{e['finding']} [{kind}] you: {human}  judge: {judge}\n"
                  f"  issue: {e['issue']}\n  judge said: {critique}")


if __name__ == "__main__":
    main()
