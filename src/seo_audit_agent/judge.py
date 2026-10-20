"""LLM-as-judge for audit reports, plus the hand-labelling and agreement workflow.

    uv run seo-judge label                                   # 1. findings to label in evals/labels.json
    uv run seo-judge run --model claude-opus-5-5 --model gpt-6.1-sol   # 2. judge with one or more models
    uv run seo-judge agree --model gpt-6.1-sol               # 3. one judge vs your labels, with disagreements
    uv run seo-judge compare                                 # 4. every judge side by side

Every `run` writes a NEW folder, evals/judgements/<model>/<timestamp>/, so nothing is ever overwritten.
Running the same model twice gives you its self-consistency (how often it disagrees with itself).

The judge only rules on what code can't check (evals.py handles grounding and lengths):
  - reasoning_sound: does the issue follow from the evidence, and would the fix resolve it?
  - priority: is the impact rating right for this site, relative to its other findings?
"""
import argparse
import json
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from seo_audit_agent.evals import load_runs
from seo_audit_agent.report import AuditRun, priority_label

DEFAULT_MODEL = "claude-opus-5-5"  # your choice in the last commit
CASES = "evals/cases"
LABELS = Path("evals/labels.json")
JUDGEMENTS = Path("evals/judgements")

# USD per 1M tokens (input, output), standard tier. Only used to estimate cost in `compare`.
# Check the providers' pricing pages; models missing here show cost as n/a.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),   # matches the costs Langfuse computed for your audit runs
    "claude-haiku-5-5": (0.10, 0.50),     # for prompts under 100k tokens (all of ours)
    "claude-fable-5-1": (10.00, 50.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),         # promotional price, listed "at least through November 21, 2026"
    "gpt-6-astra": (10.00, 50.00),
}


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


class JudgementRecord(Judgement):
    """A saved judgement: the verdicts plus what it cost to get them."""
    model: str
    case: str
    input_tokens: int
    output_tokens: int
    seconds: float


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


# ---------- Providers: same rubric and schema, different API shapes ----------

def provider_for(model: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4", "chat-")):
        return "openai"
    raise ValueError(f"Don't know which provider serves {model!r} (expected claude-* or gpt-*)")


def call_anthropic(client, model: str, prompt: str) -> tuple[dict, int, int]:
    """Anthropic: structured outputs (output_format). Opus 5.5, Sonnet 5.5 and Fable 5.1 reject
    forced tool calls, so a forced submit_verdicts tool no longer works on current models."""
    resp = client.messages.parse(model=model, max_tokens=8192, system=RUBRIC,
                                 messages=[{"role": "user", "content": prompt}], output_format=Judgement)
    if resp.parsed_output is None:
        raise RuntimeError(f"{model} returned no parsed output (refusal or empty response)")
    return resp.parsed_output.model_dump(), resp.usage.input_tokens, resp.usage.output_tokens


def call_openai(client, model: str, prompt: str) -> tuple[dict, int, int]:
    """OpenAI: structured output via text_format (the SDK turns the Pydantic model into a strict schema)."""
    resp = client.responses.parse(model=model, instructions=RUBRIC, input=prompt, text_format=Judgement)
    if resp.output_parsed is None:
        raise RuntimeError(f"{model} returned no parsed output (refusal or empty response)")
    return resp.output_parsed.model_dump(), resp.usage.input_tokens, resp.usage.output_tokens


def make_client(provider: str):
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(timeout=180, max_retries=3)
    from openai import OpenAI
    return OpenAI(timeout=180, max_retries=3)  # reads OPENAI_API_KEY


CALLERS = {"anthropic": call_anthropic, "openai": call_openai}


def judge_run(run: AuditRun, client, model: str = DEFAULT_MODEL, case: str = "") -> JudgementRecord:
    started = time.perf_counter()
    raw, tokens_in, tokens_out = CALLERS[provider_for(model)](client, model, render_case(run))
    judgement = Judgement.model_validate(raw)
    expected = set(range(1, len(run.report.findings) + 1))
    if {v.finding for v in judgement.verdicts} != expected:
        raise ValueError(f"{model} returned verdicts for {sorted(v.finding for v in judgement.verdicts)}, "
                         f"expected {sorted(expected)}")
    return JudgementRecord(verdicts=judgement.verdicts, model=model, case=case, input_tokens=tokens_in,
                           output_tokens=tokens_out, seconds=round(time.perf_counter() - started, 1))


# ---------- Storage: evals/judgements/<model>/<timestamp>/<case>.json, never overwritten ----------

def model_dir(model: str) -> Path:
    return JUDGEMENTS / model.replace("/", "_")


def list_runs(model: str) -> list[Path]:
    """Run folders for a model, oldest first (timestamps sort as text)."""
    d = model_dir(model)
    return sorted(p for p in d.iterdir() if p.is_dir() and any(p.glob("*.json"))) if d.exists() else []


def judged_models() -> list[str]:
    return sorted(p.name for p in JUDGEMENTS.iterdir() if p.is_dir() and list_runs(p.name)) \
        if JUDGEMENTS.exists() else []


def load_judgements(run_dir: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))}


def migrate_flat_judgements() -> None:
    """Judgements from the first version lived directly in evals/judgements/*.json. Move them into
    the per-model layout (they were made with the default model) instead of mixing layouts."""
    flat = sorted(JUDGEMENTS.glob("*.json")) if JUDGEMENTS.exists() else []
    if not flat:
        return
    dest = model_dir(DEFAULT_MODEL) / "first-run"
    dest.mkdir(parents=True, exist_ok=True)
    for p in flat:
        p.rename(dest / p.name)
    print(f"Moved {len(flat)} earlier judgement(s) to {dest}/")


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


# ---------- 3/4. agreement ----------

def cohen_kappa(pairs: list[tuple]) -> float:
    """Agreement corrected for chance: 1 = perfect, 0 = no better than guessing, <0 = worse."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / n
    classes = {c for pair in pairs for c in pair}
    expected = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in classes)
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


def verdict_pairs(a: dict, b: dict, key: str) -> list[tuple]:
    """Pair two judgement sets finding by finding (judge vs judge, or a judge vs itself)."""
    pairs = []
    for case in a.keys() & b.keys():
        va = {v["finding"]: v[key] for v in a[case]["verdicts"]}
        vb = {v["finding"]: v[key] for v in b[case]["verdicts"]}
        pairs += [(va[f], vb[f]) for f in va.keys() & vb.keys()]
    return pairs


def cost_summary(model: str, judgements: dict) -> dict:
    n = len(judgements) or 1
    tokens_in = sum(j.get("input_tokens", 0) for j in judgements.values())
    tokens_out = sum(j.get("output_tokens", 0) for j in judgements.values())
    price = PRICES.get(model)
    cost = (tokens_in * price[0] + tokens_out * price[1]) / 1e6 if price else None
    seconds = sum(j.get("seconds", 0) for j in judgements.values())
    return {"cost": cost, "cost_per_case": cost / n if cost is not None else None, "seconds_per_case": seconds / n}


def fmt(x, pct=False) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{x:.0%}" if pct else f"{x:.2f}"


def pick_run(model: str, run_name: str | None) -> Path:
    runs = list_runs(model)
    if not runs:
        sys.exit(f"No judgements for {model}: run `seo-judge run --model {model}` first.")
    if run_name is None:
        return runs[-1]
    match = [r for r in runs if r.name == run_name]
    if not match:
        sys.exit(f"No run {run_name!r} for {model}. Available: {', '.join(r.name for r in runs)}")
    return match[0]


# ---------- CLI ----------

def main() -> None:
    parser = argparse.ArgumentParser(prog="seo-judge", description="LLM judge for SEO audit reports")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("label", help="Write/refresh evals/labels.json for hand labelling")
    run_p = sub.add_parser("run", help="Judge every case with one or more models (new folder per run)")
    run_p.add_argument("--model", action="append", help=f"Repeat for several judges (default {DEFAULT_MODEL})")
    run_p.add_argument("--scores", action="store_true", help="Also attach scores to each case's Langfuse trace")
    agree_p = sub.add_parser("agree", help="One judge vs your labels, with every disagreement")
    agree_p.add_argument("--model", default=DEFAULT_MODEL)
    agree_p.add_argument("--run", help="Run folder name (default: the latest)")
    sub.add_parser("compare", help="All judges side by side: agreement, cost, consistency")
    for p in sub.choices.values():
        p.add_argument("--cases", default=CASES)
    args = parser.parse_args()
    load_dotenv()
    migrate_flat_judgements()

    if args.command == "label":
        runs = load_runs(args.cases)
        existing = json.loads(LABELS.read_text()) if LABELS.exists() else {}
        labels = build_labels(runs, existing)
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
        total = sum(len(v) for v in labels.values())
        done = sum(e["reasoning_sound"] is not None for v in labels.values() for e in v)
        print(f"{LABELS}: {total} findings across {len(labels)} cases, {done} labelled so far.")

    elif args.command == "run":
        runs = [(p, r) for p, r in load_runs(args.cases) if r.report is not None]
        langfuse = None
        if args.scores:
            from langfuse import get_client
            langfuse = get_client()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for model in args.model or [DEFAULT_MODEL]:
            client = make_client(provider_for(model))
            out_dir = model_dir(model) / stamp
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{model} -> {out_dir}/")
            done = 0
            for path, run in runs:
                try:
                    record = judge_run(run, client, model, case=path.stem)
                except Exception as e:  # one failed case shouldn't lose the rest of the run
                    print(f"  FAILED  {path.stem}: {type(e).__name__}: {e}")
                    continue
                (out_dir / f"{path.stem}.json").write_text(record.model_dump_json(indent=2) + "\n")
                done += 1
                unsound = [v.finding for v in record.verdicts if not v.reasoning_sound]
                print(f"  {path.stem}: unsound {unsound or 'none'}  ({record.seconds}s)")
                if langfuse and run.trace_id:
                    rate = 1 - len(unsound) / len(record.verdicts)
                    langfuse.create_score(trace_id=run.trace_id, name=f"judge_reasoning_sound_rate/{model}",
                                          value=rate, data_type="NUMERIC", comment=f"run {stamp}")
            print(f"  {done}/{len(runs)} cases judged")
            if done == 0:
                out_dir.rmdir()  # an empty run would only confuse `compare`
        if langfuse:
            langfuse.flush()

    elif args.command == "agree":
        if not LABELS.exists():
            sys.exit("No labels yet: run `seo-judge label` and fill in evals/labels.json first.")
        run_dir = pick_run(args.model, args.run)
        r = agreement_report(json.loads(LABELS.read_text()), load_judgements(run_dir))
        print(f"\n{args.model}  (run {run_dir.name})")
        print(f"Reasoning  (n={r['reasoning_n']}): accuracy {fmt(r['reasoning_accuracy'], True)}, "
              f"kappa {fmt(r['reasoning_kappa'])}")
        print(f"  catching unsound findings: precision {fmt(r['unsound_precision'], True)}, "
              f"recall {fmt(r['unsound_recall'], True)}")
        print(f"Priority   (n={r['priority_n']}): accuracy {fmt(r['priority_accuracy'], True)}, "
              f"kappa {fmt(r['priority_kappa'])}")
        if r["disagreements"]:
            print(f"\nDisagreements ({len(r['disagreements'])}): read these, then decide who was right")
        for case, e, v, kind in r["disagreements"]:
            human, judge = (e["reasoning_sound"], v["reasoning_sound"]) if kind == "reasoning" \
                else (e["priority"], v["priority"])
            critique = v["reasoning_critique"] if kind == "reasoning" else v["priority_critique"]
            print(f"\n- {case} #{e['finding']} [{kind}] you: {human}  judge: {judge}\n"
                  f"  issue: {e['issue']}\n  judge said: {critique}")

    elif args.command == "compare":
        models = judged_models()
        if not models:
            sys.exit("No judgements yet: run `seo-judge run --model ...` first.")
        labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
        latest = {m: load_judgements(list_runs(m)[-1]) for m in models}

        print("\nAgainst your labels (latest run of each judge)")
        header = f"{'judge':22} {'n':>3} {'reason κ':>9} {'unsound recall':>15} {'precision':>10} " \
                 f"{'priority κ':>11} {'$/case':>8} {'s/case':>7}"
        print(header + "\n" + "-" * len(header))
        for m in models:
            r = agreement_report(labels, latest[m])
            c = cost_summary(m, latest[m])
            cost = f"{c['cost_per_case']:.3f}" if c["cost_per_case"] is not None else "n/a"
            print(f"{m:22} {r['reasoning_n']:>3} {fmt(r['reasoning_kappa']):>9} "
                  f"{fmt(r['unsound_recall'], True):>15} {fmt(r['unsound_precision'], True):>10} "
                  f"{fmt(r['priority_kappa']):>11} {cost:>8} {c['seconds_per_case']:>7.1f}")

        if len(models) > 1:
            print("\nJudge vs judge (reasoning κ): where judges split, the finding or the rubric is ambiguous")
            for a, b in combinations(models, 2):
                print(f"  {a} vs {b}: {fmt(cohen_kappa(verdict_pairs(latest[a], latest[b], 'reasoning_sound')))}")

        print("\nSelf-consistency (same judge, its last two runs; reasoning κ)")
        for m in models:
            runs = list_runs(m)
            if len(runs) < 2:
                print(f"  {m}: only 1 run. Run it again to measure its noise floor.")
                continue
            pairs = verdict_pairs(load_judgements(runs[-2]), load_judgements(runs[-1]), "reasoning_sound")
            print(f"  {m}: {fmt(cohen_kappa(pairs))}  ({runs[-2].name} vs {runs[-1].name})")


if __name__ == "__main__":
    main()
