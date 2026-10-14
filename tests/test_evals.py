from seo_audit_agent.evals import evaluate
from seo_audit_agent.report import AuditRun

TOOL_OUTPUTS = [
    {"tool": "fetch_page", "input": {}, "output": {"status": 200}},
    {"tool": "check_meta_tags", "input": {}, "output": {
        "title": "Home", "title_length": 4, "description": None, "description_length": 0,
        "robots": None, "x_robots_tag": None, "open_graph": {"image": None}}},
    {"tool": "check_links", "input": {}, "output": {"broken": []}},
    {"tool": "check_speed", "input": {}, "output": {
        "lab_metrics": {"LCP": "4.2 s"}, "heaviest_resources": [{"kb": 5442}]}},
]


def make_run(findings, tool_outputs=TOOL_OUTPUTS, not_checked=()):
    return AuditRun(question="q", tool_outputs=tool_outputs, report={
        "url": "https://shop.test/", "summary": "s", "findings": findings, "not_checked": list(not_checked)})


def finding(evidence, fix="Do the thing", category="meta_tags"):
    return {"category": category, "issue": "i", "fix": fix, "impact": "medium", "effort": "low",
            "evidence": [dict(zip(("tool", "field", "value"), e)) for e in evidence]}


def results(run):
    return {r.name: r for r in evaluate(run)}


def test_grounded_evidence_passes_including_null_list_index_and_nbsp():
    run = make_run([finding([("check_meta_tags", "open_graph.image", "null"),
                             ("check_speed", "heaviest_resources[0].kb", "5442"),
                             ("check_speed", "lab_metrics.LCP", "4.2 s")])])  # plain space vs  
    assert results(run)["evidence_grounded"].passed


def test_wrong_value_missing_field_and_unrun_tool_all_fail():
    run = make_run([finding([("check_speed", "heaviest_resources[0].kb", "5000"),
                             ("check_meta_tags", "h1", "x"),
                             ("fetch_page", "status", "200")])],
                   tool_outputs=TOOL_OUTPUTS[1:])  # fetch_page never ran
    failures = results(run)["evidence_grounded"].failures
    assert len(failures) == 3
    assert "cited '5000' but the tool returned '5442'" in failures[0]
    assert "does not exist" in failures[1]
    assert "never run" in failures[2]


def test_long_title_suggestion_fails():
    long_title = "Vinnayakk | Applied AI Engineer – Generative AI for Marketing & Creative Teams"  # 78
    run = make_run([finding([("check_meta_tags", "title_length", "4")], fix=f"Use '{long_title}'")])
    assert "78 chars" in results(run)["suggestion_lengths"].failures[0]


def test_description_suggestion_must_be_70_to_160():
    run = make_run([finding([("check_meta_tags", "description_length", "0")],
                            fix="Use 'Short description here'")])
    assert not results(run)["suggestion_lengths"].passed


def test_failed_tool_must_be_disclosed():
    outputs = TOOL_OUTPUTS[:3] + [{"tool": "check_speed", "input": {}, "output": {"error": "429"}}]
    run = make_run([finding([("check_meta_tags", "title_length", "4")])], tool_outputs=outputs)
    assert not results(run)["failed_tools_disclosed"].passed
    run = make_run([finding([("check_meta_tags", "title_length", "4")])], tool_outputs=outputs,
                   not_checked=["mobile speed: PageSpeed API failed"])
    assert results(run)["failed_tools_disclosed"].passed


def test_missing_critical_issues_fail():
    outputs = [
        {"tool": "fetch_page", "input": {}, "output": {"status": 200}},
        {"tool": "check_meta_tags", "input": {}, "output": {"title": "Home", "robots": "noindex, follow"}},
        {"tool": "check_links", "input": {}, "output": {"broken": [{"url": "https://shop.test/gone"}]}},
    ]
    run = make_run([finding([("check_meta_tags", "title", "Home")])], tool_outputs=outputs)
    failures = results(run)["critical_issues_reported"].failures
    assert any("noindex" in f for f in failures)
    assert any("broken link" in f for f in failures)
