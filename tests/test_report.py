import pytest
from pydantic import ValidationError

from seo_audit_agent.report import (SUBMIT_TOOL, AuditReport, Finding, priority_label,
                                    prioritised, render_markdown)


def finding(issue, impact, effort):
    return Finding(category="speed", issue=issue, fix=f"Fix {issue}", impact=impact, effort=effort,
                   evidence=[{"tool": "check_speed", "field": "page_weight_kb", "value": "7365"}])


def test_priority_labels():
    assert priority_label(finding("a", "high", "low")) == "P1"
    assert priority_label(finding("b", "high", "high")) == "P2"
    assert priority_label(finding("c", "medium", "medium")) == "P2"
    assert priority_label(finding("d", "low", "low")) == "P3"


def test_prioritised_puts_quick_high_impact_wins_first():
    report = AuditReport(url="u", summary="s", findings=[
        finding("small", "low", "low"),
        finding("big-hard", "high", "high"),
        finding("big-easy", "high", "low"),
    ])
    assert [f.issue for f in prioritised(report)] == ["big-easy", "big-hard", "small"]


def test_evidence_is_required():
    with pytest.raises(ValidationError):
        Finding(category="speed", issue="x", fix="y", impact="high", effort="low", evidence=[])


def test_unknown_tool_in_evidence_is_rejected():
    with pytest.raises(ValidationError):
        Finding(category="speed", issue="x", fix="y", impact="high", effort="low",
                evidence=[{"tool": "made_up_tool", "field": "f", "value": "v"}])


def test_render_markdown_shows_priority_and_evidence():
    report = AuditReport(url="https://shop.test/", summary="Heavy page.",
                         findings=[finding("GIF | heavy", "high", "low")], not_checked=["desktop speed"])
    md = render_markdown(report)
    assert "| 1 | P1 | GIF \\| heavy |" in md            # pipes escaped so the table holds
    assert "`check_speed.page_weight_kb` = 7365" in md
    assert "- desktop speed" in md


def test_submit_tool_schema_comes_from_pydantic():
    props = SUBMIT_TOOL["input_schema"]["properties"]
    assert {"url", "summary", "findings", "not_checked"} <= props.keys()
