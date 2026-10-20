import copy
import importlib
from types import SimpleNamespace

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

VALID_REPORT = {
    "url": "https://shop.test/",
    "summary": "Reachable, but the title is too short.",
    "findings": [{
        "category": "meta_tags", "issue": "Title is only 4 characters",
        "evidence": [{"tool": "check_meta_tags", "field": "title_length", "value": "4"}],
        "fix": "Rewrite the title as 'Handmade Leather Wallets | Shop'",
        "impact": "medium", "effort": "low",
    }],
    "not_checked": ["mobile speed"],
}


def _msg(content, stop_reason):
    return Message(id="msg_1", type="message", role="assistant", model="test", content=content,
                   stop_reason=stop_reason, stop_sequence=None,
                   usage=Usage(input_tokens=100, output_tokens=20))


def _tool(name, input, id="tu_1"):
    return ToolUseBlock(id=id, type="tool_use", name=name, input=input)


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")  # no real key needed in CI
    module = importlib.import_module("seo_audit_agent.agent")
    monkeypatch.setitem(module.TOOL_FUNCS, "check_meta_tags", lambda url: {"title": "Shop", "title_length": 4})
    return module


def fake_claude(monkeypatch, agent, replies, parsed=None):
    """Replace the API calls with scripted replies; record a copy of every request.
    `create` = normal tool-calling turns; `parse` = the structured-output 'report now' turn."""
    replies, calls = iter(replies), []

    def create(**kw):
        calls.append({"api": "create", **copy.deepcopy(kw)})
        return next(replies)

    def parse(**kw):
        calls.append({"api": "parse", **copy.deepcopy({k: v for k, v in kw.items() if k != "output_format"}),
                      "output_format": kw["output_format"]})
        report = kw["output_format"].model_validate(parsed) if parsed else None
        return SimpleNamespace(parsed_output=report, usage=Usage(input_tokens=100, output_tokens=50))

    monkeypatch.setattr(agent.client.messages, "create", create)
    monkeypatch.setattr(agent.client.messages, "parse", parse)
    return calls


def assert_no_forced_tools(calls):
    """Opus/Sonnet 5.5 and Fable 5.1 return 400 for tool_choice "tool"/"any": never send them."""
    assert all(c.get("tool_choice", {}).get("type") in ("auto", "none") for c in calls)


# ---------- free-text mode (-q questions) ----------

def test_free_text_calls_tool_then_answers(monkeypatch, agent):
    calls = fake_claude(monkeypatch, agent, [
        _msg([_tool("check_meta_tags", {"url": "https://shop.test/"})], "tool_use"),
        _msg([TextBlock(type="text", text="The title is short.")], "end_turn"),
    ])
    run = agent.run_agent("Is the title OK? https://shop.test/")

    assert run.text == "The title is short."
    assert run.report is None
    assert run.tool_outputs[0]["output"]["title_length"] == 4
    tool_result = calls[1]["messages"][-1]["content"][0]   # what we sent back to Claude on turn 2
    assert tool_result["tool_use_id"] == "tu_1"
    assert all(t["name"] != "submit_report" for t in calls[0]["tools"])  # no submit tool in -q mode


def test_free_text_step_limit_forces_an_answer(monkeypatch, agent):
    calls = fake_claude(monkeypatch, agent, [
        _msg([_tool("check_meta_tags", {"url": "https://shop.test/"})], "tool_use"),
        _msg([TextBlock(type="text", text="Partial answer.")], "end_turn"),
    ])
    run = agent.run_agent("Audit https://shop.test/", max_turns=2)

    assert run.text == "Partial answer."
    assert calls[0]["tool_choice"] == {"type": "auto"}
    assert calls[1]["tool_choice"] == {"type": "none"}
    last_user_msg = calls[1]["messages"][-1]["content"]
    assert last_user_msg[0]["type"] == "tool_result"              # tool results must come first
    assert last_user_msg[-1]["text"] == agent.STEP_LIMIT_NOTE     # then the wrap-up instruction


# ---------- structured mode (full audits) ----------

def test_structured_returns_validated_report(monkeypatch, agent):
    calls = fake_claude(monkeypatch, agent, [
        _msg([_tool("check_meta_tags", {"url": "https://shop.test/"})], "tool_use"),
        _msg([_tool("submit_report", VALID_REPORT, id="tu_2")], "tool_use"),
    ])
    run = agent.run_agent("Audit https://shop.test/", structured=True)

    assert run.report.findings[0].evidence[0].tool == "check_meta_tags"
    assert run.tool_outputs[0]["tool"] == "check_meta_tags"       # saved alongside the report
    assert any(t["name"] == "submit_report" for t in calls[0]["tools"])
    assert_no_forced_tools(calls)


def test_invalid_report_is_sent_back_then_final_turn_uses_structured_output(monkeypatch, agent):
    no_evidence = copy.deepcopy(VALID_REPORT)
    no_evidence["findings"][0]["evidence"] = []                   # violates min_length=1
    calls = fake_claude(monkeypatch, agent, [
        _msg([_tool("submit_report", no_evidence)], "tool_use"),
    ], parsed=VALID_REPORT)
    run = agent.run_agent("Audit https://shop.test/", structured=True)

    rejection = calls[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] is True and "evidence" in rejection["content"]
    assert calls[1]["api"] == "parse" and calls[1]["tool_choice"] == {"type": "none"}
    assert calls[1]["output_format"] is agent.AuditReport
    assert run.report.findings[0].issue == "Title is only 4 characters"
    assert_no_forced_tools(calls)


def test_prose_instead_of_report_gets_a_structured_report_turn(monkeypatch, agent):
    calls = fake_claude(monkeypatch, agent, [
        _msg([TextBlock(type="text", text="Here is my audit in prose...")], "end_turn"),
    ], parsed=VALID_REPORT)
    run = agent.run_agent("Audit https://shop.test/", structured=True)

    assert calls[1]["messages"][-1]["content"] == agent.SUBMIT_NUDGE
    assert calls[1]["api"] == "parse"
    assert run.report is not None
    assert_no_forced_tools(calls)


def test_last_turn_reports_via_structured_output(monkeypatch, agent):
    calls = fake_claude(monkeypatch, agent, [
        _msg([_tool("check_meta_tags", {"url": "https://shop.test/"})], "tool_use"),
    ], parsed=VALID_REPORT)
    run = agent.run_agent("Audit https://shop.test/", max_turns=2, structured=True)

    assert [c["api"] for c in calls] == ["create", "parse"]
    assert calls[1]["messages"][-1]["content"][-1]["text"] == agent.STEP_LIMIT_NOTE
    assert run.tool_outputs[0]["tool"] == "check_meta_tags" and run.report is not None
    assert_no_forced_tools(calls)


def test_empty_structured_output_raises(monkeypatch, agent):
    fake_claude(monkeypatch, agent, [], parsed=None)              # parse returns no report
    with pytest.raises(agent.ReportError):
        agent.run_agent("Audit https://shop.test/", max_turns=1, structured=True)


def test_crashing_tool_returns_error_data(agent):
    def boom(url):
        raise RuntimeError("kaboom")

    agent.TOOL_FUNCS["fetch_title"] = boom
    assert agent.call_tool("fetch_title", {"url": "x"}) == {"error": "RuntimeError: kaboom"}
