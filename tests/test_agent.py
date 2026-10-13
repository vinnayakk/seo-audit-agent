import copy
import importlib

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


def fake_claude(monkeypatch, agent, replies):
    """Replace the API call with scripted replies; record a copy of every request."""
    replies, calls = iter(replies), []
    monkeypatch.setattr(agent.client.messages, "create",
                        lambda **kw: calls.append(copy.deepcopy(kw)) or next(replies))
    return calls


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


def test_invalid_report_is_sent_back_and_resubmitted(monkeypatch, agent):
    no_evidence = copy.deepcopy(VALID_REPORT)
    no_evidence["findings"][0]["evidence"] = []                   # violates min_length=1
    calls = fake_claude(monkeypatch, agent, [
        _msg([_tool("submit_report", no_evidence)], "tool_use"),
        _msg([_tool("submit_report", VALID_REPORT, id="tu_2")], "tool_use"),
    ])
    run = agent.run_agent("Audit https://shop.test/", structured=True)

    rejection = calls[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] is True and "evidence" in rejection["content"]
    assert calls[1]["tool_choice"] == {"type": "tool", "name": "submit_report"}  # forced resubmit
    assert run.report is not None


def test_prose_instead_of_report_gets_nudged(monkeypatch, agent):
    calls = fake_claude(monkeypatch, agent, [
        _msg([TextBlock(type="text", text="Here is my audit in prose...")], "end_turn"),
        _msg([_tool("submit_report", VALID_REPORT)], "tool_use"),
    ])
    run = agent.run_agent("Audit https://shop.test/", structured=True)

    assert calls[1]["messages"][-1]["content"] == agent.SUBMIT_NUDGE
    assert calls[1]["tool_choice"] == {"type": "tool", "name": "submit_report"}
    assert run.report is not None


def test_last_turn_forces_submit(monkeypatch, agent):
    calls = fake_claude(monkeypatch, agent, [
        _msg([_tool("check_meta_tags", {"url": "https://shop.test/"})], "tool_use"),
        _msg([_tool("submit_report", VALID_REPORT, id="tu_2")], "tool_use"),
    ])
    agent.run_agent("Audit https://shop.test/", max_turns=2, structured=True)
    assert calls[1]["tool_choice"] == {"type": "tool", "name": "submit_report"}


def test_no_valid_report_raises(monkeypatch, agent):
    fake_claude(monkeypatch, agent, [
        _msg([_tool("submit_report", {"url": "x"})], "tool_use"),  # missing required fields
    ])
    with pytest.raises(agent.ReportError):
        agent.run_agent("Audit https://shop.test/", max_turns=1, structured=True)


def test_crashing_tool_returns_error_data(agent):
    def boom(url):
        raise RuntimeError("kaboom")

    agent.TOOL_FUNCS["fetch_title"] = boom
    assert agent.call_tool("fetch_title", {"url": "x"}) == {"error": "RuntimeError: kaboom"}
