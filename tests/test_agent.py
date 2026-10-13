import copy
import importlib

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def _msg(content, stop_reason):
    return Message(id="msg_1", type="message", role="assistant", model="test", content=content,
                   stop_reason=stop_reason, stop_sequence=None,
                   usage=Usage(input_tokens=100, output_tokens=20))


def test_agent_loop_calls_tool_then_answers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")       # no real key needed in CI
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)  # tracing becomes a no-op
    agent = importlib.import_module("seo_audit_agent.agent")

    replies = iter([
        _msg([ToolUseBlock(id="tu_1", type="tool_use", name="fetch_title",
                           input={"url": "https://shop.test/"})], "tool_use"),
        _msg([TextBlock(type="text", text="The title is fine.")], "end_turn"),
    ])
    calls = []
    monkeypatch.setattr(agent.client.messages, "create", lambda **kw: calls.append(copy.deepcopy(kw["messages"])) or next(replies))
    monkeypatch.setitem(agent.TOOL_FUNCS, "fetch_title", lambda url: {"title": "Shop", "url": url})

    answer = agent.run_agent("What's the title of https://shop.test/?")

    assert answer == "The title is fine."
    assert len(calls) == 2
    tool_result = calls[1][-1]["content"][0]   # what we sent back to Claude on turn 2
    assert tool_result["tool_use_id"] == "tu_1"
    assert '"title": "Shop"' in tool_result["content"]


def test_crashing_tool_returns_error_data(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    agent = importlib.import_module("seo_audit_agent.agent")

    def boom(url):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(agent.TOOL_FUNCS, "fetch_title", boom)
    assert agent.call_tool("fetch_title", {"url": "x"}) == {"error": "RuntimeError: kaboom"}
