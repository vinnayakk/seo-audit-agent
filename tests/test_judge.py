import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from seo_audit_agent import judge
from seo_audit_agent.report import AuditRun

RUN = AuditRun(question="q", trace_id="t1", tool_outputs=[
    {"tool": "check_links", "input": {}, "output": {"broken": [
        {"url": "http://shop.test/sale", "final_url": "https://www.shop.test/sale", "status": 404}]}},
], report={"url": "https://shop.test/", "summary": "s", "findings": [
    {"category": "links", "issue": "Sale link 404s", "fix": "Point it at https://www.shop.test/sale",
     "impact": "high", "effort": "low",
     "evidence": [{"tool": "check_links", "field": "broken[0].status", "value": "404"}]},
    {"category": "meta_tags", "issue": "No og:image", "fix": "Add og:image",
     "impact": "low", "effort": "low",
     "evidence": [{"tool": "check_links", "field": "broken[0].url", "value": "http://shop.test/sale"}]},
]})


def verdict(n, sound, priority="ok"):
    return {"finding": n, "reasoning_critique": "c", "reasoning_sound": sound,
            "priority_critique": "p", "priority": priority}


class FakeAnthropic:
    def __init__(self, verdicts, refuse=False):
        self.calls, self._verdicts, self._refuse = [], verdicts, refuse
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        parsed = None if self._refuse else kw["output_format"].model_validate({"verdicts": self._verdicts})
        return SimpleNamespace(parsed_output=parsed, usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


class FakeOpenAI:
    def __init__(self, verdicts, refuse=False):
        self.calls, self._verdicts, self._refuse = [], verdicts, refuse
        self.responses = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        parsed = None if self._refuse else kw["text_format"].model_validate({"verdicts": self._verdicts})
        return SimpleNamespace(output_parsed=parsed, usage=SimpleNamespace(input_tokens=900, output_tokens=150))


# ---------- providers ----------

def test_provider_routing():
    assert judge.provider_for("claude-sonnet-5-5") == "anthropic"
    assert judge.provider_for("claude-fable-5-1") == "anthropic"
    assert judge.provider_for("gpt-6.1-sol") == "openai"
    with pytest.raises(ValueError):
        judge.provider_for("llama-4")


def test_anthropic_judge_sees_full_tool_outputs_and_records_usage():
    client = FakeAnthropic([verdict(1, False), verdict(2, True)])
    record = judge.judge_run(RUN, client, "claude-sonnet-5-5", case="shop")
    prompt = client.calls[0]["messages"][0]["content"]
    assert '"final_url": "https://www.shop.test/sale"' in prompt   # whole output, not just cited fields
    assert "[1] (links) impact=high effort=low priority=P1" in prompt
    assert client.calls[0]["output_format"] is judge.Judgement     # structured outputs, no forced tool
    assert "tool_choice" not in client.calls[0]
    assert [v.reasoning_sound for v in record.verdicts] == [False, True]
    assert (record.model, record.case, record.input_tokens, record.output_tokens) == \
           ("claude-sonnet-5-5", "shop", 1000, 200)


def test_openai_judge_gets_same_rubric_and_prompt():
    client = FakeOpenAI([verdict(1, False), verdict(2, True)])
    record = judge.judge_run(RUN, client, "gpt-6.1-sol", case="shop")
    call = client.calls[0]
    assert call["instructions"] == judge.RUBRIC
    assert call["input"] == judge.render_case(RUN)             # identical prompt across providers
    assert call["text_format"] is judge.Judgement
    assert record.input_tokens == 900 and record.verdicts[0].reasoning_sound is False


def test_anthropic_empty_output_is_an_error():
    with pytest.raises(RuntimeError, match="no parsed output"):
        judge.judge_run(RUN, FakeAnthropic([], refuse=True), "claude-opus-5-5")


def test_openai_refusal_is_an_error():
    with pytest.raises(RuntimeError, match="no parsed output"):
        judge.judge_run(RUN, FakeOpenAI([], refuse=True), "gpt-6-luna")


def test_judge_must_cover_every_finding():
    with pytest.raises(ValueError, match="expected"):
        judge.judge_run(RUN, FakeAnthropic([verdict(1, True)]), "claude-sonnet-5-5")


def test_critique_comes_before_verdict_in_schema():
    fields = list(judge.FindingVerdict.model_fields)
    assert fields.index("reasoning_critique") < fields.index("reasoning_sound")
    assert fields.index("priority_critique") < fields.index("priority")


# ---------- labels and agreement ----------

def test_build_labels_keeps_existing_labels():
    runs = [(Path("shop.json"), RUN)]
    first = judge.build_labels(runs, {})
    assert first["shop"][0]["reasoning_sound"] is None
    first["shop"][0].update(reasoning_sound=False, note="points at the 404")
    again = judge.build_labels(runs, first)
    assert again["shop"][0]["reasoning_sound"] is False
    assert again["shop"][0]["note"] == "points at the 404"


def test_cohen_kappa():
    assert judge.cohen_kappa([(True, True), (False, False)]) == 1.0
    assert judge.cohen_kappa([(True, False), (False, True)]) == -1.0
    # agrees 3/4 times, but "always True" would agree often too: kappa corrects for that
    assert judge.cohen_kappa([(True, True), (True, True), (True, True), (False, True)]) == 0.0


def test_agreement_report_measures_catching_unsound_findings():
    labels = {"shop": [
        {"finding": 1, "issue": "a", "reasoning_sound": False, "priority": "ok"},
        {"finding": 2, "issue": "b", "reasoning_sound": True, "priority": "too_high"},
        {"finding": 3, "issue": "c", "reasoning_sound": False, "priority": None},
        {"finding": 4, "issue": "d", "reasoning_sound": None, "priority": None},
    ]}
    judgements = {"shop": {"verdicts": [verdict(1, False), verdict(2, False, "too_high"),
                                        verdict(3, True), verdict(4, False)]}}
    r = judge.agreement_report(labels, judgements)
    assert r["reasoning_n"] == 3 and r["priority_n"] == 2
    assert r["unsound_recall"] == 0.5 and r["unsound_precision"] == 0.5
    assert r["priority_accuracy"] == 1.0
    assert len(r["disagreements"]) == 2


def test_cost_summary_uses_price_table():
    j = {"a": {"input_tokens": 1_000_000, "output_tokens": 100_000, "seconds": 10}}
    assert judge.cost_summary("gpt-6-luna", j)["cost"] == pytest.approx(0.10 + 0.05)
    assert judge.cost_summary("some-new-model", j)["cost"] is None        # unknown price -> n/a


# ---------- storage and CLI ----------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Run the CLI inside a temp folder with one saved case."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(judge, "JUDGEMENTS", Path("evals/judgements"))
    monkeypatch.setattr(judge, "LABELS", Path("evals/labels.json"))
    (tmp_path / "evals/cases").mkdir(parents=True)
    (tmp_path / "evals/cases/shop.json").write_text(RUN.model_dump_json())
    return tmp_path


def cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["seo-judge", *argv])
    judge.main()


def test_each_run_gets_its_own_folder_per_model(workspace, monkeypatch, capsys):
    fakes = {"anthropic": FakeAnthropic([verdict(1, False), verdict(2, True)]),
             "openai": FakeOpenAI([verdict(1, True), verdict(2, True)])}
    monkeypatch.setattr(judge, "make_client", lambda provider: fakes[provider])
    stamps = iter(["20261020-100000", "20261020-110000"])
    monkeypatch.setattr(judge, "datetime", SimpleNamespace(now=lambda: SimpleNamespace(
        strftime=lambda fmt: next(stamps))))

    cli(monkeypatch, "run", "--model", "claude-sonnet-5-5", "--model", "gpt-6.1-sol")
    cli(monkeypatch, "run", "--model", "claude-sonnet-5-5")              # second run: new folder

    sonnet_runs = [p.name for p in judge.list_runs("claude-sonnet-5-5")]
    assert sonnet_runs == ["20261020-100000", "20261020-110000"]         # nothing overwritten
    saved = json.loads((workspace / "evals/judgements/gpt-6.1-sol/20261020-100000/shop.json").read_text())
    assert saved["model"] == "gpt-6.1-sol" and saved["input_tokens"] == 900
    assert judge.judged_models() == ["claude-sonnet-5-5", "gpt-6.1-sol"]

    labels = {"shop": [{"finding": 1, "issue": "Sale link 404s", "reasoning_sound": False, "priority": "ok"},
                       {"finding": 2, "issue": "No og:image", "reasoning_sound": True, "priority": "ok"}]}
    (workspace / "evals/labels.json").write_text(json.dumps(labels))
    capsys.readouterr()
    cli(monkeypatch, "compare")
    out = capsys.readouterr().out
    assert "claude-sonnet-5-5" in out and "gpt-6.1-sol" in out
    assert "Judge vs judge" in out
    assert "claude-sonnet-5-5: 1.00" in out          # two identical sonnet runs agree with themselves
    assert "gpt-6.1-sol: only 1 run" in out


def test_failed_run_leaves_no_empty_folder(workspace, monkeypatch, capsys):
    monkeypatch.setattr(judge, "make_client", lambda provider: FakeAnthropic([], refuse=True))
    cli(monkeypatch, "run", "--model", "claude-opus-5-5")
    assert "0/1 cases judged" in capsys.readouterr().out
    assert judge.list_runs("claude-opus-5-5") == []
    assert not any((workspace / "evals/judgements/claude-opus-5-5").iterdir())


def test_flat_judgements_from_v1_are_moved_not_lost(workspace, monkeypatch):
    flat = workspace / "evals/judgements"
    flat.mkdir(parents=True)
    (flat / "shop.json").write_text(json.dumps({"verdicts": [verdict(1, True), verdict(2, True)]}))
    judge.migrate_flat_judgements()
    assert not (flat / "shop.json").exists()
    assert (flat / judge.DEFAULT_MODEL / "first-run/shop.json").exists()
