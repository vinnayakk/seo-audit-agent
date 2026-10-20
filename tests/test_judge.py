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


class FakeClient:
    def __init__(self, verdicts):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._verdicts = verdicts

    def _create(self, **kw):
        self.calls.append(kw)
        block = SimpleNamespace(type="tool_use", input={"verdicts": self._verdicts})
        return SimpleNamespace(content=[block])


def test_judge_sees_full_tool_outputs_and_numbered_findings():
    client = FakeClient([verdict(1, False), verdict(2, True)])
    result = judge.judge_run(RUN, client)
    prompt = client.calls[0]["messages"][0]["content"]
    assert '"final_url": "https://www.shop.test/sale"' in prompt   # whole output, not just cited fields
    assert "[1] (links) impact=high effort=low priority=P1" in prompt
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_verdicts"}
    assert [v.reasoning_sound for v in result.verdicts] == [False, True]


def test_judge_must_cover_every_finding():
    with pytest.raises(ValueError, match="expected"):
        judge.judge_run(RUN, FakeClient([verdict(1, True)]))       # finding 2 missing


def test_critique_comes_before_verdict_in_schema():
    fields = list(judge.FindingVerdict.model_fields)
    assert fields.index("reasoning_critique") < fields.index("reasoning_sound")
    assert fields.index("priority_critique") < fields.index("priority")


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
        {"finding": 3, "issue": "c", "reasoning_sound": False, "priority": None},   # priority unlabelled
        {"finding": 4, "issue": "d", "reasoning_sound": None, "priority": None},    # not labelled at all
    ]}
    judgements = {"shop": {"verdicts": [verdict(1, False), verdict(2, False, "too_high"),
                                        verdict(3, True), verdict(4, False)]}}
    r = judge.agreement_report(labels, judgements)
    assert r["reasoning_n"] == 3 and r["priority_n"] == 2
    assert r["unsound_recall"] == 0.5      # caught #1, missed #3
    assert r["unsound_precision"] == 0.5   # flagged #1 (right) and #2 (wrong)
    assert r["priority_accuracy"] == 1.0
    assert len(r["disagreements"]) == 2
