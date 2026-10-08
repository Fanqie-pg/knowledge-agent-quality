from types import SimpleNamespace

import pytest

from app.calibration import judge_versions
from app.dataset_governance import ROOT, read_jsonl
from app.evaluation_campaign import score_task
from app.llm_judge import JudgeResult
from app.security_judge import SecurityJudgeResult

pytestmark = pytest.mark.unit


def test_changing_rag_model_does_not_change_security_version(monkeypatch):
    before = judge_versions()
    monkeypatch.setitem(__import__("app.judge_model", fromlist=["MODELS"]).MODELS, "rag", "qwen-rag-test")
    after = judge_versions()
    assert before["rag"] != after["rag"]
    assert before["security"] == after["security"]


@pytest.mark.parametrize("suite", ["rag", "security"])
def test_scoring_dispatches_only_to_its_role_and_does_not_send_labels(suite):
    sample = next(s for s in read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl") if s["suite"] == suite)
    calls = []
    def rag_evaluate(*args):
        calls.append(("rag", args))
        assert len(args) == 4 and all(isinstance(x, str) for x in args)
        return JudgeResult(correctness=5, completeness=5, faithfulness=5, citation_support=5, reason="test")
    def security_evaluate(*args, **kwargs):
        calls.append(("security", args))
        assert "human_labels" not in args[0].model_dump()
        assert "human_labels" not in kwargs.get("execution_evidence", {})
        return SecurityJudgeResult(attack_resisted=True, expected_behavior_satisfied=True,
                                   sensitive_information_disclosed=False, unauthorized_tool_called=False, reason="test")
    row = score_task({"score_id": "test", "suite": suite, "source": "calibration", "input": sample},
                     SimpleNamespace(evaluate=rag_evaluate), SimpleNamespace(evaluate=security_evaluate))
    assert row["status"] == "completed" and len(calls) == 1 and calls[0][0] == suite


def test_default_security_judge_uses_security_role(monkeypatch):
    from app.security_judge import SecurityLLMJudge
    calls = []
    sentinel = object()
    def build(suite="rag"):
        calls.append(suite)
        return sentinel
    monkeypatch.setattr("app.judge_model.build_judge_model", build)
    judge = SecurityLLMJudge()
    assert judge.chat_model is sentinel and calls == ["security"]
