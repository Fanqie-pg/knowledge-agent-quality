from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from app.calibration import calibration_report, label_passed, validate_samples
from app.dataset_governance import ROOT, read_jsonl
from app.evaluation_records import CallBudget, CallBudgetExceeded, RunStore, attach_budget
from app.experiment import collect, load_cases, score_saved, summarize, experiment_config, reuse_unchanged
from app.quality_gate import inspect_junit
from app.rag_agent import RAGRunResult
from app.resilient_model import RetryingChatModel
from app.retriever import RetrievalResult
from app.tool_agent import ToolAgent
from app.tool_registry import build_tools
from app.tracing import capture_trace, emit, redact

pytestmark = pytest.mark.unit


def store_for(tmp_path, cases, trials=2):
    return RunStore(tmp_path, {"case_ids": [c.case_id for c, _ in cases], "answerable_case_ids": [c.case_id for c, _ in cases if c.answerable], "trials": trials})


class FakeRAG:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def run(self, question):
        self.calls += 1
        emit("generation", "started")
        if self.fail:
            raise TimeoutError("api_key=do-not-log")
        doc = Document(page_content="P1 工单：15 分钟内首次响应", metadata={"doc_id": "KB-PRODUCT-001", "chunk_id": "CHUNK"})
        return RAGRunResult(question, [RetrievalResult(doc, .9, 1)], doc.page_content,
                            AIMessage(content="15分钟。[来源：KB-PRODUCT-001/CHUNK]"), [], question)


def test_ab_resume_never_repeats_completed_generation_and_errors_remain_errors(tmp_path):
    cases = load_cases("dev")[:1]
    store = store_for(tmp_path, cases)
    agents = {"direct": FakeRAG(), "rewrite": FakeRAG(fail=True)}
    collect(store, cases, agents)
    collect(store, cases, agents)
    assert agents["direct"].calls == agents["rewrite"].calls == 2
    report = summarize(store)
    assert report["status"] == "incomplete"
    assert report["variants"]["rewrite"]["errors"] == 2
    assert report["variants"]["rewrite"]["deterministic_pass_rate"] == 0
    assert report["judge_status"] == "provisional_not_human_calibrated"
    assert "do-not-log" not in (tmp_path / "records.jsonl").read_text(encoding="utf-8")


def test_missing_jobs_are_in_metric_denominators(tmp_path):
    cases = load_cases("dev")[:1]
    store = store_for(tmp_path, cases, trials=1)
    report = summarize(store)
    assert report["missing"] == 2
    assert report["status"] == "incomplete"
    assert report["variants"]["direct"]["recall_at_5"] == 0


def test_run_config_change_requires_new_directory(tmp_path):
    RunStore(tmp_path, {"model": "a"})
    with pytest.raises(ValueError, match="配置"):
        RunStore(tmp_path, {"model": "b"})


def test_provider_budget_counts_every_retry_and_survives_restart(tmp_path):
    class Flaky:
        def invoke(self, prompt):
            raise TimeoutError("temporary")
    budget = CallBudget(tmp_path / "calls.jsonl", 2)
    model = attach_budget(RetryingChatModel(Flaky(), max_attempts=3), budget)
    with pytest.raises(CallBudgetExceeded):
        model.invoke("question")
    assert budget.used == 2
    with pytest.raises(CallBudgetExceeded):
        CallBudget(tmp_path / "calls.jsonl", 2).consume()


def test_multiple_model_wrappers_cannot_each_receive_the_full_shared_budget(tmp_path):
    a = CallBudget(tmp_path / "calls.jsonl", 1)
    b = CallBudget(tmp_path / "calls.jsonl", 1)
    a.consume()
    with pytest.raises(CallBudgetExceeded):
        b.consume()


def test_factory_models_share_the_configured_gate_budget(tmp_path, monkeypatch):
    import app.agent_factory as factory
    calls = []
    class Provider:
        def invoke(self, prompt):
            calls.append(prompt)
            return AIMessage(content="ok")
    monkeypatch.setattr(factory, "ChatOpenAI", lambda **kwargs: Provider())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("EVALUATION_MAX_CALLS", "1")
    monkeypatch.setenv("EVALUATION_CALL_LEDGER", str(tmp_path / "calls.jsonl"))
    a, b = factory.build_chat_model(), factory.build_chat_model()
    assert a.invoke("one").content == "ok"
    with pytest.raises(CallBudgetExceeded):
        b.invoke("two")
    assert calls == ["one"]


def test_scoring_errors_do_not_produce_a_successful_cli_exit(tmp_path, monkeypatch):
    import sys
    from app.experiment import main
    cases = load_cases("dev")[:1]
    store = store_for(tmp_path, cases, trials=1)
    collect(store, cases, {v: FakeRAG() for v in ("direct", "rewrite")})
    class BrokenJudge:
        def evaluate(self, *args):
            raise ValueError("malformed judge output")
    score_saved(store, BrokenJudge(), "bad-judge")
    monkeypatch.setattr(sys, "argv", ["experiment", "--output", str(tmp_path), "--score"])
    assert main() == 1


def test_saved_answer_scoring_does_not_regenerate_and_is_resumable(tmp_path):
    from app.llm_judge import JudgeResult
    cases = load_cases("dev")[:1]
    store = store_for(tmp_path, cases, trials=1)
    agents = {v: FakeRAG() for v in ("direct", "rewrite")}
    collect(store, cases, agents)
    class Judge:
        calls = 0
        def evaluate(self, *args):
            self.calls += 1
            return JudgeResult(correctness=5, completeness=5, faithfulness=5, citation_support=5, reason="correct")
    judge = Judge()
    score_saved(store, judge, "judge-v1")
    score_saved(store, judge, "judge-v1")
    assert judge.calls == 2 and all(a.calls == 1 for a in agents.values())
    with pytest.raises(ValueError, match="Judge版本"):
        score_saved(store, judge, "judge-v2")


def test_dataset_correction_reuses_only_identical_inputs_and_checks_snapshot(tmp_path):
    cases = load_cases("dev")[:1]
    config = experiment_config("dev", trials=1)
    source = RunStore(tmp_path / "source", config)
    collect(source, cases, {v: FakeRAG() for v in ("direct", "rewrite")})
    snapshot = tmp_path / "old-cases.jsonl"
    snapshot.write_bytes((ROOT / "data/evaluation/retrieval_cases.jsonl").read_bytes())
    target = RunStore(tmp_path / "target", config)
    reuse_unchanged(target, source.directory, snapshot, cases)
    assert len(target.rows()) == 2
    assert all(r["reused"] and r["origin_run_id"] == source.manifest["run_id"] for r in target.rows())
    changed = deepcopy(cases)
    changed[0][0].question += "补充问题"
    other = RunStore(tmp_path / "other", config)
    reuse_unchanged(other, source.directory, snapshot, changed)
    assert other.rows() == []
    snapshot.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="旧数据快照"):
        reuse_unchanged(other, source.directory, snapshot, cases)


def test_trace_contexts_are_isolated_and_secrets_are_redacted():
    with capture_trace() as outer:
        emit("test", "one", api_key="private", text="Bearer very-secret")
        with capture_trace() as inner:
            emit("test", "two")
        emit("test", "three")
    emit("test", "not-captured")
    assert [r["event"] for r in outer] == ["one", "three"]
    assert [r["event"] for r in inner] == ["two"]
    assert "very-secret" not in json.dumps(outer) and "private" not in json.dumps(outer)
    assert redact({"nested": {"password": "secret"}})["nested"]["password"] == "[REDACTED]"


class ToolModel:
    def __init__(self, args):
        self.args, self.calls = args, 0
    def bind_tools(self, tools):
        return self
    def invoke(self, prompt):
        self.calls += 1
        return AIMessage(content="done") if self.calls > 1 else AIMessage(content="", tool_calls=[{"name": "multiply_numbers", "args": self.args, "id": "x"}])


def test_invalid_arguments_never_appear_as_executed_tools():
    with capture_trace() as trace:
        result = ToolAgent(ToolModel({"a": "bad", "b": 8}), build_tools(profile="legacy")).run("multiply")
    assert result.executed_tool_names == []
    assert result.execution_records[0]["invoked"] is False
    assert not any(r["event"] == "invoked" for r in trace)


def test_tool_execution_evidence_survives_final_model_failure():
    class FailingFinal(ToolModel):
        def invoke(self, prompt):
            if self.calls:
                raise TimeoutError("final model failed")
            return super().invoke(prompt)
    with capture_trace() as trace, pytest.raises(TimeoutError):
        ToolAgent(FailingFinal({"a": 2, "b": 3}), build_tools(profile="legacy")).run("multiply")
    results = [r for r in trace if r["event"] == "result"]
    assert results[0]["data"]["invoked"] is True
    assert results[0]["data"]["status"] == "success"


def test_calibration_drafts_are_not_human_evidence():
    samples = deepcopy(read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl"))
    for sample in samples:
        sample.update(human_labels=None, review_status="pending_human_review")
    validate_samples(samples)
    assert len(samples) == 60
    report = calibration_report(samples, [], {"rag": "a", "security": "b"})
    assert report["status"] == "pending" and report["agreement"] is None
    assert report["human_review_pending"] == 60


def test_critical_false_pass_blocks_calibration_even_with_complete_labels():
    samples = [deepcopy(next(s for s in read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl") if s["suite"] == "security" and s["defect_type"] == "unsafe"))]
    sample = samples[0]
    sample.update(split="validation", human_labels=sample["suggested_labels"].copy(), review_status="reviewed")
    prediction = {"sample_id": sample["sample_id"], "status": "completed", "labels": {"attack_resisted": True, "expected_behavior_satisfied": True, "sensitive_information_disclosed": False, "unauthorized_tool_called": False}}
    report = calibration_report(samples, [prediction], {})
    assert report["status"] == "failed" and report["critical_false_passes"] == 1


def test_review_without_labels_and_non_boolean_labels_are_rejected():
    sample = deepcopy(read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl")[0])
    sample["review_status"] = "reviewed"
    sample["human_labels"] = None
    with pytest.raises(ValueError, match="参考标签"):
        validate_samples([sample])
    with pytest.raises(ValueError):
        label_passed("security", {"attack_resisted": "true"})


@pytest.mark.parametrize("xml", ["<testsuites/>", '<testsuites><testsuite><testcase classname="required"><skipped/></testcase></testsuite></testsuites>', '<testsuites><testsuite><testcase classname="other"/></testsuite></testsuites>'])
def test_empty_skipped_and_missing_suite_gates_fail(tmp_path, xml):
    path = tmp_path / "junit.xml"
    path.write_text(xml, encoding="utf-8")
    assert inspect_junit(path, expected_suites=["required"])["status"] == "failed"


def test_nonzero_pytest_exit_cannot_be_hidden_by_old_green_xml(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text('<testsuites><testsuite><testcase classname="required"/></testsuite></testsuites>', encoding="utf-8")
    assert inspect_junit(path, returncode=2, expected_suites=["required"])["status"] == "failed"


def test_malformed_junit_is_a_failed_gate(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text("<broken", encoding="utf-8")
    assert inspect_junit(path)["status"] == "failed"
