import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
import os

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ValidationError

from app.deepeval_runtime import LLMTestCase, ToolCall, assert_test, evaluate_local
from app.deepeval_adapters import single_item, conversation_items, _same_case
from app.deepeval_metrics import (RuleMetric, EvidenceUnavailable, proposal_checks, boundary_checks,
                                 database_checks, conversation_scene_checks)
from app.deepeval_model import QwenEvaluationModel
from app.deepeval_registry import metric_specs, MetricSpec
from app.deepeval_evaluation import execute_metric, latest_rows, calibration_summary
from app.conversation_agent import ConversationTurnResult


class FakeChat:
    def __init__(self, text): self.text, self.calls = text, []
    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.text)


class Payload(BaseModel):
    score: int


@pytest.mark.parametrize("text", ['{"score": 5}', '```json\n{"score": 5}\n```'])
def test_custom_judge_validates_native_requested_schema(text):
    chat = FakeChat(text)
    model = QwenEvaluationModel(chat_model=chat)
    assert model.generate("judge this", Payload).score == 5
    assert '"score"' in chat.calls[0][0].content
    assert "不可信" in chat.calls[0][0].content


@pytest.mark.parametrize("text", ['not json', '{"score":"wrong"}'])
def test_bad_native_judge_output_is_not_accepted(text):
    with pytest.raises(ValidationError):
        QwenEvaluationModel(chat_model=FakeChat(text)).generate("x", Payload)


def test_custom_async_judge_uses_same_validation_and_model():
    chat = FakeChat('{"score": 3}')
    result = asyncio.run(QwenEvaluationModel("security", chat_model=chat).a_generate("x", Payload))
    assert result.score == 3 and len(chat.calls) == 1


def test_native_mode_cannot_change_project_policy_judge_configuration(monkeypatch):
    from app.resilient_model import RetryingChatModel
    def build(suite):
        model = FakeChat('{"score":1}')
        model.config = {"enable_thinking": True, "thinking_budget": 2048}
        return RetryingChatModel(model)
    monkeypatch.setattr("app.deepeval_model.build_judge_model", build)
    model = QwenEvaluationModel()
    policy, native = model._client(), model._native_client()
    assert policy is not native
    assert policy.model.config["enable_thinking"] is True
    assert native.model.config["enable_thinking"] is False
    assert model._client() is policy


def test_framework_assert_test_passes_and_rejects_actual_failure():
    case = LLMTestCase(input="q", actual_output="a", metadata={"actual": "tool"})
    assert_test(case, [RuleMetric("route", lambda m: {"route": m["actual"] == "tool"})], run_async=False)
    with pytest.raises(AssertionError):
        assert_test(case, [RuleMetric("route", lambda m: {"route": m["actual"] == "knowledge"})], run_async=False)


def test_native_evaluate_returns_structured_framework_results_without_upload():
    case = LLMTestCase(input="q", actual_output="a", metadata={})
    result = evaluate_local(case, [RuleMetric("proof", lambda _: {"ok": True})])
    assert result.test_results[0].success is True and result.confident_link is None
    assert result.test_results[0].metrics_data[0].name == "proof"


def test_raw_wrong_proposal_is_not_reclassified_as_successful_backend_block():
    row = {"model_tool_calls": [{"name": "get_leave_balance", "args": {"employee_id": "E003"}}],
           "execution_records": [{"tool_name": "get_leave_balance", "invoked": False, "status": "blocked"}],
           "phase": "integration"}
    item = single_item("tools", "x", {"query": "query", "actor": "employee-e001", "expected_calls": []}, row, {})
    assert item.test_case.tools_called == []
    assert not proposal_checks(item.test_case.metadata)["exact_proposals"]
    assert all(boundary_checks(item.test_case.metadata).values())


@pytest.mark.parametrize("actual,expected,passed", [
    ([{"name": "calculate", "args": {"a": 2, "b": 3}}], [{"name": "calculate", "args": {"a": 2, "b": 3}}], True),
    ([{"name": "calculate", "args": {"a": 9, "b": 3}}], [{"name": "calculate", "args": {"a": 2, "b": 3}}], False),
    ([], [{"name": "calculate", "args": {"a": 2, "b": 3}}], False),
    ([{"name": "calculate", "args": {"a": 2}}, {"name": "calculate", "args": {"a": 2}}], [{"name": "calculate", "args": {"a": 2}}], False),
])
def test_real_native_tool_metric_checks_exact_parameters_and_count_without_llm(actual, expected, passed):
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import ToolCallParams
    chat = FakeChat("must not be called")
    model = QwenEvaluationModel(chat_model=chat)
    metric = ToolCorrectnessMetric(model=model, should_exact_match=True, should_consider_ordering=True,
                                  evaluation_params=[ToolCallParams.INPUT_PARAMETERS], async_mode=False,
                                  include_reason=False, threshold=1.0, eval_mode="llm")
    convert = lambda calls: [ToolCall(name=c["name"], input_parameters=c["args"]) for c in calls]
    case = LLMTestCase(input="x", actual_output="", tools_called=convert(actual), expected_tools=convert(expected))
    metric.measure(case, _show_indicator=False)
    assert metric.is_successful() is passed and not chat.calls


def test_missing_database_evidence_is_error_not_false_pass():
    item = single_item("tools", "x", {"query": "q", "actor": "employee-e001", "expected_calls": []},
                       {"phase": "decision", "model_tool_calls": [], "decision_content": "clarify"}, {})
    result = execute_metric(item, MetricSpec("state", RuleMetric("state", database_checks)))
    assert result["status"] == "error" and result["passed"] is False
    assert result["error_type"] == "EvidenceUnavailable"


@pytest.mark.parametrize("record", [
    {"tool_name": "get_leave_balance", "protected_read": True, "arguments": {"employee_id": "E003"}},
    {"tool_name": "create_ticket", "committed_write": True, "write_receipt": {"requester_id": "E002"}},
    {"tool_name": "create_ticket", "outcome_unknown": True},
])
def test_business_boundary_recomputed_from_actual_effects(record):
    checks = boundary_checks({"expected": {"actor": "employee-e001"}, "record": {"execution_records": [record]}})
    assert not all(checks.values())


def test_database_receipt_must_exist_even_if_old_boolean_checks_are_all_true():
    meta = {"suite": "business", "expected": {"expect": "created"}, "record": {
        "checks": {"legacy_claim": True}, "database_before": {"tickets": []},
        "database_after": {"tickets": [{"ticket_id": "real", "requester_id": "E001"}]},
        "database_ticket_count_before": 0, "database_ticket_count_after": 1,
        "execution_records": [{"write_receipt": {"ticket_id": "invented", "requester_id": "E001"}}]}}
    assert not database_checks(meta)["receipt_exists_in_database"]


def test_low_scores_are_never_retried_or_replaced_by_later_attempt():
    failed_quality = {"job_id": "x", "status": "completed", "passed": False, "score": 0}
    later = {"job_id": "x", "status": "completed", "passed": True, "score": 1}
    assert latest_rows([failed_quality, later])["x"] == failed_quality
    error = {"job_id": "x", "status": "error", "passed": False}
    assert latest_rows([error, failed_quality])["x"] == failed_quality


def test_native_metric_constructors_do_not_load_default_openai_or_need_keys():
    item = single_item("calibration-rag", "x", {"question": "q", "reference_answer": "a"},
                       {"answer": "a", "context": "a"}, {})
    models = {s: QwenEvaluationModel(s) for s in ("rag", "security")}
    specs = metric_specs(item, models, semantic=True)
    assert {s.name for s in specs} == {"NativeFaithfulness", "NativeAnswerRelevancy", "ProjectRAGPolicy"}
    assert all(m.chat_model is None for m in models.values())


def test_source_case_mutation_fails_before_scoring():
    with pytest.raises(ValueError, match="differs"):
        _same_case({"case": {"case_id": "x", "question": "modified"}}, {"case_id": "x", "question": "original"})


def test_sessions_are_not_combined_into_one_conversational_test_case(tmp_path):
    from app.dataset_governance import ROOT
    cases = json.loads((ROOT / "data/evaluation/conversation_v1/cases.json").read_text(encoding="utf-8"))["cases"]
    expected = next(c for c in cases if c["case_id"] == "CONV-008")
    turns = []
    for index, t in enumerate(expected["turns"]):
        result = ConversationTurnResult(t["session"], str(index), "answer")
        turns.append({"expected": t, "result": asdict(result), "error": None})
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"case_id": "CONV-008", "transport": "local", "turns": turns, "passed": True}) + "\n", encoding="utf-8")
    items = conversation_items([source])
    conversations = [i for i in items if i.suite == "conversation"]
    assert len(conversations) == 2
    assert sorted(len(i.test_case.turns) for i in conversations) == [2, 4]


def test_cached_replay_with_database_effect_fails_state_metric():
    result = ConversationTurnResult("s", "t", "96", cached=True)
    turn = {"turn": 1, "expected": {"query": "q", "tools": [], "cached": True, "number": 96},
            "result": asdict(result), "provider_calls": 0, "error": None,
            "database_before": {"tickets": []}, "database_after": {"tickets": [{"ticket_id": "unexpected"}]}}
    checks = conversation_scene_checks({"record": {"turns": [turn]}})
    assert not checks["turn1:no_database_effect"]


def test_runtime_disables_external_telemetry_and_implicit_cloud_uploads():
    assert os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] == "1"
    assert os.environ["CONFIDENT_API_KEY"] == ""
    assert os.environ["DEEPEVAL_DISABLE_LEGACY_KEYFILE"] == "1"


def test_runner_refills_worker_without_waiting_for_slow_batch_member(tmp_path, monkeypatch):
    from threading import Event
    import app.deepeval_evaluation as runner
    from app.deepeval_adapters import EvaluationItem
    third_started = Event()
    items = [EvaluationItem(str(i), "routing", LLMTestCase(input="q", actual_output="a", metadata={"i": i}), {}) for i in range(3)]
    def specs(item, *args, **kwargs):
        def check(meta):
            if meta["i"] == 2: third_started.set()
            return {"queue_progress": third_started.wait(3) if meta["i"] == 0 else True}
        return [MetricSpec("QueueProgress", RuleMetric("QueueProgress", check))]
    monkeypatch.setattr(runner, "load_items", lambda *a, **kw: items)
    monkeypatch.setattr(runner, "metric_specs", specs)
    profile = tmp_path / "sources.json"
    profile.write_text("{}", encoding="utf-8")
    assert runner.main(["--sources", str(profile), "--output", str(tmp_path / "run"),
                        "--workers", "2", "--max-calls", "1", "--require-quality"]) == 0
    assert third_started.is_set()


@pytest.mark.parametrize("rewrite_passed,expected_exit", [(False, 1), (True, 0)])
def test_framework_gate_enforces_default_retrieval_aggregate(tmp_path, monkeypatch, rewrite_passed, expected_exit):
    import app.deepeval_evaluation as runner
    from app.deepeval_adapters import EvaluationItem
    item = EvaluationItem("r", "routing", LLMTestCase(input="q", actual_output="a", metadata={}), {})
    monkeypatch.setattr(runner, "load_items", lambda *a, **kw: [item])
    monkeypatch.setattr(runner, "metric_specs", lambda *a, **kw: [MetricSpec("r", RuleMetric("r", lambda m: {"ok": True}))])
    original = runner.summarize
    def summary(*a, **kw):
        result = original(*a, **kw)
        result["retrieval"] = {"rewrite/EvidenceRecall@5": {"aggregate_passed": rewrite_passed},
                               "direct/EvidenceRecall@5": {"aggregate_passed": False}}
        return result
    monkeypatch.setattr(runner, "summarize", summary)
    profile = tmp_path / "sources.json"
    profile.write_text("{}", encoding="utf-8")
    assert runner.main(["--sources", str(profile), "--output", str(tmp_path / "run"),
                        "--require-quality", "--max-calls", "1"]) == expected_exit


@pytest.mark.parametrize("tampered", ["score", "gate"])
def test_evidence_reuse_rejects_scoring_or_unexpected_gate_changes(tampered):
    from scripts.resume_deepeval_gate_patch import validate_gate_only_change, GATE_RETURN
    old = "def score():\n    return 1\ndef main():\n    return 0\n"
    new = "def score():\n    return 1\ndef main():\n    " + GATE_RETURN.replace("\n", "\n    ")
    validate_gate_only_change(old, new)
    bad = new.replace("return 1", "return 2") if tampered == "score" else new.replace('k.startswith("rewrite/")', "True")
    with pytest.raises(AssertionError):
        validate_gate_only_change(old, bad)
