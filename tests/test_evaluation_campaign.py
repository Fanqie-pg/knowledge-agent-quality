import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from app.evaluation_campaign import open_campaign, planned_jobs, execute_job, summarize
from app.retriever import RetrievalResult

pytestmark = pytest.mark.unit


def test_campaign_freeze_rejects_changed_judge_effort(tmp_path, monkeypatch):
    store = open_campaign(tmp_path, freeze=True)
    assert all(j["split"] in {"dev", "holdout", "baseline"} for j in store.manifest["config"]["jobs"])
    original = store.manifest["config"]["judge"]
    monkeypatch.setattr("app.evaluation_campaign.judge_model_config", lambda suite="rag": {**original[suite], "thinking_budget": 512})
    with pytest.raises(ValueError, match="Frozen"):
        open_campaign(tmp_path)


def test_campaign_missing_results_are_visible_not_passes(tmp_path, monkeypatch):
    monkeypatch.setattr("app.evaluation_campaign.historical_ab_config",
                        lambda: {"historical_ab": None, "historical_ab_sha256": None})
    store = open_campaign(tmp_path, freeze=True)
    report = summarize(store)
    assert report["generation_rows"] == 0
    assert sum(g["missing"] for g in report["groups"].values()) == report["generation_expected"]
    assert report["judge_completed"] == 0
    assert report["historical_rescore"] == {"expected": 0, "completed": 0, "passed": 0}


@pytest.mark.parametrize("change", ["deleted", "changed"])
def test_frozen_history_cannot_disappear_or_change(tmp_path, monkeypatch, change):
    import app.evaluation_campaign as campaign
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    assert campaign.historical_ab_config() == {"historical_ab": None, "historical_ab_sha256": None}
    records = tmp_path / "reports/ab-dev-v2.0.1/records.jsonl"
    records.parent.mkdir(parents=True)
    records.write_text('{"record_id":"first"}\n', encoding="utf-8")
    frozen = campaign.historical_ab_config()
    assert frozen["historical_ab"] == "reports/ab-dev-v2.0.1" and frozen["historical_ab_sha256"]
    if change == "deleted":
        records.unlink()
    else:
        records.write_text('{"record_id":"changed"}\n', encoding="utf-8")
    store = SimpleNamespace(rows=lambda: [], manifest={"config": frozen})
    with pytest.raises(ValueError, match="Historical answers"):
        campaign.scoring_tasks(store)


def test_campaign_persists_raw_answer_context_and_rule_failure():
    job = next(j for j in planned_jobs() if j["suite"] == "rag" and j["case"]["answerable"])
    evidence = RetrievalResult(Document(page_content="evidence", metadata={"doc_id": "D", "chunk_id": "C"}), .9, 1)
    result = SimpleNamespace(final_response=AIMessage(content="unsupported citation [来源：D/UNKNOWN]"),
                             context="evidence", retrieval_query="original query", retrieval_results=[evidence], prompt_message=[])
    row = execute_job(job, {job["variant"]: SimpleNamespace(run=lambda q: result)})
    assert row["status"] == "completed" and row["structural_passed"] is False
    assert row["rules"]["citations_grounded"] is False
    assert row["answer"] == result.final_response.content
    assert row["retrieval"][0]["content"] == "evidence"


def test_campaign_keeps_execution_errors_separate_from_quality_failure():
    job = next(j for j in planned_jobs() if j["suite"] == "routing")
    def fail(query):
        raise TimeoutError()
    row = execute_job(job, {"router": SimpleNamespace(route=fail)})
    assert row["status"] == "error" and row["error_type"] == "TimeoutError"
    assert "passed" not in row
