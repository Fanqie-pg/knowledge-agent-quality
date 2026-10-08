"""Frozen v1 retrieval baseline; v2 remains an exploratory corpus until reviewed."""
import pytest

pytestmark = [pytest.mark.evaluation, pytest.mark.quality_gate]


def test_retrieval_quality_gate():
    from app.agent_factory import build_knowledge_retriever
    from app.retrieval_case_loader import load_retrieval_cases
    from app.retrieval_evaluator import evaluate_retrieval_cases
    cases = load_retrieval_cases("data/evaluation/baseline_v1/retrieval_cases.jsonl")
    summary = evaluate_retrieval_cases(build_knowledge_retriever(), cases, k=5)
    assert summary.evaluated_case_count > 0
    assert summary.mean_recall_at_k >= .85, summary
    assert summary.mrr >= .85, summary
    assert summary.mean_evidence_recall_at_k >= .85, summary
