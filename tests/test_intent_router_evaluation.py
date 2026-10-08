import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.intent_router import AgentRoute, IntentRouter
from app.routing_case_loader import load_routing_cases
from app.routing_evaluator import evaluate_routing_cases


pytestmark = [pytest.mark.evaluation, pytest.mark.quality_gate]

@pytest.fixture(scope="module")
def real_router():
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")

    from app.agent_factory import build_chat_model
    chat_model = build_chat_model()

    return IntentRouter(chat_model)

def test_real_router_meets_quality_threshold(real_router):
    cases = load_routing_cases(
        "data/evaluation/baseline_v1/routing_cases.jsonl"
    )

    summary = evaluate_routing_cases(
        real_router,
        cases,
    )

    failed_cases = [
        {
            "case_id": result.case_id,
            "query": result.query,
            "expected": result.expected_route.value,
            "actual": result.actual_route.value,
            "reason": result.reason,
        }
        for result in summary.case_results
        if not result.correct
    ]

    report = {
        "case_count": summary.case_count,
        "accuracy": summary.accuracy,
        "per_route_accuracy": {
            route.value: accuracy
            for route, accuracy
            in summary.per_route_accuracy.items()
        },
        "confusion_matrix": {
            expected.value: {
                actual.value: count
                for actual, count in actual_counts.items()
            }
            for expected, actual_counts
            in summary.confusion_matrix.items()
        },
        "failed_cases": failed_cases,
    }

    assert summary.accuracy >= 0.9, report
    for route in AgentRoute:
        assert (
            summary.per_route_accuracy[route] >= 0.75
        ), report