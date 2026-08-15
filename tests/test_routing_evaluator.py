import pytest

from app.intent_router import AgentRoute, RouteDecision
from app.routing_case_loader import RoutingCase
from app.routing_evaluator import evaluate_routing_cases


class FakeRouter:

    def __init__(self, decisions):
        self.decisions = decisions

    def route(self, query):
        return self.decisions[query]


def build_case(case_id, query, expected_route):
    return RoutingCase(
        case_id=case_id,
        query=query,
        expected_route=expected_route,
        category="test",
        difficulty="easy",
    )


def test_evaluate_routing_cases_calculates_metrics():
    cases = [
        build_case(
            "ROUTE-001",
            "计算12乘以8",
            AgentRoute.TOOL,
        ),
        build_case(
            "ROUTE-002",
            "P1多久首次响应",
            AgentRoute.KNOWLEDGE,
        ),
        build_case(
            "ROUTE-003",
            "你好",
            AgentRoute.CHAT,
        ),
    ]

    router = FakeRouter(
        {
            "计算12乘以8": RouteDecision(
                route="tool",
                reason="需要计算",
            ),
            "P1多久首次响应": RouteDecision(
                route="chat",
                reason="错误地判断成聊天",
            ),
            "你好": RouteDecision(
                route="chat",
                reason="普通问候",
            ),
        }
    )

    summary = evaluate_routing_cases(router, cases)

    assert summary.case_count == 3
    assert summary.correct_count == 2
    assert summary.accuracy == pytest.approx(2 / 3)

    assert summary.per_route_accuracy == {
        AgentRoute.TOOL: 1.0,
        AgentRoute.KNOWLEDGE: 0.0,
        AgentRoute.CHAT: 1.0,
    }
    assert (
        summary.confusion_matrix[AgentRoute.KNOWLEDGE][AgentRoute.CHAT] == 1
    )

    assert summary.case_results[1].correct is False
    assert summary.case_results[1].reason == "错误地判断成聊天"


def test_evaluate_routing_cases_rejects_empty_cases():
    router = FakeRouter({})

    with pytest.raises(ValueError, match="路由评测用例不能为空"):
        evaluate_routing_cases(router, [])