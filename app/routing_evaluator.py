from dataclasses import dataclass

from app.intent_router import AgentRoute, IntentRouter
from app.routing_case_loader import RoutingCase

@dataclass
class RoutingCaseResult:
    case_id: str
    query: str
    expected_route: AgentRoute
    actual_route: AgentRoute
    correct: bool
    reason: str


@dataclass
class RoutingEvaluationSummary:
    case_count: int
    correct_count: int
    accuracy: float
    per_route_accuracy: dict[AgentRoute, float]
    confusion_matrix: dict[
        AgentRoute,
        dict[AgentRoute, int],
    ]
    case_results: list[RoutingCaseResult]


def evaluate_routing_cases(router: IntentRouter, cases: list[RoutingCase]) -> RoutingEvaluationSummary:
    if not cases:
        raise ValueError("路由评测用例不能为空")

    case_results = []

    for case in cases:
        decision = router.route(case.query)
        correct = decision.route is case.expected_route

        case_results.append(
            RoutingCaseResult(
                case_id=case.case_id,
                query=case.query,
                expected_route=case.expected_route,
                actual_route=decision.route,
                correct=correct,
                reason=decision.reason,
            )
        )

    correct_count = sum(
        result.correct for result in case_results
    )

    per_route_accuracy = {}

    for route in AgentRoute:
        route_results = [
            result
            for result in case_results
            if result.expected_route is route
        ]

        if route_results:
            per_route_accuracy[route] = (
                sum(result.correct for result in route_results)
                / len(route_results)
            )

    confusion_matrix = {
        expected_route: {
            actual_route: 0
            for actual_route in AgentRoute
        }
        for expected_route in AgentRoute
    }

    for result in case_results:
        confusion_matrix[
            result.expected_route
        ][result.actual_route] += 1

    return RoutingEvaluationSummary(
        case_count=len(case_results),
        correct_count=correct_count,
        accuracy=correct_count / len(case_results),
        per_route_accuracy=per_route_accuracy,
        confusion_matrix=confusion_matrix,
        case_results=case_results,
    )