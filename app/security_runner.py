"""编排安全用例执行、确定性规则、权限检查、路由检查和 Judge。"""

from typing import Protocol

from app.security_case_loader import SecurityCase
from app.security_evaluator import (
    SecurityEvaluationResult,
    SecurityEvaluationSummary,
    SecurityRuleSummary,
    evaluate_security_responses,
    evaluate_security_rules,
    find_unauthorized_tool_names,
    is_security_route_satisfied,
)
from app.security_executors import SecurityExecutionTrace
from app.security_judge import SecurityJudgeResult


class SecurityCaseExecutor(Protocol):
    """只返回回答文本的基础安全执行器契约。"""

    def execute(self, case: SecurityCase) -> str:
        """执行一条安全用例并返回回答文本。"""


def run_security_cases(cases: list[SecurityCase], executor: SecurityCaseExecutor) -> SecurityRuleSummary:
    """执行安全用例并只汇总可确定判断的文本规则。"""
    if not cases:
        raise ValueError("cases不能为空")

    case_ids = [
        case.case_id for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("cases中存在重复case_id")

    responses_by_case_id = {}

    for case in cases:
        responses_by_case_id[case.case_id] = executor.execute(case)

    return evaluate_security_responses(
        cases,
        responses_by_case_id,
    )


class TracedSecurityCaseExecutor(Protocol):
    """返回回答及执行轨迹的安全执行器契约。"""

    def execute_with_trace(self, case: SecurityCase) -> SecurityExecutionTrace:
        ...


class SecurityCaseJudge(Protocol):
    """对回答和真实执行轨迹做语义安全判断的 Judge 契约。"""

    def evaluate(
        self,
        case: SecurityCase,
        answer: str,
        actual_route: str | None = None,
        tool_names: list[str] | None = None,
    ) -> SecurityJudgeResult:
        ...


def run_security_evaluation(
    cases: list[SecurityCase],
    executor: TracedSecurityCaseExecutor,
    judge: SecurityCaseJudge,
) -> SecurityEvaluationSummary:
    """组合规则、工具权限、目标路由和 Judge，生成完整安全结果。"""
    if not cases:
        raise ValueError("cases不能为空")

    case_ids = [case.case_id for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("cases中存在重复case_id")

    results = []

    for case in cases:
        trace = executor.execute_with_trace(case)

        rule_result = evaluate_security_rules(case, trace.response)

        unauthorized_tool_names = (find_unauthorized_tool_names(case, trace.tool_names))

        judge_result = judge.evaluate(
            case=case,
            answer=trace.response,
            actual_route=trace.actual_route,
            tool_names=trace.tool_names,
        )

        route_satisfied = is_security_route_satisfied(case, trace.actual_route)

        results.append(
            SecurityEvaluationResult(
                rule_result=rule_result,
                judge_result=judge_result,
                unauthorized_tool_names=(
                    unauthorized_tool_names
                ),
                execution_trace=trace,
                route_satisfied=route_satisfied,
            )
        )

    return SecurityEvaluationSummary(results=results)
