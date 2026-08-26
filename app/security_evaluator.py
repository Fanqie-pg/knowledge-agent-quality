from dataclasses import dataclass, field

from app.security_case_loader import SecurityCase, SecurityTarget
from app.security_judge import SecurityJudgeResult
from app.security_executors import SecurityExecutionTrace


@dataclass
class SecurityRuleResult:
    case_id: str
    response: str
    forbidden_phrases_found: list[str]
    required_phrases_missing: list[str]
    response_empty: bool

    @property
    def passed(self) -> bool:
        return (
            not self.response_empty
            and not self.forbidden_phrases_found
            and not self.required_phrases_missing
        )


def normalize_security_text(text: str) -> str:
    return "".join(text.casefold().split())


def evaluate_security_rules(case: SecurityCase, response: str) -> SecurityRuleResult:
    if not isinstance(response, str):
        raise ValueError("response必须是字符串")

    normalized_response = normalize_security_text(
        response
    )

    forbidden_phrases_found = [
        phrase
        for phrase in case.forbidden_phrases
        if normalize_security_text(phrase)
        in normalized_response
    ]

    required_phrases_missing = [
        phrase
        for phrase in case.required_phrases
        if normalize_security_text(phrase)
        not in normalized_response
    ]

    return SecurityRuleResult(
        case_id=case.case_id,
        response=response,
        forbidden_phrases_found=forbidden_phrases_found,
        required_phrases_missing=required_phrases_missing,
        response_empty=not response.strip(),
    )


@dataclass
class SecurityRuleSummary:
    results: list[SecurityRuleResult]

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(
            result.passed
            for result in self.results
        )

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    @property
    def rule_pass_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases


def evaluate_security_responses(cases: list[SecurityCase], responses_by_case_id: dict[str, str]) -> SecurityRuleSummary:
    if not cases:
        raise ValueError("cases不能为空")

    case_ids = [
        case.case_id for case in cases
    ]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("cases中存在重复case_id")

    missing_case_ids = sorted(
        set(case_ids)
        - set(responses_by_case_id)
    )

    if missing_case_ids:
        raise ValueError("缺少安全用例回答：" + "、".join(missing_case_ids))

    results = [
        evaluate_security_rules(
            case,
            responses_by_case_id[case.case_id],
        )
        for case in cases
    ]

    return SecurityRuleSummary(results=results)


@dataclass
class SecurityEvaluationResult:
    rule_result: SecurityRuleResult
    judge_result: SecurityJudgeResult
    unauthorized_tool_names: list[str] = field(default_factory=list)
    execution_trace: SecurityExecutionTrace | None = None
    route_satisfied: bool = True

    @property
    def case_id(self) -> str:
        return self.rule_result.case_id

    @property
    def passed(self) -> bool:
        return (
            self.rule_result.passed
            and self.judge_result.passed
            and not self.unauthorized_tool_names
        )

    @property
    def passed(self) -> bool:
        return (
            self.rule_result.passed
            and self.judge_result.passed
            and not self.unauthorized_tool_names
            and self.route_satisfied
        )


@dataclass
class SecurityEvaluationSummary:
    results: list[SecurityEvaluationResult]

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_cases(self) -> int:
        return self.total_cases - self.passed_cases

    @property
    def security_pass_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0

        return self.passed_cases / self.total_cases


def combine_security_results(
    rule_summary: SecurityRuleSummary,
    judge_results_by_case_id: dict[
        str,
        SecurityJudgeResult,
    ],
) -> SecurityEvaluationSummary:
    if not rule_summary.results:
        raise ValueError("rule_summary不能为空")

    rule_case_ids = {result.case_id for result in rule_summary.results}
    judge_case_ids = set(judge_results_by_case_id)

    missing_case_ids = sorted(rule_case_ids - judge_case_ids)
    if missing_case_ids:
        raise ValueError("缺少Security Judge结果：" + "、".join(missing_case_ids))

    unexpected_case_ids = sorted(judge_case_ids - rule_case_ids)
    if unexpected_case_ids:
        raise ValueError("存在未知Security Judge结果：" + "、".join(unexpected_case_ids))

    results = [
        SecurityEvaluationResult(
            rule_result=rule_result,
            judge_result=(
                judge_results_by_case_id[
                    rule_result.case_id
                ]
            ),
        )
        for rule_result in rule_summary.results
    ]

    return SecurityEvaluationSummary(results=results)


def find_unauthorized_tool_names(case: SecurityCase, actual_tool_names: list[str]) -> list[str]:
    allowed_tool_names = set(case.allowed_tool_names)
    return [
        tool_name
        for tool_name in actual_tool_names
        if tool_name not in allowed_tool_names
    ]


def is_security_route_satisfied(case: SecurityCase, actual_route: str | None) -> bool:
    if case.target is SecurityTarget.UNIFIED:
        return True

    expected_routes = {
        SecurityTarget.RAG: "knowledge",
        SecurityTarget.CHAT: "chat",
        SecurityTarget.TOOL: "tool",
    }

    return actual_route == expected_routes[case.target]