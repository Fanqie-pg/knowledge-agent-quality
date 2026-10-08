import pytest

from app.security_case_loader import SecurityAttackType, load_security_cases
from app.security_evaluator import evaluate_security_rules, find_unauthorized_tool_names
from app.security_executors import UnifiedSecurityExecutor
from app.security_judge import SecurityLLMJudge


pytestmark = pytest.mark.evaluation


@pytest.fixture(scope="module")
def unified_agent():
    from app.agent_factory import build_unified_agent

    return build_unified_agent()


@pytest.fixture(scope="module")
def security_executor(unified_agent):
    return UnifiedSecurityExecutor(unified_agent)


@pytest.fixture(scope="module")
def security_judge(unified_agent):
    return SecurityLLMJudge()


def test_unified_agent_blocks_unauthorized_tool_call(security_executor, security_judge):
    all_cases = load_security_cases("data/evaluation/baseline_v1/security_cases.jsonl")
    cases = [
        case
        for case in all_cases
        if case.attack_type is SecurityAttackType.UNAUTHORIZED_TOOL_CALL
    ]

    assert len(cases) == 1
    case = cases[0]

    trace = security_executor.execute_with_trace(case)

    rule_result = evaluate_security_rules(case, trace.response)

    unauthorized_tool_names = find_unauthorized_tool_names(case, trace.tool_names)

    judge_result = security_judge.evaluate(
        case=case,
        answer=trace.response,
        actual_route=trace.actual_route,
        tool_names=trace.tool_names,
    )

    print(f"用例：{case.case_id}")
    print(f"问题：{case.query}")
    print(f"实际路由：{trace.actual_route}")
    print(f"越权工具：{unauthorized_tool_names}")
    print(f"回答：{trace.response}")
    print(f"规则结果：{rule_result.passed}")
    print(f"Judge结果：{judge_result.model_dump()}")
    print(f"模型请求工具：{trace.requested_tool_names}")
    print(f"真正执行工具：{trace.tool_names}")
    print(f"权限拦截工具：{trace.blocked_tool_names}")

    passed = (
        rule_result.passed
        and not unauthorized_tool_names
        and judge_result.passed
    )

    assert passed, {
        "rule_result": rule_result,
        "unauthorized_tool_names": unauthorized_tool_names,
        "judge_result": judge_result.model_dump(),
        "trace": trace,
    }


def test_unified_agent_resists_direct_injection_and_prompt_extraction(security_executor,security_judge):
    all_cases = load_security_cases("data/evaluation/baseline_v1/security_cases.jsonl")

    expected_case_ids = {"SEC-001", "SEC-002", "SEC-003"}

    cases = [
        case for case in all_cases
        if case.case_id in expected_case_ids
    ]

    assert [case.case_id for case in cases] == ["SEC-001", "SEC-002", "SEC-003"]

    failures = []

    for case in cases:
        trace = (security_executor.execute_with_trace(case))

        rule_result = evaluate_security_rules(case, trace.response)

        unauthorized_tool_names = (find_unauthorized_tool_names(case, trace.tool_names))

        judge_result = security_judge.evaluate(
            case=case,
            answer=trace.response,
            actual_route=trace.actual_route,
            tool_names=trace.tool_names,
        )

        passed = (rule_result.passed and not unauthorized_tool_names and judge_result.passed)

        print("-" * 60)
        print(f"用例：{case.case_id}")
        print(f"实际路由：{trace.actual_route}")
        print(f"模型请求工具：{trace.requested_tool_names}")
        print(f"真正执行工具：{trace.tool_names}")
        print(f"回答：{trace.response}")
        print(f"规则结果：{rule_result.passed}")
        print(f"Judge结果：{judge_result.model_dump()}")
        print(f"最终通过：{passed}")

        if not passed:
            failures.append(
                {
                    "case_id": case.case_id,
                    "rule_result": rule_result,
                    "unauthorized_tool_names": (
                        unauthorized_tool_names
                    ),
                    "judge_result": (
                        judge_result.model_dump()
                    ),
                    "trace": trace,
                }
            )

    assert not failures, failures


def test_unified_agent_does_not_disclose_sensitive_canary(security_executor, security_judge):
    all_cases = load_security_cases("data/evaluation/baseline_v1/security_cases.jsonl")

    cases = [
        case for case in all_cases
        if case.attack_type is SecurityAttackType.SENSITIVE_DATA_EXTRACTION
    ]

    assert len(cases) == 1
    case = cases[0]
    assert case.case_id == "SEC-008"

    trace = security_executor.execute_with_trace(case)

    rule_result = evaluate_security_rules(case, trace.response)

    unauthorized_tool_names = find_unauthorized_tool_names(case, trace.tool_names)

    judge_result = security_judge.evaluate(
        case=case,
        answer=trace.response,
        actual_route=trace.actual_route,
        tool_names=trace.tool_names,
    )

    print(f"用例：{case.case_id}")
    print(f"实际路由：{trace.actual_route}")
    print(f"模型请求工具：{trace.requested_tool_names}")
    print(f"真正执行工具：{trace.tool_names}")
    print(f"回答：{trace.response}")
    print(f"发现禁用内容：{rule_result.forbidden_phrases_found}"
    )
    print(f"Judge结果：{judge_result.model_dump()}")

    passed = (
        trace.actual_route == "chat"
        and rule_result.passed
        and not unauthorized_tool_names
        and judge_result.passed
    )

    assert passed, {
        "actual_route": trace.actual_route,
        "rule_result": rule_result,
        "unauthorized_tool_names": (
            unauthorized_tool_names
        ),
        "judge_result": (
            judge_result.model_dump()
        ),
        "trace": trace,
    }
