import pytest

pytestmark = pytest.mark.unit

import json


from app.intent_router import AgentRoute
from app.routing_case_loader import load_routing_cases


def write_cases(tmp_path, cases):
    path = tmp_path / "routing_cases.jsonl"

    # json.dumps() 会把中文转为 \u 开头的 Unicode 编码，ensure_ascii=False确保中文原样显示
    content = "\n".join(
        json.dumps(case, ensure_ascii=False)
        for case in cases
    )
    path.write_text(content, encoding="utf-8")
    return path


def build_case(case_id="ROUTE-TEST-001", expected_route="tool"):
    return {
        "case_id": case_id,
        "query": "12乘以8是多少？",
        "expected_route": expected_route,
        "category": "calculation",
        "difficulty": "easy",
    }


def test_load_routing_cases_parses_cases(tmp_path):
    path = write_cases(
        tmp_path,
        [
            build_case(),
            build_case(
                case_id="ROUTE-TEST-002",
                expected_route="knowledge",
            ),
        ],
    )
    cases = load_routing_cases(path)

    assert len(cases) == 2
    assert cases[0].case_id == "ROUTE-TEST-001"
    assert cases[0].expected_route is AgentRoute.TOOL
    assert cases[1].expected_route is AgentRoute.KNOWLEDGE


def test_load_routing_cases_rejects_missing_file(tmp_path):
    with pytest.raises(
        ValueError,
        match="路由评测文件不存在",
    ):
        load_routing_cases(tmp_path / "missing.jsonl")


def test_load_routing_cases_rejects_empty_file(tmp_path):
    path = write_cases(tmp_path, [])

    with pytest.raises(ValueError, match="路由评测文件不能为空"):
        load_routing_cases(path)


def test_load_routing_cases_rejects_invalid_route(tmp_path):
    path = write_cases(
        tmp_path,
        [build_case(expected_route="unknown")],
    )

    with pytest.raises(ValueError, match="第1行字段不合法"):
        load_routing_cases(path)


def test_load_routing_cases_rejects_duplicate_case_id(tmp_path):
    path = write_cases(
        tmp_path,
        [
            build_case(),
            build_case(),
        ],
    )

    with pytest.raises(ValueError, match="发现重复case_id：ROUTE-TEST-001"):
        load_routing_cases(path)


def test_real_routing_cases_are_balanced():
    cases = load_routing_cases(
        "data/evaluation/routing_cases.jsonl"
    )

    route_counts = {
        route: sum(
            case.expected_route is route
            for case in cases
        )
        for route in AgentRoute
    }

    assert len(cases) == 36
    assert route_counts == {
        AgentRoute.TOOL: 12,
        AgentRoute.KNOWLEDGE: 12,
        AgentRoute.CHAT: 12,
    }