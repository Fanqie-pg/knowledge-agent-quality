import pytest

from app.agent_factory import build_unified_agent
from app.citation_validator import validate_citations
from app.intent_router import AgentRoute

pytestmark = pytest.mark.evaluation


@pytest.fixture(scope="module")
def unified_agent():
    return build_unified_agent(tool_profile="legacy")


def test_unified_agent_executes_tool_route(unified_agent):
    result = unified_agent.run("12乘以8是多少？")

    assert result.route_decision.route is AgentRoute.TOOL
    assert result.tool_result is not None
    assert result.rag_result is None

    tool_calls = result.tool_result.decision.tool_calls

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "multiply_numbers"
    assert tool_calls[0]["args"] == {
        "a": 12,
        "b": 8,
    }

    assert len(result.tool_result.tool_messages) == 1
    assert result.tool_result.tool_messages[0].content == "96.0"
    assert "96" in result.final_response.content


def test_unified_agent_executes_knowledge_route(unified_agent):
    result = unified_agent.run("P1工单多久必须首次响应？")

    assert result.route_decision.route is AgentRoute.KNOWLEDGE
    assert result.tool_result is None
    assert result.rag_result is not None
    assert result.rag_result.retrieval_results

    answer = result.final_response.content

    assert "15" in answer

    citation_result = validate_citations(answer, result.rag_result.retrieval_results)

    assert citation_result.has_citations
    assert citation_result.all_citations_grounded


def test_unified_agent_executes_chat_route(unified_agent):
    result = unified_agent.run("你好，请简单介绍一下自己")

    assert result.route_decision.route is AgentRoute.CHAT
    assert result.tool_result is None
    assert result.rag_result is None
    assert result.final_response.content.strip()
    assert "[来源：" not in result.final_response.content
