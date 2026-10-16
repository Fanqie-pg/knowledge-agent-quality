import pytest

from app.agent_factory import build_unified_agent
from app.intent_router import AgentRoute
from app.mcp_tool_registry import build_mcp_tools
from app.unified_agent import AgentRunStatus


pytestmark = [
    pytest.mark.evaluation,
    pytest.mark.quality_gate,
]


@pytest.fixture(scope="module")
def mcp_unified_agent(tmp_path_factory):
    from app.business_repository import SQLiteBusinessRepository
    from app.identity import load_user_context
    repository = SQLiteBusinessRepository(tmp_path_factory.mktemp("mcp-business") / "business.sqlite3")
    repository.seed_demo()
    user = load_user_context("hr-test")
    mcp_tools = build_mcp_tools(user_context=user, database_path=repository.path, profile="legacy")
    return build_unified_agent(tools=mcp_tools, user_context=user)


def test_mcp_unified_agent_executes_tool(mcp_unified_agent):
    result = mcp_unified_agent.run("请使用工具计算12乘以8。")

    assert result.status is AgentRunStatus.SUCCESS, result.failure
    assert result.route_decision is not None, result.failure

    assert result.route_decision.route is AgentRoute.TOOL
    assert result.tool_result is not None
    assert result.rag_result is None

    assert result.tool_result.requested_tool_names == ["multiply_numbers"]
    assert result.tool_result.executed_tool_names == ["multiply_numbers"]

    assert len(result.tool_result.tool_messages) == 1

    tool_message = result.tool_result.tool_messages[0]
    assert tool_message.name == "multiply_numbers"
    assert tool_message.status == "success"
    assert tool_message.content == "96.0"

    assert "96" in result.final_response.content


def test_mcp_unified_agent_handles_business_error(mcp_unified_agent):
    result = mcp_unified_agent.run("请使用工具查询员工E999的剩余年假。")

    assert result.status is AgentRunStatus.SUCCESS, result.failure
    assert result.route_decision is not None, result.failure

    assert result.route_decision.route is AgentRoute.TOOL
    assert result.tool_result is not None

    assert result.tool_result.requested_tool_names == ["get_leave_balance"]
    assert result.tool_result.executed_tool_names == ["get_leave_balance"]

    assert len(result.tool_result.tool_messages) == 1

    tool_message = result.tool_result.tool_messages[0]
    assert tool_message.name == "get_leave_balance"
    assert tool_message.status == "error"
    assert tool_message.content.endswith("员工不存在")

    assert "不存在" in result.final_response.content
