import asyncio
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from mcp.client.stdio import StdioServerParameters

from app.mcp_client import MCPToolClient
from app.mcp_langchain_adapter import build_mcp_langchain_tools
from app.tool_agent import ToolAgent


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeToolModel:

    def __init__(self, decision: AIMessage, final_response: AIMessage):
        self.decision = decision
        self.final_response = final_response
        self.bound_tools = None
        self.received_inputs = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, model_input):
        self.received_inputs.append(model_input)

        if len(self.received_inputs) == 1:
            return self.decision

        return self.final_response


def build_real_mcp_tools():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
    )
    client = MCPToolClient(parameters, read_timeout_seconds=5)
    return asyncio.run(build_mcp_langchain_tools(client))


def test_tool_agent_executes_real_mcp_tool():
    decision = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "multiply_numbers",
                "args": {"a": 12, "b": 8},
                "id": "mcp-call-1",
                "type": "tool_call",
            }
        ],
    )
    final_response = AIMessage(content="12乘以8等于96。", tool_calls=[])
    model = FakeToolModel(decision, final_response)

    tools = build_real_mcp_tools()
    agent = ToolAgent(model, tools)

    result = agent.run("12乘以8是多少？")

    assert {tool.name for tool in model.bound_tools} == {
        "multiply_numbers",
        "get_leave_balance",
    }

    assert result.requested_tool_names == ["multiply_numbers"]
    assert result.executed_tool_names == ["multiply_numbers"]
    assert result.blocked_tool_names == []

    assert len(result.tool_messages) == 1
    assert result.tool_messages[0].content == "96.0"
    assert result.tool_messages[0].status == "success"
    assert result.tool_messages[0].tool_call_id == "mcp-call-1"

    assert result.final_response is final_response
    assert result.final_response.content == "12乘以8等于96。"


def test_tool_agent_converts_real_mcp_business_error():
    decision = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_leave_balance",
                "args": {"employee_id": "E999"},
                "id": "mcp-call-2",
                "type": "tool_call",
            }
        ],
    )

    final_response = AIMessage(content="未找到该员工。", tool_calls=[])
    model = FakeToolModel(decision, final_response)

    tools = build_real_mcp_tools()
    agent = ToolAgent(model, tools)

    result = agent.run("查询员工E999的剩余年假")

    assert len(result.tool_messages) == 1

    tool_message = result.tool_messages[0]
    assert tool_message.status == "error"
    assert tool_message.content.endswith("员工不存在")
    assert tool_message.name == "get_leave_balance"
    assert tool_message.tool_call_id == "mcp-call-2"

    assert result.final_response is final_response