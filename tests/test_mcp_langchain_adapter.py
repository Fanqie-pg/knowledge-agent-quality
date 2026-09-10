import asyncio

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from app.mcp_langchain_adapter import MCPToolExecutionError, build_mcp_langchain_tools


pytestmark = pytest.mark.unit


MCP_MULTIPLY_TOOL = Tool(
    name="multiply_numbers",
    description="计算两个数的乘积。",
    inputSchema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    },
)


class FakeMCPClient:

    def __init__(self, call_result: CallToolResult):
        self.call_result = call_result
        self.calls = []

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=[MCP_MULTIPLY_TOOL])

    async def call_tool(self, tool_name: str, arguments: dict) -> CallToolResult:
        self.calls.append((tool_name, arguments))
        return self.call_result


def build_success_result() -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text="96.0")],
        structuredContent={"result": 96.0},
    )


def test_adapter_converts_mcp_tool_metadata():
    client = FakeMCPClient(build_success_result())

    tools = asyncio.run(build_mcp_langchain_tools(client))

    assert len(tools) == 1

    tool = tools[0]
    assert tool.name == "multiply_numbers"
    assert tool.description == "计算两个数的乘积。"
    assert tool.args_schema == MCP_MULTIPLY_TOOL.input_schema
    assert tool.metadata == {"source": "mcp", "side_effects": False, "requires_identity": False}


def test_adapter_supports_synchronous_tool_invocation():
    client = FakeMCPClient(build_success_result())
    tool = asyncio.run(build_mcp_langchain_tools(client))[0]

    result = tool.invoke({"a": 12, "b": 8})

    assert result == 96.0
    assert client.calls == [("multiply_numbers", {"a": 12, "b": 8})]


def test_adapter_supports_asynchronous_tool_invocation():
    client = FakeMCPClient(build_success_result())
    tool = asyncio.run(build_mcp_langchain_tools(client))[0]

    result = asyncio.run(tool.ainvoke({"a": 12, "b": 8}))

    assert result == 96.0
    assert client.calls == [("multiply_numbers", {"a": 12, "b": 8})]


def test_adapter_converts_mcp_tool_error():
    error_result = CallToolResult(
        content=[TextContent(type="text", text="员工不存在")],
        isError=True,
    )
    client = FakeMCPClient(error_result)
    tool = asyncio.run(build_mcp_langchain_tools(client))[0]

    with pytest.raises(MCPToolExecutionError, match="员工不存在"):
        tool.invoke({"a": 12, "b": 8})
