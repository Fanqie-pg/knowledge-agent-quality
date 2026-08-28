import asyncio
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_server_parameters() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
    )


def call_mcp_tool(tool_name: str, arguments: dict[str, object]):
    async def call_tool():
        async with Client(build_server_parameters()) as client:
            return await client.call_tool(tool_name, arguments)

    return asyncio.run(call_tool())


def test_mcp_server_exposes_expected_tools():
    async def list_tools():
        async with Client(build_server_parameters()) as client:
            return await client.list_tools()

    result = asyncio.run(list_tools())
    tools_by_name = {tool.name: tool for tool in result.tools}

    assert set(tools_by_name) == {
        "multiply_numbers",
        "get_leave_balance",
    }
    multiply_schema = tools_by_name["multiply_numbers"].input_schema
    assert set(multiply_schema["properties"]) == {"a", "b"}
    assert set(multiply_schema["required"]) == {"a", "b"}

    leave_schema = tools_by_name["get_leave_balance"].input_schema
    assert set(leave_schema["properties"]) == {"employee_id"}
    assert leave_schema["required"] == ["employee_id"]


def test_mcp_server_calls_tools_successfully():
    async def call_tools():
        async with Client(build_server_parameters()) as client:
            multiply_result = await client.call_tool(
                "multiply_numbers",
                {"a": 12, "b": 8},
            )
            leave_result = await client.call_tool(
                "get_leave_balance",
                {"employee_id": "E001"},
            )
            return multiply_result, leave_result

    multiply_result, leave_result = asyncio.run(call_tools())

    assert multiply_result.is_error is False
    assert multiply_result.content[0].text == "96.0"
    assert multiply_result.structured_content == {"result": 96.0}

    assert leave_result.is_error is False
    assert leave_result.content[0].text == "5"
    assert leave_result.structured_content == {"result": 5}


def test_mcp_server_rejects_invalid_tool_arguments():
    result = call_mcp_tool("multiply_numbers", {"a": "abc", "b": 8})

    assert result.is_error is True
    assert result.structured_content is None
    assert "multiply_numbers" in result.content[0].text
    assert "abc" in result.content[0].text


def test_mcp_server_rejects_unknown_tool():
    result = call_mcp_tool("missing_tool", {})

    assert result.is_error is True
    assert result.structured_content is None
    assert result.content[0].text == "Unknown tool: missing_tool"


def test_mcp_server_returns_safe_business_error():
    result = call_mcp_tool("get_leave_balance", {"employee_id": "E999"})

    assert result.is_error is True
    assert result.structured_content is None

    error_text = result.content[0].text
    assert error_text.endswith("员工不存在")
    assert "Traceback" not in error_text