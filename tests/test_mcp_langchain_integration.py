import asyncio
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters

from app.mcp_client import MCPToolClient
from app.mcp_langchain_adapter import MCPToolExecutionError, build_mcp_langchain_tools


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_real_mcp_client(repository, user_context) -> MCPToolClient:
    from app.mcp_tool_registry import UserBoundMCPClient
    parameters = UserBoundMCPClient(user_context=user_context, database_path=repository.path, profile="legacy").client().server_parameters
    return MCPToolClient(parameters, read_timeout_seconds=5)


def test_adapter_invokes_real_mcp_tools(seeded_business_repository, trusted_hr_context):
    client = build_real_mcp_client(seeded_business_repository, trusted_hr_context)

    tools = asyncio.run(build_mcp_langchain_tools(client))
    tools_by_name = {tool.name: tool for tool in tools}

    assert set(tools_by_name) == {"multiply_numbers", "get_leave_balance"}

    multiply_result = tools_by_name["multiply_numbers"].invoke({"a": 12, "b": 8})
    leave_result = tools_by_name["get_leave_balance"].invoke({"employee_id": "E001"})

    assert multiply_result == 96.0
    assert leave_result == 5


def test_adapter_converts_real_mcp_business_error(seeded_business_repository, trusted_hr_context):
    client = build_real_mcp_client(seeded_business_repository, trusted_hr_context)
    tools = asyncio.run(build_mcp_langchain_tools(client))
    tools_by_name = {tool.name: tool for tool in tools}

    with pytest.raises(MCPToolExecutionError, match="员工不存在"):
        tools_by_name["get_leave_balance"].invoke({"employee_id": "E999"})
