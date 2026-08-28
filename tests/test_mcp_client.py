import asyncio
import sys
from pathlib import Path

import pytest
from mcp.client.stdio import StdioServerParameters

from app.mcp_client import MCPToolClient, MCPUnavailableError, MCPTimeoutError


pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SLOW_SERVER_PATH = PROJECT_ROOT / "tests" / "fixtures" / "mcp_slow_server.py"


def test_mcp_client_converts_missing_command_to_unavailable_error():
    parameters = StdioServerParameters(command="missing-mcp-command-9f8e7d",)
    client = MCPToolClient(parameters)

    with pytest.raises(MCPUnavailableError, match="MCP服务不可用") as error:
        asyncio.run(client.list_tools())

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_mcp_client_converts_early_server_exit_to_unavailable_error():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.missing_mcp_server"],
        cwd=PROJECT_ROOT,
    )
    client = MCPToolClient(parameters)

    with pytest.raises(MCPUnavailableError, match="MCP服务不可用") as error:
        asyncio.run(client.list_tools())

    assert error.value.__cause__ is not None


def test_mcp_client_converts_unresponsive_server_to_timeout_error():
    parameters = StdioServerParameters(command=sys.executable, args=["-c", "import time; time.sleep(30)"])
    client = MCPToolClient(parameters, read_timeout_seconds=0.2)

    with pytest.raises(MCPTimeoutError, match="MCP服务响应超时") as error:
        asyncio.run(client.list_tools())

    assert isinstance(error.value.__cause__, TimeoutError)


def test_mcp_client_converts_slow_tool_to_timeout_error():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(SLOW_SERVER_PATH)],
        cwd=PROJECT_ROOT,
    )

    discovery_client = MCPToolClient(parameters, read_timeout_seconds=5)
    tools_result = asyncio.run(discovery_client.list_tools())
    assert [tool.name for tool in tools_result.tools] == ["slow_tool"]

    timeout_client = MCPToolClient(parameters, read_timeout_seconds=0.5)

    with pytest.raises(MCPTimeoutError, match="MCP服务响应超时") as error:
        asyncio.run(timeout_client.call_tool("slow_tool", {"delay_seconds": 30}))

    assert isinstance(error.value.__cause__, TimeoutError)