from typing import Any
import asyncio

from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, ListToolsResult


class MCPUnavailableError(RuntimeError):
    """MCP服务无法启动、连接中断或通信失败。"""


class MCPTimeoutError(MCPUnavailableError):
    """MCP服务在规定时间内没有完成操作。"""


class MCPToolClient:

    def __init__(self, server_parameters: StdioServerParameters, read_timeout_seconds: float = 10):
        self.server_parameters = server_parameters
        self.read_timeout_seconds = read_timeout_seconds

    async def list_tools(self) -> ListToolsResult:
        async def operation() -> ListToolsResult:
            async with Client(self.server_parameters, read_timeout_seconds=self.read_timeout_seconds) as client:
                return await client.list_tools()

        try:
            return await asyncio.wait_for(operation(), timeout=self.read_timeout_seconds)
        except TimeoutError as error:
            raise MCPTimeoutError("MCP服务响应超时") from error
        except Exception as error:
            raise MCPUnavailableError("MCP服务不可用") from error

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        async def operation() -> CallToolResult:
            async with Client(self.server_parameters, read_timeout_seconds=self.read_timeout_seconds) as client:
                return await client.call_tool(tool_name, arguments)

        try:
            return await asyncio.wait_for(operation(), timeout=self.read_timeout_seconds)
        except TimeoutError as error:
            raise MCPTimeoutError("MCP服务响应超时") from error
        except Exception as error:
            raise MCPUnavailableError("MCP服务不可用") from error