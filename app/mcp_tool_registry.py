import asyncio
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool
from mcp.client.stdio import StdioServerParameters

from app.mcp_client import MCPToolClient
from app.mcp_langchain_adapter import build_mcp_langchain_tools


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_mcp_tools(read_timeout_seconds: float = 10) -> list[StructuredTool]:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
    )
    client = MCPToolClient(parameters, read_timeout_seconds=read_timeout_seconds)

    return asyncio.run(build_mcp_langchain_tools(client))