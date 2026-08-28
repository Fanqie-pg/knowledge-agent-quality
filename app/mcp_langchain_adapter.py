"""在 MCP 2.x Tool/Result 与 LangChain StructuredTool 之间做薄适配。"""

import asyncio
from typing import Any, Protocol

from langchain_core.tools import StructuredTool
from mcp.types import CallToolResult, TextContent, Tool, ListToolsResult



class MCPClientProtocol(Protocol):
    """描述适配器所需的最小 MCP Client 能力，便于注入 Fake。"""

    async def list_tools(self) -> ListToolsResult:
        ...

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> CallToolResult:
        ...


class MCPToolExecutionError(ValueError):
    """MCP工具返回了可预期的执行失败。"""


def _extract_text(result: CallToolResult) -> str:
    """合并 MCP 结果中的所有文本内容块。"""
    texts = [
        content.text
        for content in result.content
        if isinstance(content, TextContent)
    ]
    return "\n".join(texts)


def _convert_tool_result(result: CallToolResult) -> Any:
    """优先返回结构化结果，并把 MCP 业务失败转换成稳定异常。"""
    if result.is_error:
        error_message = _extract_text(result)
        raise MCPToolExecutionError(error_message or "MCP工具执行失败")

    structured_content = result.structured_content

    if (isinstance(structured_content, dict) and "result" in structured_content):
        return structured_content["result"]

    if structured_content is not None:
        return structured_content

    return _extract_text(result)


async def _call_mcp_tool(client: MCPClientProtocol, tool_name: str, arguments: dict[str, Any]) -> Any:
    """调用 MCP 工具并统一处理协议结果。"""
    result = await client.call_tool(tool_name, arguments)
    return _convert_tool_result(result)


def _build_langchain_tool(client: MCPClientProtocol, mcp_tool: Tool) -> StructuredTool:
    """把单个 MCP Tool 的元数据、Schema 和调用函数映射到 LangChain。"""
    tool_name = mcp_tool.name

    def invoke_tool(**arguments: Any) -> Any:
        """为同步 ToolAgent 桥接异步 MCP 调用。"""
        return asyncio.run(_call_mcp_tool(client, tool_name, arguments))

    async def ainvoke_tool(**arguments: Any) -> Any:
        """在异步 Agent 中直接复用 MCP 调用。"""
        return await _call_mcp_tool(client, tool_name, arguments)

    return StructuredTool(
        name=tool_name,
        description=mcp_tool.description or "",
        args_schema=mcp_tool.input_schema,
        func=invoke_tool,
        coroutine=ainvoke_tool,
        metadata={"source": "mcp"},
    )


async def build_mcp_langchain_tools(client: MCPClientProtocol) -> list[StructuredTool]:
    """动态发现 MCP 工具并批量转换成 StructuredTool。"""
    tools_result = await client.list_tools()

    return [
        _build_langchain_tool(client, mcp_tool)
        for mcp_tool in tools_result.tools
    ]
