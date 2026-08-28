import asyncio

from mcp.server import MCPServer


mcp = MCPServer(name="slow-test-server")


@mcp.tool(name="slow_tool")
async def slow_tool(delay_seconds: float) -> str:
    """模拟执行时间较长的工具。"""
    await asyncio.sleep(delay_seconds)
    return "执行完成"


if __name__ == "__main__":
    mcp.run("stdio")