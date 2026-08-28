import asyncio
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
    )

    async with Client(server_parameters) as client:
        tools_result = await client.list_tools()

        print("MCP工具列表：")
        for tool in tools_result.tools:
            print(f"名称：{tool.name}")
            print(f"描述：{tool.description}")
            print(f"参数Schema：{tool.input_schema}")
            print("-" * 60)

        multiply_result = await client.call_tool("multiply_numbers", {"a": 12, "b": 8})

        print("乘法调用是否失败：", multiply_result.is_error)
        print("乘法文本内容：", multiply_result.content)
        print("乘法结构化结果：", multiply_result.structured_content)

        leave_result = await client.call_tool(
            "get_leave_balance",
            {"employee_id": "E001"},
        )

        print("年假调用是否失败：", leave_result.is_error)
        print("年假文本内容：", leave_result.content)
        print("年假结构化结果：", leave_result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())