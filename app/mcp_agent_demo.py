import asyncio
import sys
from pathlib import Path

from mcp.client.stdio import StdioServerParameters

from app.agent_factory import build_chat_model
from app.mcp_client import MCPToolClient
from app.mcp_langchain_adapter import (build_mcp_langchain_tools)
from app.tool_agent import ToolAgent


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_mcp_tools():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=PROJECT_ROOT,
    )
    client = MCPToolClient(parameters, read_timeout_seconds=10)
    return asyncio.run(build_mcp_langchain_tools(client))


def main() -> None:
    chat_model = build_chat_model()
    tools = build_mcp_tools()
    agent = ToolAgent(chat_model, tools)

    questions = [
        "请使用工具计算12乘以8。",
        "请查询员工E001的剩余年假。",
        "请查询员工E999的剩余年假。",
    ]

    print(
        "MCP工具：",
        [
            {
                "name": tool.name,
                "source": tool.metadata["source"],
            }
            for tool in tools
        ],
    )
    print("=" * 70)

    for question in questions:
        result = agent.run(question)

        print("用户问题：", question)
        print("模型请求工具：", result.requested_tool_names)
        print("实际执行工具：", result.executed_tool_names)
        print(
            "工具结果：",
            [
                {
                    "name": message.name,
                    "status": message.status,
                    "content": message.content,
                }
                for message in result.tool_messages
            ],
        )
        print("最终回答：", result.final_response.content)
        print("-" * 70)


if __name__ == "__main__":
    main()