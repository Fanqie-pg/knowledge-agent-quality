from app.agent_factory import build_unified_agent
from app.mcp_tool_registry import build_mcp_tools


def main() -> None:
    mcp_tools = build_mcp_tools()
    agent = build_unified_agent(tools=mcp_tools,)

    print(
        "已加载MCP工具：",
        [
            {
                "name": tool.name,
                "source": tool.metadata["source"],
            }
            for tool in mcp_tools
        ],
    )
    print("=" * 70)

    queries = [
        "12乘以8是多少？",
        "P1工单多久必须首次响应？",
        "你好，请介绍一下自己。",
    ]

    for query in queries:
        result = agent.run(query)

        print(f"用户问题：{result.query}")
        print("路由结果：", result.route_decision.route.value)
        print("路由原因：", result.route_decision.reason)

        if result.tool_result is not None:
            print("模型请求工具：", result.tool_result.requested_tool_names)
            print("实际执行工具：", result.tool_result.executed_tool_names)
            print(
                "MCP工具结果：",
                [
                    {
                        "name": message.name,
                        "status": message.status,
                        "content": message.content,
                    }
                    for message in result.tool_result.tool_messages
                ],
            )

        if result.rag_result is not None:
            print("检索问题：", result.rag_result.retrieval_query)
            print(
                "召回来源：",
                [
                    item.document.id
                    for item
                    in result.rag_result.retrieval_results
                ],
            )

        print("最终回答：", result.final_response.content)
        print("-" * 70)


if __name__ == "__main__":
    main()