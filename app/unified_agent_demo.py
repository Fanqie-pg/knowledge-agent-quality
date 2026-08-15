from app.agent_factory import build_unified_agent


def main():
    agent = build_unified_agent()

    queries = [
        "12乘以8是多少？",
        "P1工单多久必须首次响应，处理过程中多久同步一次进展？",
        "你好，请介绍一下自己",
    ]

    for query in queries:
        result = agent.run(query)

        print(f"用户问题：{result.query}")
        print(f"路由结果：{result.route_decision.route.value}")
        print(f"路由原因：{result.route_decision.reason}")

        if result.tool_result is not None:
            print(f"工具决策：{result.tool_result.decision.tool_calls}")
            tool_outputs = [
                message.content for message in result.tool_result.tool_messages
            ]
            print(f"工具结果：{tool_outputs}")

        if result.rag_result is not None:
            print(f"检索问题：{result.rag_result.retrieval_query}")
            rag_outputs = [
                item.document.id for item in result.rag_result.retrieval_results
            ]
            print(f"召回来源：{rag_outputs}")

        print(f"最终回答：{result.final_response.content}")
        print("-" * 70)


if __name__ == "__main__":
    main()