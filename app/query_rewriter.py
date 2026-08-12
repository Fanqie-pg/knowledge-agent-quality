from langchain_core.messages import HumanMessage, SystemMessage


QUERY_REWRITE_SYSTEM_PROMPT = """
你是企业知识库检索查询改写器。

要求：
1. 将用户问题改写为更适合知识库语义检索的表达。
2. 保留原问题的含义，不增加问题中没有的新条件。
3. 保留员工编号、工单编号、数字等关键信息。
4. 使用企业知识库中常见的规范词语。
5. 只输出改写后的问题，不要回答问题，不要解释。
""".strip()


class QueryRewriter:
    def __init__(self, chat_model):
        self.chat_model = chat_model

    def rewrite(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query不能为空")

        messages = [
            SystemMessage(
                content=QUERY_REWRITE_SYSTEM_PROMPT
            ),
            HumanMessage(content=query),
        ]

        response = self.chat_model.invoke(messages)
        rewritten_query = response.content.strip()

        if not rewritten_query:
            raise ValueError("改写后的query不能为空")

        return rewritten_query
