from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.intent_router import AgentRoute, IntentRouter, RouteDecision
from app.rag_agent import RAGAgent, RAGRunResult
from app.tool_agent import AgentRunResult, ToolAgent


CHAT_SYSTEM_PROMPT = """
你是企业智能助手。

请自然、简洁地回答普通问候、闲聊和自我介绍。
不要声称已经调用工具或查询知识库。
""".strip()


@dataclass
class UnifiedAgentRunResult:
    query: str
    route_decision: RouteDecision
    final_response: AIMessage
    tool_result: AgentRunResult | None = None
    rag_result: RAGRunResult | None = None


class UnifiedAgent:

    def __init__(self, router: IntentRouter, tool_agent: ToolAgent, rag_agent: RAGAgent, chat_model):
        self.router = router
        self.tool_agent = tool_agent
        self.rag_agent = rag_agent
        self.chat_model = chat_model

    def run(self, query: str) -> UnifiedAgentRunResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query不能为空")

        route_decision = self.router.route(query)

        if route_decision.route is AgentRoute.TOOL:
            tool_result = self.tool_agent.run(query)
            return UnifiedAgentRunResult(
                query=query,
                route_decision=route_decision,
                final_response=tool_result.final_response,
                tool_result=tool_result,
            )

        if route_decision.route is AgentRoute.KNOWLEDGE:
            rag_result = self.rag_agent.run(query)

            return UnifiedAgentRunResult(
                query=query,
                route_decision=route_decision,
                final_response=rag_result.final_response,
                rag_result=rag_result,
            )

        if route_decision.route is AgentRoute.CHAT:
            messages = [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=query),
            ]
            final_response = self.chat_model.invoke(messages)

            return UnifiedAgentRunResult(
                query=query,
                route_decision=route_decision,
                final_response=final_response,
            )

        raise RuntimeError(
            f"不支持的路由：{route_decision.route}"
        )