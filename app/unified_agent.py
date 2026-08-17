import logging
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.intent_router import AgentRoute, IntentRouter, RouteDecision, InvalidRouteResponseError
from app.rag_agent import RAGAgent, RAGRunResult
from app.retriever import RetrievalUnavailableError
from app.tool_agent import AgentRunResult, ToolAgent
from app.resilient_model import ModelUnavailableError


logger = logging.getLogger(__name__)


class AgentRunStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"


class AgentFailureStage(str, Enum):
    ROUTING = "routing"
    TOOL = "tool"
    KNOWLEDGE = "knowledge"
    CHAT = "chat"


class AgentFailureCode(str, Enum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INVALID_ROUTE_RESPONSE = "INVALID_ROUTE_RESPONSE"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"


@dataclass
class AgentFailure:
    stage: AgentFailureStage
    code: AgentFailureCode
    message: str


RETRIEVAL_UNAVAILABLE_RESPONSE = "知识库检索服务暂时不可用，请稍后重试。"

INVALID_ROUTE_RESPONSE_MESSAGE = "系统暂时无法识别请求类型，请稍后重试。"

MODEL_UNAVAILABLE_RESPONSE = "系统暂时无法处理请求，请稍后重试。"

CHAT_SYSTEM_PROMPT = """
你是企业智能助手。

请自然、简洁地回答普通问候、闲聊和自我介绍。
不要声称已经调用工具或查询知识库。
""".strip()


@dataclass
class UnifiedAgentRunResult:
    query: str
    route_decision: RouteDecision | None
    final_response: AIMessage
    status: AgentRunStatus = AgentRunStatus.SUCCESS
    tool_result: AgentRunResult | None = None
    rag_result: RAGRunResult | None = None
    failure: AgentFailure | None = None


def _build_degraded_result(
    query: str,
    route_decision: RouteDecision | None,
    stage: AgentFailureStage,
    code: AgentFailureCode,
    message: str,
    error: Exception,
) -> UnifiedAgentRunResult:
    logger.error(
        "统一Agent执行降级："
        "stage=%s, code=%s, error_type=%s",
        stage.value,
        code.value,
        type(error).__name__,
    )

    return UnifiedAgentRunResult(
        query=query,
        route_decision=route_decision,
        final_response=AIMessage(content=message),
        status=AgentRunStatus.DEGRADED,
        failure=AgentFailure(
            stage=stage,
            code=code,
            message=message,
        ),
    )


class UnifiedAgent:

    def __init__(self, router: IntentRouter, tool_agent: ToolAgent, rag_agent: RAGAgent, chat_model):
        self.router = router
        self.tool_agent = tool_agent
        self.rag_agent = rag_agent
        self.chat_model = chat_model

    def run(self, query: str) -> UnifiedAgentRunResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query不能为空")

        try:
            route_decision = self.router.route(query)
        except ModelUnavailableError as exc:
            return _build_degraded_result(
                query=query,
                route_decision=None,
                stage=AgentFailureStage.ROUTING,
                code=AgentFailureCode.MODEL_UNAVAILABLE,
                message=MODEL_UNAVAILABLE_RESPONSE,
                error=exc,
            )
        except InvalidRouteResponseError as exc:
            return _build_degraded_result(
                query=query,
                route_decision=None,
                stage=AgentFailureStage.ROUTING,
                code=AgentFailureCode.INVALID_ROUTE_RESPONSE,
                message=INVALID_ROUTE_RESPONSE_MESSAGE,
                error=exc,
            )

        if route_decision.route is AgentRoute.TOOL:
            try:
                tool_result = self.tool_agent.run(query)
            except ModelUnavailableError as exc:
                return _build_degraded_result(
                    query=query,
                    route_decision=route_decision,
                    stage=AgentFailureStage.TOOL,
                    code=AgentFailureCode.MODEL_UNAVAILABLE,
                    message=MODEL_UNAVAILABLE_RESPONSE,
                    error=exc,
                )

            return UnifiedAgentRunResult(
                query=query,
                route_decision=route_decision,
                final_response=tool_result.final_response,
                tool_result=tool_result,
            )

        if route_decision.route is AgentRoute.KNOWLEDGE:
            try:
                rag_result = self.rag_agent.run(query)
            except ModelUnavailableError as exc:
                return _build_degraded_result(
                    query=query,
                    route_decision=route_decision,
                    stage=AgentFailureStage.KNOWLEDGE,
                    code=AgentFailureCode.MODEL_UNAVAILABLE,
                    message=MODEL_UNAVAILABLE_RESPONSE,
                    error=exc,
                )
            except RetrievalUnavailableError as exc:
                return _build_degraded_result(
                    query=query,
                    route_decision=route_decision,
                    stage=AgentFailureStage.KNOWLEDGE,
                    code=AgentFailureCode.RETRIEVAL_UNAVAILABLE,
                    message=RETRIEVAL_UNAVAILABLE_RESPONSE,
                    error=exc,
                )

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
            try:
                final_response = self.chat_model.invoke(messages)
            except ModelUnavailableError as exc:
                return _build_degraded_result(
                    query=query,
                    route_decision=route_decision,
                    stage=AgentFailureStage.CHAT,
                    code=AgentFailureCode.MODEL_UNAVAILABLE,
                    message=MODEL_UNAVAILABLE_RESPONSE,
                    error=exc,
                )
            return UnifiedAgentRunResult(
                query=query,
                route_decision=route_decision,
                final_response=final_response,
            )

        raise RuntimeError(
            f"不支持的路由：{route_decision.route}"
        )