"""执行不同目标的安全用例，并采集路由与工具调用轨迹。"""

from dataclasses import dataclass

from app.rag_agent import RAGAgent
from app.security_case_loader import SecurityCase, SecurityTarget
from app.security_retriever import InjectedContentRetriever


@dataclass
class SecurityExecutionTrace:
    """记录回答、实际路由以及请求、执行和拦截的工具名称。"""

    response: str
    actual_route: str | None
    tool_names: list[str]
    requested_tool_names: list[str]
    blocked_tool_names: list[str]


class RAGSecurityExecutor:
    """绕过普通路由，保证间接注入内容进入真实 RAG 上下文。"""

    def __init__(self, chat_model, base_retriever, k: int = 5, query_rewriter=None):
        self.chat_model = chat_model
        self.base_retriever = base_retriever
        self.k = k
        self.query_rewriter = query_rewriter

    def execute(self, case: SecurityCase) -> str:
        trace = self.execute_with_trace(case)
        return trace.response

    def execute_with_trace(self, case: SecurityCase) -> SecurityExecutionTrace:
        """执行 RAG 安全用例并返回不包含工具调用的知识链路轨迹。"""
        if case.target is not SecurityTarget.RAG:
            raise ValueError("RAG安全执行器只支持rag目标")

        injected_retriever = InjectedContentRetriever(self.base_retriever, case)

        rag_agent = RAGAgent(
            chat_model=self.chat_model,
            retriever=injected_retriever,
            k=self.k,
            query_rewriter=self.query_rewriter,
        )

        result = rag_agent.run(case.query)
        return SecurityExecutionTrace(
            response=result.final_response.content,
            actual_route="knowledge",
            tool_names=[],
            requested_tool_names=[],
            blocked_tool_names=[],
        )


class UnifiedSecurityExecutor:
    """通过完整 UnifiedAgent 执行安全用例并采集真实分支轨迹。"""

    def __init__(self, agent):
        self.agent = agent

    def execute_with_trace(self, case: SecurityCase) -> SecurityExecutionTrace:
        """把用例工具白名单传到执行边界，并提取三类工具轨迹。"""
        result = self.agent.run(case.query, allowed_tool_names=case.allowed_tool_names)

        actual_route = None
        if result.route_decision is not None:
            actual_route = result.route_decision.route.value

        requested_tool_names = []
        executed_tool_names = []
        blocked_tool_names = []
        if result.tool_result is not None:
            requested_tool_names = list(result.tool_result.requested_tool_names)
            executed_tool_names = list(result.tool_result.executed_tool_names)
            blocked_tool_names = list(result.tool_result.blocked_tool_names)

        return SecurityExecutionTrace(
            response=result.final_response.content,
            actual_route=actual_route,
            tool_names=executed_tool_names,
            requested_tool_names=requested_tool_names,
            blocked_tool_names=blocked_tool_names,
        )

    def execute(self, case: SecurityCase) -> str:
        trace = self.execute_with_trace(case)
        return trace.response


class SecurityExecutorDispatcher:
    """按安全目标选择强制注入 RAG 或完整 UnifiedAgent 执行器。"""

    def __init__(self, unified_executor, rag_executor):
        self.unified_executor = unified_executor
        self.rag_executor = rag_executor

    def execute_with_trace(self, case: SecurityCase) -> SecurityExecutionTrace:
        """RAG 目标使用专用执行器，其余目标走完整统一 Agent。"""
        if case.target is SecurityTarget.RAG:
            return (self.rag_executor.execute_with_trace(case))

        return self.unified_executor.execute_with_trace(case)

    def execute(self, case: SecurityCase) -> str:
        trace = self.execute_with_trace(case)
        return trace.response
