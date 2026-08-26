from dataclasses import dataclass

from app.rag_agent import RAGAgent
from app.security_case_loader import SecurityCase, SecurityTarget
from app.security_retriever import InjectedContentRetriever


@dataclass
class SecurityExecutionTrace:
    response: str
    actual_route: str | None
    tool_names: list[str]
    requested_tool_names: list[str]
    blocked_tool_names: list[str]


class RAGSecurityExecutor:
    def __init__(self, chat_model, base_retriever, k: int = 5, query_rewriter=None):
        self.chat_model = chat_model
        self.base_retriever = base_retriever
        self.k = k
        self.query_rewriter = query_rewriter

    def execute(self, case: SecurityCase) -> str:
        trace = self.execute_with_trace(case)
        return trace.response

    def execute_with_trace(self, case: SecurityCase) -> SecurityExecutionTrace:
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

    def __init__(self, agent):
        self.agent = agent

    def execute_with_trace(self, case: SecurityCase) -> SecurityExecutionTrace:
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

    def __init__(self, unified_executor, rag_executor):
        self.unified_executor = unified_executor
        self.rag_executor = rag_executor

    def execute_with_trace(self, case: SecurityCase) -> SecurityExecutionTrace:
        if case.target is SecurityTarget.RAG:
            return (self.rag_executor.execute_with_trace(case))

        return self.unified_executor.execute_with_trace(case)

    def execute(self, case: SecurityCase) -> str:
        trace = self.execute_with_trace(case)
        return trace.response