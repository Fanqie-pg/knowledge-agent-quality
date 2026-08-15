from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.intent_router import AgentRoute, RouteDecision
from app.unified_agent import UnifiedAgent


pytestmark = pytest.mark.unit


class FakeRouter:

    def __init__(self, route):
        self.decision = RouteDecision(route=route, reason="测试路由")
        self.received_queries = []

    def route(self, query):
        self.received_queries.append(query)
        return self.decision


class FakeExecutionAgent:

    def __init__(self, answer):
        self.received_queries = []
        self.result = SimpleNamespace(
            final_response=AIMessage(content=answer)
        )

    def run(self, query):
        self.received_queries.append(query)
        return self.result


class FakeChatModel:

    def __init__(self):
        self.received_messages = []
        self.response = AIMessage(content="你好！")

    def invoke(self, messages):
        self.received_messages.append(messages)
        return self.response


def build_unified_agent(route):
    router = FakeRouter(route)
    tool_agent = FakeExecutionAgent("工具回答")
    rag_agent = FakeExecutionAgent("知识库回答")
    chat_model = FakeChatModel()

    agent = UnifiedAgent(
        router=router,
        tool_agent=tool_agent,
        rag_agent=rag_agent,
        chat_model=chat_model,
    )

    return agent, router, tool_agent, rag_agent, chat_model


def test_run_routes_tool_query_to_tool_agent():
    agent, router, tool_agent, rag_agent, chat_model = build_unified_agent(AgentRoute.TOOL)

    result = agent.run("12乘以8是多少？")

    assert result.route_decision.route is AgentRoute.TOOL
    assert result.tool_result is tool_agent.result
    assert result.rag_result is None
    assert result.final_response.content == "工具回答"

    assert router.received_queries == ["12乘以8是多少？"]
    assert tool_agent.received_queries == ["12乘以8是多少？"]
    assert rag_agent.received_queries == []
    assert chat_model.received_messages == []


def test_run_routes_knowledge_query_to_rag_agent():
    agent, router, tool_agent, rag_agent, chat_model = build_unified_agent(AgentRoute.KNOWLEDGE)

    result = agent.run("P1故障必须多长时间内响应？")

    assert result.route_decision.route is AgentRoute.KNOWLEDGE
    assert result.tool_result is None
    assert result.rag_result is rag_agent.result
    assert result.final_response.content == "知识库回答"

    assert router.received_queries == ["P1故障必须多长时间内响应？"]
    assert tool_agent.received_queries == []
    assert rag_agent.received_queries == ["P1故障必须多长时间内响应？"]
    assert chat_model.received_messages == []


def test_run_routes_chat_query_to_chat_agent():
    agent, router, tool_agent, rag_agent, chat_model = build_unified_agent(AgentRoute.CHAT)

    result = agent.run("天空是什么颜色？")

    assert result.route_decision.route is AgentRoute.CHAT
    assert result.tool_result is None
    assert result.rag_result is None
    assert result.final_response.content == "你好！"

    assert router.received_queries == ["天空是什么颜色？"]
    assert tool_agent.received_queries == []
    assert rag_agent.received_queries == []
    assert len(chat_model.received_messages) == 1

    messages = chat_model.received_messages[0]

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "天空是什么颜色？" in  messages[1].content


@pytest.mark.parametrize(
    "invalid_query",
    ["", " ", "\t", "\n", None],
)
def test_run_rejects_empty_query(invalid_query):
    agent, router, tool_agent, rag_agent, chat_model = build_unified_agent(AgentRoute.KNOWLEDGE)

    with pytest.raises(ValueError, match="query不能为空"):
        agent.run(invalid_query)

    assert router.received_queries == []
    assert tool_agent.received_queries == []
    assert rag_agent.received_queries == []
    assert chat_model.received_messages == []