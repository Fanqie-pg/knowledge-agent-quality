import pytest
from langchain_core.tools import StructuredTool

import app.agent_factory as agent_factory


pytestmark = pytest.mark.unit


class FakeChatModel:

    def __init__(self):
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self


def injected_tool(value: str) -> str:
    """用于验证工具依赖注入。"""
    return value


def test_build_unified_agent_uses_injected_tools(monkeypatch):
    chat_model = FakeChatModel()
    tool = StructuredTool.from_function(func=injected_tool)

    monkeypatch.setattr(agent_factory, "build_chat_model", lambda: chat_model)
    monkeypatch.setattr(agent_factory, "build_knowledge_retriever", lambda: object())

    agent = agent_factory.build_unified_agent(tools=[tool])
    assert set(agent.tool_agent.tools_by_name) == {"injected_tool"}
    assert agent.tool_agent.tools_by_name["injected_tool"] is tool
    assert chat_model.bound_tools == [tool]