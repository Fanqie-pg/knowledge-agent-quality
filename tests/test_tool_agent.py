from langchain_core.messages import AIMessage

from app.tool_agent import ToolAgent
from app.tool_registry import build_tools


class FakeToolModel:
    def __init__(self):
        self.bound_tools = None
        self.last_prompt = None
        self.response = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "multiply_numbers",
                    "args": {"a": 12, "b": 8},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        self.final_response = AIMessage(
            content="12乘以8等于96。",
            tool_calls=[]
        )
        self.received_inputs = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, prompt):
        self.last_prompt = prompt
        self.received_inputs.append(prompt)
        if len(self.received_inputs) == 1:
            return self.response
        else:
            return self.final_response


def test_decide_binds_tools_and_returns_tool_call():
    tools = build_tools()
    fake_model = FakeToolModel()
    agent = ToolAgent(fake_model, tools)
    result = agent.decide("12乘以8是多少？")

    assert fake_model.bound_tools == tools
    assert fake_model.last_prompt == "12乘以8是多少？"
    assert result is fake_model.response
    assert result.tool_calls[0]["name"] == "multiply_numbers"
    assert result.tool_calls[0]["args"] == {"a": 12, "b": 8}


def test_execute_tool_calls_returns_correlated_tool_message():
    tools = build_tools()
    fake_model = FakeToolModel()
    agent = ToolAgent(fake_model, tools)
    decision = agent.decide("12乘以8是多少？")
    tool_messages = agent.execute_tool_calls(decision)
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "96.0"
    assert tool_messages[0].tool_call_id == "call-1"
    assert tool_messages[0].name == "multiply_numbers"

def test_run_executes_tool_and_returns_final_answer():
    tools = build_tools()
    fake_model = FakeToolModel()
    agent = ToolAgent(fake_model, tools)
    result = agent.run("12乘以8是多少？")

    assert result.decision is fake_model.response
    assert result.final_response is fake_model.final_response
    assert result.final_response.content == "12乘以8等于96。"
    assert result.final_response.tool_calls == []
    assert len(result.tool_messages) == 1
    assert result.tool_messages[0].content == "96.0"
    assert result.tool_messages[0].name == "multiply_numbers"
    assert result.tool_messages[0].tool_call_id == "call-1"

    assert len(fake_model.received_inputs) == 2
    second_input = fake_model.received_inputs[1]
    assert len(second_input) == 3
    assert second_input[0].content == "12乘以8是多少？"
    assert second_input[1].tool_calls[0]["name"] == "multiply_numbers"
    assert second_input[2].content == "96.0"
    assert second_input[2].tool_call_id == "call-1"


def test_run_returns_direct_answer_without_tool_call():
    tools = build_tools()
    fake_model = FakeToolModel()
    agent = ToolAgent(fake_model, tools)
    direct_response = AIMessage(
        content="你好，我是企业知识助手。",
        tool_calls=[],
    )
    fake_model.response = direct_response
    result = agent.run("你好，请介绍一下自己")

    assert result.decision is direct_response
    assert result.tool_messages == []
    assert result.final_response is direct_response
    assert result.final_response.content == "你好，我是企业知识助手。"
    assert result.final_response.tool_calls == []
    assert len(fake_model.received_inputs) == 1


def test_execute_tool_calls_converts_business_error_to_tool_message():
    decision = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_leave_balance",
                "args": {"employee_id": "E999"},
                "id": "call-error-1",
                "type": "tool_call",
            }
        ],
    )
    tools = build_tools()
    fake_model = FakeToolModel()
    agent = ToolAgent(fake_model, tools)
    tool_messages = agent.execute_tool_calls(decision)

    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert tool_messages[0].name == "get_leave_balance"
    assert tool_messages[0].tool_call_id == "call-error-1"
    assert "员工不存在" in tool_messages[0].content