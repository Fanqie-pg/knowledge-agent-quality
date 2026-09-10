"""A correct tool call must not become a wrong numeric business answer."""
import pytest
from langchain_core.messages import AIMessage

from app.identity import load_user_context
from app.tool_agent import ToolAgent
from app.tool_registry import build_tools
from app.tracing import capture_trace


pytestmark = pytest.mark.unit


class WrongFinalModel:
    def __init__(self, name, args, answer):
        self.calls = 0
        self.decision = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "scalar", "type": "tool_call"}])
        self.final = AIMessage(content=answer)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self.decision if self.calls == 1 else self.final


@pytest.mark.parametrize("name,args,wrong,actual", [
    ("get_leave_balance", {"employee_id": "E001"}, "员工E001剩余年假是7天。", "5"),
    ("get_leave_balance", {"employee_id": "E001"}, "员工E005已查询成功。", "5"),
    ("get_leave_balance", {"employee_id": "E002"}, "剩余年假3天。", "0"),
    ("calculate", {"operation": "multiply", "a": 12, "b": 8}, "计算结果是86。", "96"),
    ("calculate", {"operation": "subtract", "a": 0, "b": 5}, "计算结果是5。", "-5"),
    ("calculate", {"operation": "add", "a": 2.5, "b": 1.25}, "已经计算完成。", "3.75"),
])
def test_numeric_mistranscription_falls_back_to_actual_receipt(seeded_business_repository, name, args, wrong, actual):
    user = load_user_context("hr")
    model = WrongFinalModel(name, args, wrong)
    agent = ToolAgent(model, build_tools(user_context=user, repository=seeded_business_repository), user_context=user)
    with capture_trace() as events:
        result = agent.run("执行指定测试请求")
    assert result.requested_tool_names == [name] and result.execution_records[0]["status"] == "success"
    assert result.answer_mode == "numeric_receipt_validated"
    assert result.final_response is not model.final and actual in result.final_response.content
    assert any(e["event"] == "numeric_receipt_restored" and e["data"]["model_content"] == wrong for e in events)
    assert model.calls == 2


@pytest.mark.parametrize("name,args,answer", [
    ("get_leave_balance", {"employee_id": "E001"}, "员工E001剩余年假**5天**。"),
    ("get_leave_balance", {"employee_id": "E002"}, "剩余年假0天。"),
    ("calculate", {"operation": "multiply", "a": 12, "b": 8}, "12乘以8等于96。"),
    ("calculate", {"operation": "subtract", "a": 0, "b": 5}, "结果是-5。"),
    ("calculate", {"operation": "add", "a": 2.5, "b": 1.25}, "计算结果：3.75。"),
])
def test_correct_numeric_answer_is_preserved(seeded_business_repository, name, args, answer):
    user = load_user_context("hr")
    model = WrongFinalModel(name, args, answer)
    result = ToolAgent(model, build_tools(user_context=user, repository=seeded_business_repository), user_context=user).run("执行指定测试请求")
    assert result.answer_mode == "model" and result.final_response is model.final
