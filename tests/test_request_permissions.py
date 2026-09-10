"""Request identities must stay aligned across model decisions and execution."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest
from langchain_core.messages import AIMessage

from app.identity import load_user_context
from app.mcp_tool_registry import build_mcp_tools
from app.tool_agent import ToolAgent
from app.tool_permissions import resolve_permissions
from app.tool_registry import build_tools


pytestmark = pytest.mark.integration


class ContextModel:
    def __init__(self, barrier=None):
        self.inputs = []
        self.barrier = barrier

    def bind_tools(self, tools):
        self.schemas = [tool.name for tool in tools]
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        if any(message.type == "tool" for message in messages):
            return AIMessage(content=messages[-1].content)
        context = json.loads(messages[0].content.split("可信权限：\n")[1])
        if self.barrier:
            self.barrier.wait(timeout=10)
        query = messages[-1].content
        if query == "calc":
            name, args = "calculate", {"operation": "multiply", "a": 12, "b": 8}
        else:
            name = "get_leave_balance"
            args = {"employee_id": "E002" if query == "other" else (context["identity"] or {}).get("employee_id", "E001")}
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "proposal", "type": "tool_call"}])


@pytest.mark.parametrize("actor,names,scope", [
    (None, {"calculate"}, set()),
    ("employee-e001", {"calculate", "get_leave_balance", "create_ticket", "get_ticket"}, {"E001"}),
    ("hr", {"calculate", "get_leave_balance", "create_ticket", "get_ticket"}, {"E001", "E002", "E003", "E004"}),
])
def test_permission_matrix_is_explicit(actor, names, scope):
    snapshot = resolve_permissions(load_user_context(actor), ["calculate", "get_leave_balance", "create_ticket", "get_ticket", "unmapped"])
    assert set(snapshot.allowed_tool_names) == names
    assert set(snapshot.leave_employee_ids) == scope
    assert "unmapped" not in snapshot.allowed_tool_names


@pytest.mark.parametrize("transport", ["local", "mcp"])
def test_one_agent_serves_multiple_identities_including_explicit_anonymous(seeded_business_repository, transport):
    repo = seeded_business_repository
    model = ContextModel()
    def factory(user):
        return build_tools(user_context=user, repository=repo) if transport == "local" else build_mcp_tools(user_context=user, database_path=repo.path)
    agent = ToolAgent(model, build_tools(repository=repo), tools_factory=factory, user_context=load_user_context("hr"))
    for actor, query, expected in [("employee-e001", "self", "5"), ("hr", "other", "0"),
                                    (None, "self", None), ("employee-e002", "self", "0")]:
        result = agent.run(query, user_context=load_user_context(actor))
        assert result.permission_snapshot["actor"] == actor
        if expected is None:
            assert result.executed_tool_names == [] and result.execution_records[0]["status"] == "blocked"
        else:
            assert float(result.tool_messages[0].content) == float(expected)
    assert agent.user_context.principal_id == "hr"  # unchanged legacy default, not the latest request
    assert set(model.schemas) == {"calculate", "get_leave_balance", "create_ticket", "get_ticket"}
    with pytest.raises(AttributeError):
        agent.user_context = None


def test_concurrent_contexts_do_not_mutate_tool_bindings(seeded_business_repository):
    repo = seeded_business_repository
    model = ContextModel(threading.Barrier(4))
    agent = ToolAgent(model, build_tools(repository=repo), tools_factory=lambda user: build_tools(user_context=user, repository=repo))
    actors = ["employee-e001", "hr", None, "employee-e002"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(agent.run, "self", user_context=load_user_context(actor)) for actor in actors]
        results = [future.result() for future in futures]
    assert [r.permission_snapshot["actor"] for r in results] == actors
    assert [float(r.tool_messages[0].content) if r.executed_tool_names else None for r in results] == [5, 3, None, 0]
    assert agent.user_context is None


def test_decision_is_real_model_stage_without_building_backend_tools(seeded_business_repository):
    model = ContextModel()
    def forbidden_factory(user):
        raise AssertionError("纯决策不能创建或执行后端工具")
    agent = ToolAgent(model, build_tools(repository=seeded_business_repository), tools_factory=forbidden_factory)
    decision = agent.decide("other", user_context=load_user_context("employee-e001"))
    assert decision.tool_calls[0]["args"]["employee_id"] == "E002"  # bad proposal is retained, not filtered
    context = json.loads(model.inputs[0][0].content.split("可信权限：\n")[1])
    assert context["leave_employee_ids"] == ["E001"] and context["permission_source"] == "trusted_identity"
    assert len(model.inputs) == 1


def test_new_request_cannot_override_permissions(seeded_business_repository):
    agent = ToolAgent(ContextModel(), build_tools(repository=seeded_business_repository),
                      tools_factory=lambda user: build_tools(user_context=user, repository=seeded_business_repository))
    with pytest.raises(ValueError, match="不能手工覆盖"):
        agent.run("calc", user_context=None, allowed_tool_names=["get_leave_balance"])


def test_mismatched_backend_identity_rejected_before_execution(seeded_business_repository):
    repo = seeded_business_repository
    agent = ToolAgent(ContextModel(), build_tools(repository=repo),
                      tools_factory=lambda user: build_tools(user_context=load_user_context("hr"), repository=repo))
    with pytest.raises(ValueError, match="绑定身份不一致"):
        agent.run("self", user_context=load_user_context("employee-e001"))


def test_fixed_identity_closure_cannot_be_reused_for_another_user(seeded_business_repository):
    user = load_user_context("employee-e001")
    agent = ToolAgent(ContextModel(), build_tools(repository=seeded_business_repository, user_context=user), user_context=user)
    with pytest.raises(ValueError, match="固定身份工具不能切换"):
        agent.run("self", user_context=load_user_context("hr"))
