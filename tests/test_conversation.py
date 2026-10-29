"""Stateful assertions, actual SQLite side effects, and raw model proposals."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage
import pytest

from app.business_errors import BusinessError
from app.conversation_agent import ConversationAgent
from app.conversation_store import ConversationStore
from app.conversation_tools import run_tool_loop
from app.identity import load_user_context, Role
from app.tool_agent import ToolAgent
from app.tool_registry import build_tools


EMPLOYEE = load_user_context("employee-e001")
OTHER = load_user_context("employee-e002")
HR = load_user_context("hr")
PAYLOAD = {"title": "VPN连接故障", "description": "连接超时无法办公", "priority": "P2"}


def plan(route="tool", query="当前任务"):
    return AIMessage(content=json.dumps({"route": route, "standalone_query": query}, ensure_ascii=False))


def call(name, args, id="call1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id}])


def calc(a=12, b=8, operation="multiply", id="call1"):
    return call("calculate", {"operation": operation, "a": a, "b": b}, id)


class ScriptedModel:
    def __init__(self, responses):
        self.responses, self.inputs = list(responses), []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.inputs.append(messages)
        assert self.responses, "模型调用次数超出预期"
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value(messages) if callable(value) else value


def engine(tmp_path, repo, responses, **limits):
    model = ScriptedModel(responses)
    tools = ToolAgent(model, build_tools(repository=repo),
                      tools_factory=lambda ctx: build_tools(user_context=ctx, repository=repo))
    unified = SimpleNamespace(chat_model=model, tool_agent=tools, rag_agent=None)
    service = ConversationAgent(unified, ConversationStore(tmp_path / "conversation.sqlite3"), repo, **limits)
    return service, model


def turn(service, query="执行当前任务", sid="s1", tid="t1", ctx=EMPLOYEE):
    return service.run(query, session_id=sid, turn_id=tid, user_context=ctx)


def ticket_count(repo):
    with repo.connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]


def last_ticket_answer(messages):
    receipt = json.loads([m for m in messages if isinstance(m, ToolMessage)][-1].content)
    return AIMessage(content="工单创建成功：" + receipt["ticket_id"])


def test_followup_uses_verified_numeric_fact_and_only_current_session(tmp_path, seeded_business_repository):
    service, model = engine(tmp_path, seeded_business_repository,
                            [plan(), calc(), AIMessage(content="96"), plan(), calc(96, 2), AIMessage(content="192"),
                             plan(), AIMessage(content="请问要将哪个数乘以2？")])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    assert turn(service, "计算12乘8").answer == "96"
    followup = turn(service, "再乘2", tid="t2")
    assert followup.answer == "192"
    prompt = model.inputs[4][0].content
    assert '"value": 96.0' in prompt
    service.create_session(user_context=EMPLOYEE, session_id="s2")
    assert not turn(service, "再乘2", sid="s2").execution_records
    assert '"value": 96.0' not in model.inputs[-1][0].content


def test_dependent_native_tool_calls_use_actual_first_result(tmp_path, seeded_business_repository):
    def dependent(messages):
        actual = float([m for m in messages if isinstance(m, ToolMessage)][-1].content)
        return calc(actual, 3, "divide", "call2")
    service, _ = engine(tmp_path, seeded_business_repository, [plan(), calc(), dependent, AIMessage(content="32")])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service)
    assert result.answer == "32"
    assert [r["model_step"] for r in result.execution_records] == [1, 2]
    assert result.execution_records[1]["arguments"]["a"] == 96


def test_clarification_history_then_exact_write_and_receipt_followup(tmp_path, seeded_business_repository):
    def read_actual(messages):
        facts = messages[0].content
        ticket = seeded_business_repository.recover_own_request(EMPLOYEE.employee_id, key_from_db())
        assert ticket["ticket_id"] in facts
        return call("get_ticket", {"ticket_id": ticket["ticket_id"]})
    def key_from_db():
        with seeded_business_repository.connection() as conn:
            return conn.execute("SELECT request_key FROM tickets").fetchone()[0]
    service, model = engine(tmp_path, seeded_business_repository,
                            [plan(), AIMessage(content="请提供描述和优先级"), plan(), call("create_ticket", PAYLOAD),
                             last_ticket_answer, plan(), read_actual, last_ticket_answer])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    turn(service, "帮我创建工单，标题VPN连接故障")
    assert ticket_count(seeded_business_repository) == 0
    result = turn(service, "描述连接超时无法办公，优先级P2，请创建", tid="t2")
    assert ticket_count(seeded_business_repository) == 1
    assert "标题VPN连接故障" in model.inputs[3][1].content
    assert result.execution_records[0]["committed_write"]
    read = turn(service, "查询刚才的工单", tid="t3")
    assert read.execution_records[0]["protected_read"]


@pytest.mark.parametrize("ctx", [OTHER, HR, None])
def test_owner_boundary_blocks_before_any_model_call(tmp_path, seeded_business_repository, ctx):
    service, model = engine(tmp_path, seeded_business_repository, [])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    with pytest.raises(BusinessError) as error:
        turn(service, ctx=ctx)
    assert error.value.code in {"SESSION_NOT_ACCESSIBLE", "AUTHENTICATION_REQUIRED"}
    assert not model.inputs
    with pytest.raises(BusinessError):
        service.create_session(user_context=ctx, session_id="s1")


def test_scope_change_discards_history_and_facts_and_denies_old_cache(tmp_path, seeded_business_repository):
    service, model = engine(tmp_path, seeded_business_repository,
                            [plan(), call("get_leave_balance", {"employee_id": "E003"}), AIMessage(content="12天"),
                             plan(), AIMessage(content="没有权限查询E003")])
    service.create_session(user_context=HR, session_id="s1")
    turn(service, "查询E003年假", ctx=HR)
    changed = replace(HR, role=Role.EMPLOYEE, employee_scope=frozenset())
    assert service.store.inspect("s1", changed)["facts"] == {}
    result = turn(service, "再查E003", tid="t2", ctx=changed)
    assert not result.execution_records
    assert '"value": 12.0' not in model.inputs[-1][0].content
    assert model.inputs[-1][1].content.endswith("[]")
    with pytest.raises(BusinessError, match="权限已变化"):
        turn(service, "查询E003年假", ctx=changed)


def test_same_turn_replay_has_no_calls_new_turn_explicitly_creates_again(tmp_path, seeded_business_repository):
    service, model = engine(tmp_path, seeded_business_repository,
                            [plan(), call("create_ticket", PAYLOAD), last_ticket_answer,
                             plan(), call("create_ticket", PAYLOAD), last_ticket_answer])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    first = turn(service)
    replay = turn(service)
    assert replay.cached and replay.answer == first.answer
    assert replay.decisions == replay.execution_records == [] and replay.model_steps == 0
    assert ticket_count(seeded_business_repository) == 1
    with pytest.raises(BusinessError) as error:
        turn(service, "修改本轮内容")
    assert error.value.code == "TURN_CONTENT_CONFLICT"
    turn(service, "同样内容再建一张", tid="t2")
    assert ticket_count(seeded_business_repository) == 2 and len(model.inputs) == 6


def test_native_create_then_get_real_ticket_id(tmp_path, seeded_business_repository):
    def read(messages):
        receipt = json.loads([m for m in messages if isinstance(m, ToolMessage)][-1].content)
        return call("get_ticket", {"ticket_id": receipt["ticket_id"]}, "call2")
    service, _ = engine(tmp_path, seeded_business_repository,
                        [plan(), call("create_ticket", PAYLOAD), read, last_ticket_answer])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service)
    assert [r["model_step"] for r in result.execution_records] == [1, 2]
    assert result.execution_records[1]["arguments"]["ticket_id"] == result.execution_records[0]["write_receipt"]["ticket_id"]


@pytest.mark.parametrize("limit, responses, reason", [
    ({"max_steps": 1}, [calc()], "model_step_limit"),
    ({"max_tool_calls": 1}, [calc(), calc(96, 2, id="call2")], "tool_call_limit"),
    ({}, [calc(), calc(id="call2")], "repeated_call"),
    ({}, [call("create_ticket", PAYLOAD), call("create_ticket", {**PAYLOAD, "title": "另一个VPN故障"}, "call2")], "write_limit"),
    ({}, [calc(), RuntimeError("provider outage")], "model_error"),
    ({}, [call("create_ticket", PAYLOAD), RuntimeError("provider outage")], "model_error"),
])
def test_bounded_loop_stops_and_preserves_completed_receipts(tmp_path, seeded_business_repository, limit, responses, reason):
    service, _ = engine(tmp_path, seeded_business_repository, [plan(), *responses], **limit)
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service)
    assert result.stop_reason == reason and result.status == "partial"
    assert len(result.execution_records) == 1
    assert "未完整完成" in result.answer
    assert len(result.decisions) <= service.max_steps


def test_business_denial_stops_no_retry_and_keeps_raw_proposal(tmp_path, seeded_business_repository):
    service, model = engine(tmp_path, seeded_business_repository,
                            [plan(), call("get_leave_balance", {"employee_id": "E003"})])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service)
    assert result.stop_reason == "tool_error" and len(model.inputs) == 2
    assert result.decisions[0]["tool_calls"][0]["args"]["employee_id"] == "E003"
    assert result.execution_records[0]["error_code"] == "ACCESS_DENIED"
    assert not result.execution_records[0]["protected_read"]
    assert service.store.inspect("s1", EMPLOYEE)["facts"] == {}


def test_new_turn_reads_fresh_state_original_turn_replay_is_snapshot(tmp_path, seeded_business_repository):
    service, _ = engine(tmp_path, seeded_business_repository,
                        [plan(), call("get_leave_balance", {"employee_id": "E001"}), AIMessage(content="5天"),
                         plan(), call("get_leave_balance", {"employee_id": "E001"}), AIMessage(content="2天")])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    first = turn(service)
    with seeded_business_repository.connection() as conn:
        conn.execute("UPDATE employees SET leave_balance=2 WHERE employee_id='E001'")
    assert turn(service).answer == first.answer == "5天"
    fresh = turn(service, "再查一下我的年假", tid="t2")
    assert fresh.answer == "2天" and fresh.execution_records[0]["protected_read"]


def test_restart_reads_durable_history_and_cached_result(tmp_path, seeded_business_repository):
    first, _ = engine(tmp_path, seeded_business_repository, [plan(), calc(), AIMessage(content="96")])
    first.create_session(user_context=EMPLOYEE, session_id="s1")
    turn(first)
    resumed, model = engine(tmp_path, seeded_business_repository, [plan(), calc(96, 2), AIMessage(content="192")])
    assert turn(resumed).cached
    assert turn(resumed, "再乘2", tid="t2").answer == "192"
    assert '"value": 96.0' in model.inputs[1][0].content


def test_response_lost_after_commit_recovers_without_model_or_duplicate(tmp_path, seeded_business_repository):
    service, model = engine(tmp_path, seeded_business_repository, [plan(), call("create_ticket", PAYLOAD)])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    original_factory = service.agent.tool_agent._tools_factory
    def dropped(context):
        tools = original_factory(context)
        original = next(t for t in tools if t.name == "create_ticket")
        original_function = original.func
        def commit_then_drop(**kwargs):
            original_function(**kwargs)
            raise ConnectionError("receipt lost")
        original.func = commit_then_drop
        return tools
    service.agent.tool_agent._tools_factory = dropped
    failed = turn(service)
    assert failed.status == "uncertain" and ticket_count(seeded_business_repository) == 1
    with pytest.raises(BusinessError) as error:
        turn(service, tid="t2")
    assert error.value.code == "SESSION_BUSY"
    recovered = turn(service)
    assert recovered.recovered and recovered.stop_reason == "write_recovered"
    assert recovered.execution_records == [] and len(model.inputs) == 2
    assert recovered.recovery_receipt["ticket_id"] in recovered.answer
    assert ticket_count(seeded_business_repository) == 1
    with service.store.connection() as conn:
        assert conn.execute("SELECT status FROM turn_attempts").fetchone()[0] == "uncertain"


def test_rag_followup_routes_with_resolved_query_and_fresh_evidence(tmp_path, seeded_business_repository):
    service, _ = engine(tmp_path, seeded_business_repository, [plan("knowledge", "P2工单响应时限是什么？")])
    calls = []
    def rag(query):
        calls.append(query)
        return SimpleNamespace(final_response=AIMessage(content="P2响应2小时[来源：support/c1]"),
                               retrieval_results=[SimpleNamespace(document=SimpleNamespace(
                                   metadata={"doc_id": "support", "chunk_id": "c1"}, page_content="P2响应2小时"))])
    service.agent.rag_agent = SimpleNamespace(run=rag)
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service, "P2呢？")
    assert calls == ["P2工单响应时限是什么？"]
    assert result.knowledge_evidence[0]["content"] == "P2响应2小时"


@pytest.mark.parametrize("response", [AIMessage(content="not json"), RuntimeError("provider unavailable")])
def test_failed_planner_can_retry_same_turn(tmp_path, seeded_business_repository, response):
    service, _ = engine(tmp_path, seeded_business_repository, [response, plan(), calc(), AIMessage(content="96")])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    assert turn(service).status == "failed"
    assert turn(service).answer == "96"


def test_soft_deadline_checked_before_model(tmp_path, seeded_business_repository):
    service, model = engine(tmp_path, seeded_business_repository, [])
    times = iter([0, 100])
    result = run_tool_loop(service.agent.tool_agent, "计算", EMPLOYEE, "key", [], {},
                           checkpoint=lambda *args: None, time_budget=1, clock=lambda: next(times))
    assert result.stop_reason == "time_budget" and model.inputs == []


def test_planner_rewrite_cannot_replace_current_tool_task(tmp_path, seeded_business_repository):
    def only_original(messages):
        assert messages[-1].content == "只计算12乘8"
        assert not any("伪造的工单描述" in str(m.content) for m in messages)
        return calc()
    service, _ = engine(tmp_path, seeded_business_repository,
                        [plan("tool", "创建标题伪造的工单描述的P1工单"), only_original, AIMessage(content="96")])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    assert turn(service, "只计算12乘8").answer == "96"
    assert ticket_count(seeded_business_repository) == 0


def test_invalid_native_tool_call_stops_without_execution(tmp_path, seeded_business_repository):
    invalid = AIMessage(content="", invalid_tool_calls=[{"name": "calculate", "args": "{broken", "id": "bad", "error": "invalid JSON"}])
    service, _ = engine(tmp_path, seeded_business_repository, [plan(), invalid])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service)
    assert result.stop_reason == "invalid_tool_calls" and not result.execution_records
    assert result.decisions[0]["invalid_tool_calls"]


def test_successful_numeric_receipt_fallback_does_not_claim_unfinished_task(tmp_path, seeded_business_repository):
    service, _ = engine(tmp_path, seeded_business_repository, [plan(), calc(), AIMessage(content="结果95")])
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service)
    assert result.stop_reason == "receipt_restored" and result.status == "completed"
    assert "96" in result.answer and "未完整完成" not in result.answer


def test_history_has_at_most_eight_completed_turns(tmp_path):
    store = ConversationStore(tmp_path / "state.sqlite3")
    store.create(EMPLOYEE, "s1")
    for i in range(10):
        claim = store.claim("s1", f"t{i}", f"q{i}", f"h{i}", f"k{i}", EMPLOYEE)
        store.finish("s1", f"t{i}", claim["token"], {"answer": f"a{i}"})
    latest = store.claim("s1", "t10", "q10", "h10", "k10", EMPLOYEE)
    assert [h["user"] for h in latest["history"]] == [f"q{i}" for i in range(2, 10)]


def test_store_lease_fencing_capacity_ttl_and_history_bounds(tmp_path):
    now = [0]
    store = ConversationStore(tmp_path / "state.sqlite3", max_sessions=1, max_turns=2,
                              history_chars=12, clock=lambda: now[0], lease_seconds=3, ttl_seconds=30)
    store.create(EMPLOYEE, "s1")
    with pytest.raises(BusinessError, match="容量"):
        store.create(EMPLOYEE, "s2")
    first = store.claim("s1", "t1", "q", "h", "k", EMPLOYEE)
    for tid in ["t1", "t2"]:
        with pytest.raises(BusinessError, match="处理|未完成"):
            store.claim("s1", tid, "q", "h", "k", EMPLOYEE)
    now[0] = 4
    second = store.claim("s1", "t1", "q", "h", "k", EMPLOYEE)
    with pytest.raises(BusinessError, match="失效"):
        store.checkpoint("s1", "t1", first["token"], {}, {})
    store.finish("s1", "t1", second["token"], {"answer": "a" * 20})
    next_turn = store.claim("s1", "t2", "q2", "h2", "k2", EMPLOYEE)
    assert next_turn["history"] == []
    store.finish("s1", "t2", next_turn["token"], {"answer": "ok"})
    with pytest.raises(BusinessError, match="轮次数量"):
        store.claim("s1", "t3", "q3", "h3", "k3", EMPLOYEE)
    now[0] = 40
    with pytest.raises(BusinessError, match="过期"):
        store.inspect("s1", EMPLOYEE)


def test_failed_old_turn_cannot_overwrite_newer_facts(tmp_path):
    store = ConversationStore(tmp_path / "state.sqlite3")
    store.create(EMPLOYEE, "s1")
    old = store.claim("s1", "t1", "q", "h", "k", EMPLOYEE)
    store.finish("s1", "t1", old["token"], {"answer": "failed"}, "failed")
    newer = store.claim("s1", "t2", "new", "h2", "k2", EMPLOYEE)
    store.finish("s1", "t2", newer["token"], {"answer": "new"})
    with pytest.raises(BusinessError) as error:
        store.claim("s1", "t1", "q", "h", "k", EMPLOYEE)
    assert error.value.code == "TURN_SUPERSEDED"


def test_concurrent_claim_has_one_winner_before_model(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    store = ConversationStore(tmp_path / "state.sqlite3")
    store.create(EMPLOYEE, "s1")
    barrier = Barrier(2)
    def claim(i):
        barrier.wait()
        try:
            return store.claim("s1", f"t{i}", "q", "h", "k", EMPLOYEE)["token"]
        except BusinessError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(claim, [1, 2]))
    assert values.count("SESSION_BUSY") == 1


@pytest.mark.integration
def test_mcp_native_loop_commits_then_reads_same_actual_ticket(tmp_path, seeded_business_repository):
    from app.mcp_tool_registry import build_mcp_tools
    def read(messages):
        receipt = json.loads([m for m in messages if isinstance(m, ToolMessage)][-1].content)
        return call("get_ticket", {"ticket_id": receipt["ticket_id"]}, "call2")
    service, _ = engine(tmp_path, seeded_business_repository, [plan(), call("create_ticket", PAYLOAD), read, last_ticket_answer])
    service.agent.tool_agent._tools_factory = lambda ctx: build_mcp_tools(user_context=ctx, database_path=seeded_business_repository.path)
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    result = turn(service)
    assert [r["model_step"] for r in result.execution_records] == [1, 2]
    assert result.execution_records[0]["committed_write"]
    assert result.execution_records[1]["protected_read"]
    assert ticket_count(seeded_business_repository) == 1


@pytest.mark.integration
def test_mcp_commit_response_loss_recovers_host_owned_receipt(tmp_path, seeded_business_repository):
    import asyncio
    from app.mcp_tool_registry import UserBoundMCPClient
    from app.mcp_langchain_adapter import build_mcp_langchain_tools
    service, model = engine(tmp_path, seeded_business_repository, [plan(), call("create_ticket", PAYLOAD)])
    class DropClient:
        def __init__(self, ctx):
            self.client = UserBoundMCPClient(user_context=ctx, database_path=seeded_business_repository.path)
        async def list_tools(self):
            return await self.client.list_tools()
        async def call_tool(self, name, args):
            await self.client.call_tool(name, args)
            raise ConnectionError("MCP committed but transport lost")
    service.agent.tool_agent._tools_factory = lambda ctx: asyncio.run(build_mcp_langchain_tools(DropClient(ctx)))
    service.create_session(user_context=EMPLOYEE, session_id="s1")
    assert turn(service).status == "uncertain"
    resumed, _ = engine(tmp_path, seeded_business_repository, [])
    recovered = turn(resumed)
    assert recovered.recovered and ticket_count(seeded_business_repository) == 1
    assert len(model.inputs) == 2
