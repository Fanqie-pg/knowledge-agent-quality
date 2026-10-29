"""Interactive request identifiers must preserve write retry semantics."""
from types import SimpleNamespace

from app.conversation_demo import interact
from app.identity import load_user_context


class DemoAgent:
    def __init__(self):
        self.sessions = []
        self.calls = []

    def create_session(self, *, user_context, session_id=None):
        session = session_id or f"session-{len(self.sessions) + 1}"
        self.sessions.append((session, user_context))
        return session

    def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return SimpleNamespace(answer="完成", execution_records=[], cached=False, recovered=False,
                               route="tool", status="completed")


def test_retry_reuses_turn_new_input_and_session_have_new_identifiers():
    agent = DemoAgent()
    commands = iter(["创建工单", "/retry", "再创建一张", "/new", "新会话问题", "/exit"])
    context = load_user_context("employee-e001")
    assert interact(agent, context, read=lambda _: next(commands), write=lambda _: None) == 0
    first, retry, new_intent, new_session = agent.calls
    assert first == retry
    assert first[1]["turn_id"] != new_intent[1]["turn_id"]
    assert first[1]["session_id"] == new_intent[1]["session_id"]
    assert new_session[1]["session_id"] != first[1]["session_id"]
    assert all(call[1]["user_context"] is context for call in agent.calls)


def test_commands_and_empty_input_do_not_invoke_model():
    agent = DemoAgent()
    output = []
    commands = iter(["", "/retry", "/status", "/help", "/unknown", "/exit"])
    interact(agent, load_user_context("hr"), read=lambda _: next(commands), write=output.append)
    assert not agent.calls
    assert any("还没有可重试" in line for line in output)


def test_uncertain_turn_cannot_be_overwritten_before_retry():
    class UncertainAgent(DemoAgent):
        def run(self, query, **kwargs):
            result = super().run(query, **kwargs)
            result.status = "uncertain" if len(self.calls) == 1 else "completed"
            return result
    agent = UncertainAgent()
    commands = iter(["创建工单", "别的新问题", "/retry", "/exit"])
    interact(agent, load_user_context("employee-e001"), read=lambda _: next(commands), write=lambda _: None)
    assert len(agent.calls) == 2 and agent.calls[0] == agent.calls[1]


def test_restart_restores_original_pending_request_identifiers(tmp_path):
    from app.conversation_store import ConversationStore
    context = load_user_context("employee-e001")
    store = ConversationStore(tmp_path / "sessions.sqlite3")
    store.create(context, "s1")
    claim = store.claim("s1", "original-turn", "原问题", "hash", "key", context)
    store.finish("s1", "original-turn", claim["token"], {"answer": "未知"}, "uncertain")
    agent = DemoAgent()
    agent.store = store
    commands = iter(["/retry", "/exit"])
    interact(agent, context, session_id="s1", read=lambda _: next(commands), write=lambda _: None)
    assert agent.calls[0][0] == "原问题" and agent.calls[0][1]["turn_id"] == "original-turn"


def test_restart_does_not_restore_failed_turn_superseded_by_completed_turn(tmp_path):
    from app.conversation_store import ConversationStore
    context = load_user_context("employee-e001")
    store = ConversationStore(tmp_path / "sessions.sqlite3")
    store.create(context, "s1")
    for tid, status in [("old", "failed"), ("new", "completed")]:
        claim = store.claim("s1", tid, tid, tid, tid, context)
        store.finish("s1", tid, claim["token"], {"answer": "答复"}, status)
    agent = DemoAgent()
    agent.store = store
    commands = iter(["新问题", "/exit"])
    interact(agent, context, session_id="s1", read=lambda _: next(commands), write=lambda _: None)
    assert len(agent.calls) == 1 and agent.calls[0][0] == "新问题"
