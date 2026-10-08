from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.evaluation_records import CallBudget, CallBudgetExceeded
from app.judge_model import QwenJudgeModel, build_judge_model, judge_model_config
from app.tracing import capture_trace

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("suite,expected", [("rag", "qwen3.7-plus-2026-05-26"), ("security", "qwen3.8-max-0902")])
def test_judge_uses_role_model_and_preserves_usage(suite, expected):
    calls = []
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5, model_dump=lambda: {"reasoning_tokens": 1})
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"ok":true}', reasoning_content="private reasoning"))],
                               id="response-test", model=expected, usage=usage)
    def create(**kwargs):
        calls.append(kwargs)
        return response
    model = QwenJudgeModel(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), suite)
    with capture_trace() as events:
        result = model.invoke([SystemMessage(content="JSON only"), HumanMessage(content="test")])
    assert calls[0]["model"] == expected
    assert calls[0]["extra_body"] == {"enable_thinking": True, "thinking_budget": 2048}
    assert "response_format" not in calls[0] and "tools" not in calls[0]
    assert calls[0]["messages"][1] == {"role": "user", "content": "test"}
    assert result.content == '{"ok":true}'
    assert result.usage_metadata["total_tokens"] == 5
    assert events[0]["data"]["response_id"] == "response-test"


@pytest.mark.parametrize("finish,text", [("length", '{"ok":true}'), ("stop", "")])
def test_judge_does_not_accept_incomplete_or_empty_responses(finish, text):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=text))], id="r", model="qwen", usage=None)
    model = QwenJudgeModel(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))))
    with pytest.raises(ValueError, match="incomplete or empty"):
        model.invoke([HumanMessage(content="JSON")])


def test_judge_does_not_fall_back_to_agent_credentials(monkeypatch):
    monkeypatch.setattr("app.judge_model.load_dotenv", lambda *args: None)
    for name in ("QWEN_API_KEY", "QWENT_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "agent-test-credential")
    monkeypatch.setenv("GPT_API_KEY", "obsolete-gpt-credential")
    with pytest.raises(RuntimeError, match="QWEN_API_KEY"):
        build_judge_model()
    assert judge_model_config()["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_budget_is_atomic_across_threads_and_wrappers(tmp_path):
    path = tmp_path / "calls.jsonl"
    budgets = [CallBudget(path, 5) for _ in range(4)]
    def consume(index):
        try:
            budgets[index % 4].consume()
            return True
        except CallBudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(consume, range(30))) == 5
    assert len(path.read_text().splitlines()) == 5
