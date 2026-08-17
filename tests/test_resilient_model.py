import logging

import pytest
from langchain_core.messages import AIMessage

from app.resilient_model import RetryingChatModel, ModelUnavailableError


pytestmark = pytest.mark.unit


class FakeSequenceModel:

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.call_count = 0

    def invoke(self, model_input, *args, **kwargs):
        self.call_count += 1
        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


def test_invoke_retries_timeout_and_then_succeeds():
    expected_response = AIMessage(content="成功")

    fake_model = FakeSequenceModel(
        [
            TimeoutError("第一次调用超时"),
            expected_response,
        ]
    )
    model = RetryingChatModel(fake_model, max_attempts=3)

    response = model.invoke("测试问题")

    assert response is expected_response
    assert fake_model.call_count == 2


def test_invoke_raises_model_unavailable_after_max_attempts():
    fake_model = FakeSequenceModel(
        [
            TimeoutError("第一次超时"),
            TimeoutError("第二次超时"),
            TimeoutError("第三次超时"),
        ]
    )
    model = RetryingChatModel(fake_model, max_attempts=3)

    with pytest.raises(ModelUnavailableError, match="模型服务暂时不可用") as exc_info:
        model.invoke("测试问题")

    assert fake_model.call_count == 3
    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_invoke_does_not_retry_non_retryable_error():
    fake_model = FakeSequenceModel(
        [
            ValueError("请求参数错误"),
        ]
    )
    model = RetryingChatModel(
        fake_model,
        max_attempts=3,
    )
    with pytest.raises(ValueError, match="请求参数错误"):
        model.invoke("测试问题")

    assert fake_model.call_count == 1


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        (
            {"max_attempts": 0},
            "max_attempts必须大于0",
        ),
        (
            {"retry_delay_seconds": -1},
            "retry_delay_seconds不能小于0",
        ),
    ],
)
def test_rejects_invalid_retry_configuration(kwargs, expected_message):
    fake_model = FakeSequenceModel([])
    with pytest.raises(ValueError, match=expected_message):
        RetryingChatModel(fake_model, **kwargs)


def test_invoke_uses_exponential_backoff():
    expected_response = AIMessage(content="成功")
    sleep_delays = []

    fake_model = FakeSequenceModel(
        [
            TimeoutError("第一次超时"),
            ConnectionError("第二次连接失败"),
            expected_response,
        ]
    )
    model = RetryingChatModel(
        fake_model,
        max_attempts=3,
        retry_delay_seconds=0.5,
        sleep_fn=sleep_delays.append,
    )

    response = model.invoke("测试问题")

    assert response is expected_response
    assert fake_model.call_count == 3
    assert sleep_delays == [0.5, 1.0]


def test_invoke_logs_attempts_without_sensitive_message(caplog):
    fake_model = FakeSequenceModel(
        [
            TimeoutError("api_key=internal-secret"),
            TimeoutError("api_key=internal-secret"),
        ]
    )
    model = RetryingChatModel(
        fake_model,
        max_attempts=2,
    )
    with caplog.at_level(logging.WARNING, logger="app.resilient_model"):
        with pytest.raises(ModelUnavailableError):
            model.invoke("测试问题")

    assert "attempt=1/2" in caplog.text
    assert "attempt=2/2" in caplog.text
    assert "TimeoutError" in caplog.text
    assert "internal-secret" not in caplog.text


class FakeBindableModel:

    def __init__(self, bound_model):
        self.bound_model = bound_model
        self.received_tools = None
        self.received_kwargs = None

    def bind_tools(self, tools, **kwargs):
        self.received_tools = tools
        self.received_kwargs = kwargs
        return self.bound_model


def test_bind_tools_returns_retrying_bound_model():
    bound_model = FakeSequenceModel(
        [AIMessage(content="工具回答")]
    )
    base_model = FakeBindableModel(bound_model)
    sleep_fn = lambda delay: None

    model = RetryingChatModel(
        base_model,
        max_attempts=3,
        retry_delay_seconds=0.5,
        sleep_fn=sleep_fn,
    )

    bound_retrying_model = model.bind_tools(["tool-a"], tool_choice="auto")

    assert isinstance(bound_retrying_model, RetryingChatModel)
    assert bound_retrying_model.model is bound_model
    assert bound_retrying_model.max_attempts == 3
    assert bound_retrying_model.retry_delay_seconds == 0.5
    assert bound_retrying_model.sleep_fn is sleep_fn

    assert base_model.received_tools == ["tool-a"]
    assert base_model.received_kwargs == {
        "tool_choice": "auto"
    }
    assert bound_retrying_model.retryable_exceptions == model.retryable_exceptions

class TemporaryProviderError(Exception):
    pass

def test_invoke_retries_configured_provider_error():
    expected_response = AIMessage(content="成功")
    fake_model = FakeSequenceModel(
        [
            TemporaryProviderError("临时异常"),
            expected_response,
        ]
    )

    model = RetryingChatModel(
        fake_model,
        max_attempts=2,
        retryable_exceptions=(
            TemporaryProviderError,
        ),
    )

    response = model.invoke("测试问题")

    assert response is expected_response
    assert fake_model.call_count == 2