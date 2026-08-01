from types import SimpleNamespace

import pytest

from app.llm_client import LLMClient


class TimeoutLLM:
    def invoke(self, prompt):
        raise TimeoutError("模型调用超时")

class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(content="珠穆朗玛峰")


def test_chat_passes_prompt_and_returns_content():
    """
    验证：
    LLMClient 将 prompt 原样传给底层模型；
    LLMClient 从响应对象中提取并返回 content。
    """
    fake_llm = FakeLLM()
    client = LLMClient(fake_llm)
    result = client.chat("测试问题")
    assert fake_llm.last_prompt == "测试问题"
    assert result == "珠穆朗玛峰"

def test_chat_propagates_timeout_error():
    timeout_llm = TimeoutLLM()
    client = LLMClient(timeout_llm)
    with pytest.raises(TimeoutError, match="模型调用超时"):
        client.chat("测试")