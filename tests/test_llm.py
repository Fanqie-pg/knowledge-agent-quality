from types import SimpleNamespace
from app.llm_client import LLMClient




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
