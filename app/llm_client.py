

class LLMClient:

    def __init__(self, llm):
        self.llm = llm

    def chat(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt不能为空")
        response = self.llm.invoke(prompt)
        return response.content
