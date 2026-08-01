

class LLMClient:

    def __init__(self, llm):
        self.llm = llm
        

    def chat(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content
