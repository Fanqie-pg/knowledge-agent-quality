from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from app.llm_client import LLMClient


def main():

    load_dotenv()
    print("环境已加载")

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY未配置")
    
    chat_model = ChatOpenAI(
        model = "deepseek-v4-flash",
        api_key = api_key,
        base_url = "https://api.deepseek.com"
    )

    client = LLMClient(chat_model)

    answer = client.chat("世界上最高的山峰是哪个？")

    print(answer)

if __name__ == "__main__":
    main()


