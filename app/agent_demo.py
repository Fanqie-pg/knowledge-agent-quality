import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.tool_agent import ToolAgent
from app.tool_registry import build_tools


def main():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY未配置")

    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    tools = build_tools()
    agent = ToolAgent(llm, tools)
    questions = [
        "12乘以8是多少？",
        "查询员工E001的剩余年假",
        "你好，请介绍一下自己",
        "查询员工E999的剩余年假",
    ]
    for prompt in questions:
        result = agent.run(prompt)
        print("用户问题：", prompt)
        print("工具决策:", result.decision.tool_calls)
        print(
            "工具结果:",
            [message.content for message in result.tool_messages]
        )
        print("最终回答:", result.final_response.content)
        print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()