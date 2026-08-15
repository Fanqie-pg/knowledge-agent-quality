import json
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class AgentRoute(str, Enum):
    TOOL = "tool"
    KNOWLEDGE = "knowledge"
    CHAT = "chat"


class RouteDecision(BaseModel):
    route: AgentRoute
    reason: str


ROUTER_SYSTEM_PROMPT = """
你是企业智能助手的意图路由器。

请把用户问题路由到以下一种类型：

1. tool
需要执行工具操作，例如：
- 数学计算；
- 查询某位员工当前的年假余额；
- 查询个人或实时结构化数据。

2. knowledge
需要查询企业知识库，例如：
- 公司制度和政策；
- 产品说明；
- API文档；
- 故障处理流程；
- 常见问题和操作方法。

3. chat
不需要工具或企业知识库，例如：
- 问候；
- 自我介绍；
- 普通闲聊。

边界规则：
- 查询某位员工当前剩余多少年假，选择tool。
- 询问公司年假制度、结转规则或申请要求，选择knowledge。
- 只负责判断路由，不要回答用户问题。

只输出一个JSON对象，不要输出Markdown代码块或额外文字。
JSON必须包含route和reason。
route只能是tool、knowledge或chat。
""".strip()


def parse_route_decision(content: str) -> RouteDecision:
    cleaned_content = content.strip()
    if cleaned_content.startswith("```"):
        lines = cleaned_content.splitlines()

        if lines[0].strip() in {"```json", "```"}:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_content = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned_content)
    except json.JSONDecodeError as error:
        raise ValueError("路由模型返回内容不是有效JSON") from error

    return RouteDecision.model_validate(data)


class IntentRouter:

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def route(self, query: str) -> RouteDecision:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query不能为空")

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"用户问题：\n{query}"),
        ]

        response = self.chat_model.invoke(messages)

        return parse_route_decision(response.content)
