from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool


@dataclass
class AgentRunResult:
    decision: AIMessage
    tool_messages: list[ToolMessage]
    final_response: AIMessage


class ToolAgent:
    def __init__(self, chat_model, tools: list[StructuredTool]):
        self.model_with_tools = chat_model.bind_tools(tools)
        self.tools_by_name = {
            tool.name: tool for tool in tools
        }

    def decide(self, prompt: str) -> AIMessage:
        result = self.model_with_tools.invoke(prompt)
        return result

    def execute_tool_calls(self, message: AIMessage) -> list[ToolMessage]:
        tool_messages = []
        for tool_call in message.tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            try:
                content = str(tool.invoke(tool_call["args"]))
                status = "success"
            except (ValueError, KeyError) as exc:
                error_message = exc.args[0] if exc.args else str(exc)
                content = error_message
                status = "error"
            tool_messages.append(
                ToolMessage(
                    status=status,
                    content=content,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )
        return tool_messages

    def run(self, prompt: str) -> AgentRunResult:
        decision = self.decide(prompt)
        if not decision.tool_calls:
            return AgentRunResult(
                decision=decision,
                tool_messages=[],
                final_response=decision,
            )
        tool_messages = self.execute_tool_calls(decision)
        messages = [
            HumanMessage(content=prompt),
            decision,
            *tool_messages,
        ]
        final_response = self.model_with_tools.invoke(messages)
        return AgentRunResult(
            decision=decision,
            tool_messages=tool_messages,
            final_response=final_response,
        )