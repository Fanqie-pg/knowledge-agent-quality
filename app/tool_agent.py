"""实现工具决策、执行前权限检查和双轮模型消息编排。"""

import logging
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError


@dataclass
class AgentRunResult:
    """记录模型请求、实际执行和被权限拦截的完整工具轨迹。"""

    decision: AIMessage
    tool_messages: list[ToolMessage]
    final_response: AIMessage
    requested_tool_names: list[str] = field(default_factory=list)
    executed_tool_names: list[str] = field(default_factory=list)
    blocked_tool_names: list[str] = field(default_factory=list)


def _format_tool_validation_error(error: ValidationError) -> str:
    """把 Pydantic 参数错误转换成适合返回给模型的稳定业务提示。"""
    messages = []
    for detail in error.errors():
        location = detail.get("loc", ())
        parameter_name = ".".join(
            str(part) for part in location
        )
        if detail.get("type") == "missing":
            messages.append(
                f"缺少参数 {parameter_name}"
            )
        else:
            messages.append(
                f"参数 {parameter_name} 格式不正确"
            )
    return "工具参数错误：" + "；".join(messages)


logger = logging.getLogger(__name__)


class ToolAgent:
    """让模型选择 StructuredTool，并把工具结果交给模型生成最终回答。"""

    def __init__(self, chat_model, tools: list[StructuredTool]):
        self.model_with_tools = chat_model.bind_tools(tools)
        self.tools_by_name = {
            tool.name: tool for tool in tools
        }

    def decide(self, prompt: str) -> AIMessage:
        """执行第一轮模型调用，获得直接回答或工具调用决策。"""
        result = self.model_with_tools.invoke(prompt)
        return result

    def execute_tool_calls(self, message: AIMessage, allowed_tool_names: list[str] | None = None) -> list[ToolMessage]:
        """执行模型请求的工具，并将成功或安全错误转换成 ToolMessage。"""
        tool_messages = []
        allowed_tool_name_set = (
            None
            if allowed_tool_names is None
            else set(allowed_tool_names)
        )
        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]

            # 权限必须在查找和调用工具之前生效，避免“回答拒绝但工具已执行”。
            if (
                allowed_tool_name_set is not None
                and tool_name not in allowed_tool_name_set
            ):
                tool_messages.append(
                    ToolMessage(
                        status="error",
                        content=f"无权调用工具：{tool_name}",
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )
                continue

            tool = self.tools_by_name.get(tool_name)

            if tool is None:
                tool_messages.append(
                    ToolMessage(
                        status="error",
                        content=f"未知工具：{tool_name}",
                        tool_call_id=tool_call["id"],
                        name=tool_name,
                    )
                )
                continue

            try:
                content = str(tool.invoke(tool_call["args"]))
                status = "success"
            except ValidationError as exc:
                content = _format_tool_validation_error(exc)
                status = "error"
            except (ValueError, KeyError) as exc:
                error_message = exc.args[0] if exc.args else str(exc)
                content = error_message
                status = "error"
            except Exception as exc:
                logger.error(
                    "工具执行失败：tool=%s, error_type=%s",
                    tool_name,
                    type(exc).__name__,
                )
                content = "工具执行失败，请稍后重试"
                status = "error"
            tool_messages.append(
                ToolMessage(
                    status=status,
                    content=content,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )
        return tool_messages

    def run(self, prompt: str, allowed_tool_names: list[str] | None = None) -> AgentRunResult:
        """完成工具决策、执行和基于 ToolMessage 的第二轮回答。"""
        decision = self.decide(prompt)
        requested_tool_names = [
            tool_call["name"]
            for tool_call in decision.tool_calls
        ]
        if not decision.tool_calls:
            return AgentRunResult(
                decision=decision,
                tool_messages=[],
                final_response=decision,
            )

        allowed_tool_name_set = (
            None
            if allowed_tool_names is None
            else set(allowed_tool_names)
        )

        blocked_tool_names = [
            tool_name
            for tool_name in requested_tool_names
            if (
                allowed_tool_name_set is not None
                and tool_name not in allowed_tool_name_set
            )
        ]

        executed_tool_names = [
            tool_name
            for tool_name in requested_tool_names
            if (
                tool_name in self.tools_by_name
                and (
                    allowed_tool_name_set is None
                    or tool_name in allowed_tool_name_set
                )
            )
        ]

        tool_messages = self.execute_tool_calls(decision, allowed_tool_names=allowed_tool_names)
        messages = [
            HumanMessage(content=prompt),
            decision,
            *tool_messages,
        ]
        # 第二轮模型同时看到原问题、工具决策和对应的工具执行结果。
        final_response = self.model_with_tools.invoke(messages)
        return AgentRunResult(
            decision=decision,
            tool_messages=tool_messages,
            final_response=final_response,
            requested_tool_names=requested_tool_names,
            executed_tool_names=executed_tool_names,
            blocked_tool_names=blocked_tool_names,
        )
