"""MCP raw-argument callbacks share the local strict schemas and services."""
import asyncio
import json
import os

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
from pydantic import ValidationError

from app.business_audit import collect_business_events
from app.business_errors import BusinessError
from app.business_repository import SQLiteBusinessRepository
from app.identity import load_user_context
from app.tool_registry import build_tools


def configured_tools():
    return build_tools(user_context=load_user_context(os.getenv("AGENT_USER_PROFILE")),
                       repository=SQLiteBusinessRepository(os.getenv("AGENT_DATABASE_PATH", "var/business.sqlite3")),
                       profile=os.getenv("AGENT_TOOL_PROFILE", "current"), request_key=os.getenv("AGENT_REQUEST_KEY"))


async def list_tools(context, parameters):
    return ListToolsResult(tools=[Tool(name=tool.name, description=tool.description,
                                      inputSchema=tool.args_schema.model_json_schema()) for tool in configured_tools()])


async def call_tool(context, parameters):
    tools = {tool.name: tool for tool in configured_tools()}
    name = parameters.name
    with collect_business_events() as audit:
        try:
            if name not in tools:
                raise BusinessError("UNKNOWN_TOOL", "Unknown tool: " + name)
            tool = tools[name]
            tool.args_schema.model_validate(parameters.arguments or {})
            value = tool.invoke(parameters.arguments or {})
        except ValidationError:
            error = {"code": "INVALID_ARGUMENTS", "message": "工具参数错误"}
        except BusinessError as exc:
            error = {"code": exc.code, "message": exc.public_message}
        except Exception:
            error = {"code": "INTERNAL_ERROR", "message": "工具执行失败，请稍后重试"}
        else:
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else str(value)
            return CallToolResult(content=[TextContent(type="text", text=text)], structuredContent={"result": value, "audit": audit})
    for event in audit:
        if event["data"].get("code") in {"TICKET_NOT_FOUND", "TICKET_SCOPE_DENIED"}:
            event["data"]["code"] = "RESOURCE_NOT_ACCESSIBLE"
    return CallToolResult(isError=True, content=[TextContent(type="text", text=error["message"])],
                          structuredContent={"error": error, "audit": audit})


server = Server("knowledge-agent-tools", version="2.0", on_list_tools=list_tools, on_call_tool=call_tool)


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
