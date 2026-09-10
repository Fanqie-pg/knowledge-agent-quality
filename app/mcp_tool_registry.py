"""Each MCP client has an immutable identity/database binding, separate per user."""
import asyncio
import os
from pathlib import Path
import sys
from dataclasses import dataclass

from mcp.client.stdio import StdioServerParameters

from app.business_repository import ROOT
from app.business_service import current_request_key
from app.identity import load_user_context
from app.mcp_client import MCPToolClient
from app.mcp_langchain_adapter import build_mcp_langchain_tools
from app.tool_permissions import identity_binding


@dataclass(frozen=True, init=False)
class UserBoundMCPClient:
    _database: str
    _profile_id: str
    _tool_profile: str
    _request_key: str | None
    _timeout: float

    def __init__(self, *, user_context=None, database_path="var/business.sqlite3", profile="current", request_key=None, read_timeout_seconds=10):
        if user_context is not None and load_user_context(user_context.principal_id) != user_context:
            raise ValueError("MCP身份必须与可信宿主配置一致")
        value = Path(database_path)
        if request_key is not None and (not isinstance(request_key, str) or not request_key.strip() or len(request_key) > 200):
            raise ValueError("宿主请求凭据不合法")
        object.__setattr__(self, "_database", str((value if value.is_absolute() else ROOT / value).resolve()))
        object.__setattr__(self, "_profile_id", user_context.principal_id if user_context else "")
        object.__setattr__(self, "_tool_profile", profile)
        object.__setattr__(self, "_request_key", request_key)
        object.__setattr__(self, "_timeout", read_timeout_seconds)

    def client(self):
        safe_environment = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC", "USERPROFILE", "HOME", "PYTHONPATH", "PYTHONUTF8", "LANG", "LC_ALL"}
        environment = {key: value for key, value in os.environ.items() if key.upper() in safe_environment}
        environment["PYTHONUTF8"] = "1"
        environment.update(AGENT_USER_PROFILE=self._profile_id, AGENT_DATABASE_PATH=self._database,
                           AGENT_TOOL_PROFILE=self._tool_profile, AGENT_REQUEST_KEY=self._request_key or current_request_key() or "")
        return MCPToolClient(StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=ROOT,
                                                   env=environment), read_timeout_seconds=self._timeout)

    async def list_tools(self):
        return await self.client().list_tools()

    async def call_tool(self, tool_name, arguments):
        return await self.client().call_tool(tool_name, arguments)


def build_mcp_tools(read_timeout_seconds=10, *, user_context=None, database_path="var/business.sqlite3", profile="current", request_key=None):
    client = UserBoundMCPClient(user_context=user_context, database_path=database_path, profile=profile,
                               request_key=request_key, read_timeout_seconds=read_timeout_seconds)
    tools = asyncio.run(build_mcp_langchain_tools(client))
    for tool in tools:
        tool.metadata = {**(tool.metadata or {}), "identity_binding": identity_binding(user_context)}
    return tools
