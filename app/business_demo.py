"""Explicit demo seed and single-operation local/MCP commands, without model calls."""
import argparse
import json
import uuid

from pydantic import ValidationError

from app.business_errors import BusinessError
from app.business_repository import SQLiteBusinessRepository
from app.identity import load_user_context
from app.mcp_tool_registry import build_mcp_tools
from app.tool_registry import build_tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="var/business.sqlite3")
    parser.add_argument("--user", choices=["employee-e001","employee-e002","hr"])
    parser.add_argument("--transport", choices=["local","mcp"], default="local")
    sub = parser.add_subparsers(dest="command",required=True)
    sub.add_parser("init")
    calc = sub.add_parser("calculate")
    calc.add_argument("operation",choices=["add","subtract","multiply","divide"])
    calc.add_argument("a",type=float); calc.add_argument("b",type=float)
    leave = sub.add_parser("leave"); leave.add_argument("employee_id")
    create = sub.add_parser("create")
    create.add_argument("--title",required=True); create.add_argument("--description",required=True)
    create.add_argument("--priority",required=True,choices=["P1","P2","P3"])
    create.add_argument("--request-key",help="宿主重试同一创建意图时复用；不提供则生成新意图凭据")
    get = sub.add_parser("get"); get.add_argument("ticket_id")
    args = parser.parse_args()
    repository = SQLiteBusinessRepository(args.database)
    if args.command == "init":
        repository.seed_demo()
        print(json.dumps({"status":"initialized","database":str(repository.path),"synthetic_data":True}))
        return 0
    key = getattr(args,"request_key",None) or uuid.uuid4().hex
    user = load_user_context(args.user)
    if args.transport == "local":
        tools = build_tools(user_context=user,repository=repository,request_key=key)
    else:
        tools = build_mcp_tools(user_context=user,database_path=repository.path,request_key=key)
    names={"leave":"get_leave_balance","create":"create_ticket","get":"get_ticket","calculate":"calculate"}
    fields={"calculate":["operation","a","b"],"leave":["employee_id"],"create":["title","description","priority"],"get":["ticket_id"]}
    try:
        result=next(tool for tool in tools if tool.name==names[args.command]).invoke({field:getattr(args,field) for field in fields[args.command]})
    except (ValidationError,BusinessError) as exc:
        code=exc.code if isinstance(exc,BusinessError) else "INVALID_ARGUMENTS"
        print(json.dumps({"status":"error","code":code,"message":str(exc) if isinstance(exc,BusinessError) else "工具参数错误"},ensure_ascii=False))
        return 1
    print(json.dumps({"status":"success","result":result,"request_key":key},ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
