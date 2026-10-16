"""Run one tool request with host-selected identity; no conversation state."""
import argparse
import json

from app.agent_factory import build_chat_model
from app.business_repository import SQLiteBusinessRepository
from app.identity import load_user_context
from app.mcp_tool_registry import build_mcp_tools
from app.tool_agent import ToolAgent
from app.tool_registry import build_tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--user", choices=["employee-e001", "employee-e002", "hr"])
    parser.add_argument("--transport", choices=["local", "mcp"], default="local")
    parser.add_argument("--database", default="var/business.sqlite3")
    parser.add_argument("--seed-demo", action="store_true", help="显式导入合成员工数据")
    parser.add_argument("--untrusted-context", help="外部资料文本，不作为宿主授权")
    parser.add_argument("--decision-only", action="store_true", help="只查看模型提议，不执行工具")
    args = parser.parse_args()
    user = load_user_context(args.user)
    repository = SQLiteBusinessRepository(args.database)
    if args.seed_demo:
        repository.seed_demo()
    tools = build_tools(repository=repository)
    def request_tools(context):
        return (build_tools(user_context=context, repository=repository) if args.transport == "local"
                else build_mcp_tools(user_context=context, database_path=repository.path))
    agent = ToolAgent(build_chat_model(), tools, tools_factory=request_tools)
    kwargs = {"user_context": user, "untrusted_context": args.untrusted_context}
    if args.decision_only:
        decision = agent.decide(args.query, **kwargs)
        output = {"model_tool_calls": decision.tool_calls, "answer": decision.content, "executed": False}
    else:
        result = agent.run(args.query, **kwargs)
        output = {"model_tool_calls": result.decision.tool_calls, "execution_records": result.execution_records,
                  "answer": result.final_response.content}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
