"""Interactive terminal demo for owner-bound multi-turn conversations."""
import argparse
import json
import os
import uuid

from app.business_errors import BusinessError
from app.business_repository import SQLiteBusinessRepository
from app.conversation_store import ConversationStore
from app.identity import load_user_context
from app.tracing import redact


HELP = """直接输入问题并回车，可连续追问。
  /new     新建会话，清空本次对话上下文
  /retry   使用同一轮次重试上次问题，避免重复建单
  /status  查看当前身份、会话和上一轮标识
  /help    查看帮助
  /exit    退出

可以先试：请用工具计算12乘8 → 刚才的结果再乘2
或：帮我创建工单，标题是“VPN连接故障” → 描述是“连接超时无法办公” → 优先级P2，现在创建 → 查刚才的工单
"""


def interact(agent, context, *, session_id=None, verbose=False, read=input, write=print):
    session = agent.create_session(user_context=context, session_id=session_id)
    last = None
    pending = False
    if session_id and hasattr(agent, "store"):
        # create_session has already checked ownership. Recover the UI's request
        # identifiers as well as the agent's durable state after a terminal restart.
        with agent.store.connection() as conn:
            row = conn.execute("SELECT query,turn_id,status FROM turns WHERE session_id=? ORDER BY sequence DESC LIMIT 1",
                               (session,)).fetchone()
            if row and row["status"] in {"running", "uncertain", "failed"}:
                last, pending = {"query": row["query"], "turn_id": row["turn_id"]}, True
    write(f"当前身份：{context.principal_id} / {context.employee_id} / {context.role.value}")
    write(f"会话：{session}")
    write(HELP)
    if pending:
        write("此会话有未完成轮次，输入 /retry 恢复原请求。")
    while True:
        try:
            query = read("你：").strip()
        except (EOFError, KeyboardInterrupt):
            write("\n已退出。")
            return 0
        if not query:
            continue
        if query in {"/exit", "exit", "quit"}:
            write("已退出。")
            return 0
        if query == "/help":
            write(HELP)
            continue
        if query == "/status":
            write(f"身份：{context.principal_id}；会话：{session}；上一轮：{last['turn_id'] if last else '无'}")
            continue
        if query == "/new":
            try:
                session = agent.create_session(user_context=context)
                last = None
                pending = False
                write(f"已新建会话：{session}")
            except BusinessError as exc:
                write(f"[{exc.code}] {exc.public_message}")
            continue
        if query == "/retry":
            if last is None:
                write("还没有可重试的问题。")
                continue
        elif query.startswith("/"):
            write("未知命令，输入 /help 查看帮助。")
            continue
        else:
            if pending:
                write("上一轮结果尚未确认，请先用 /retry 恢复原轮次，或用 /new 明确开始新会话。")
                continue
            if len(query) > 4000:
                write("每轮问题最多4000字符，请缩短后重新输入。")
                continue
            last = {"query": query, "turn_id": uuid.uuid4().hex}
        write("助手处理中……")
        try:
            result = agent.run(last["query"], session_id=session, turn_id=last["turn_id"], user_context=context)
        except BusinessError as exc:
            pending = True
            write(f"[{exc.code}] {exc.public_message}")
            write("可用 /retry 重试原轮次；会话过期时用 /new 开始新会话。")
            continue
        except KeyboardInterrupt:
            # Keep the identifiers visible: the underlying turn may still hold
            # a lease or have committed a write before interruption.
            write(f"\n处理已中断。会话：{session}；轮次：{last['turn_id']}。请确认原轮次结果后再提交新的创建请求。")
            return 130
        except Exception as exc:
            pending = True
            write(f"处理异常（{type(exc).__name__}），可输入 /retry 重试原轮次。")
            continue
        write("助手：" + result.answer)
        pending = result.status in {"failed", "uncertain"}
        tools = [r["tool_name"] for r in result.execution_records if r["invoked"]]
        mode = "原轮缓存" if result.cached else "恢复回执" if result.recovered else "本轮执行"
        write(f"[{mode} | 路由：{result.route or '-'} | 工具：{', '.join(tools) or '无'} | 状态：{result.status}]")
        if result.status in {"failed", "uncertain"}:
            write("请用 /retry 重试原轮次，保留同一业务请求凭据。")
        if verbose:
            write(json.dumps(redact({"turn_id": result.turn_id, "stop_reason": result.stop_reason,
                                     "decisions": result.decisions, "execution_records": result.execution_records}),
                             ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="多轮对话终端 Demo（真实模型）")
    parser.add_argument("--user", choices=["employee-e001", "employee-e002", "hr"], default="employee-e001")
    parser.add_argument("--transport", choices=["local", "mcp"], default="local")
    parser.add_argument("--database", default="var/conversation-demo/business.sqlite3")
    parser.add_argument("--sessions", default="var/conversation-demo/conversations.sqlite3")
    parser.add_argument("--session", help="继续同身份的已有会话")
    parser.add_argument("--verbose", action="store_true", help="显示原始模型工具提议和执行记录")
    args = parser.parse_args(argv)
    # This workstation already has the embedding weights. Avoid an unnecessary
    # Hugging Face connectivity check on each interactive launch.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    print("正在初始化模型和本地知识库……")
    try:
        from app.agent_factory import build_conversation_agent
        repository = SQLiteBusinessRepository(args.database)
        repository.seed_demo()
        agent = build_conversation_agent(repository=repository, store=ConversationStore(args.sessions), transport=args.transport)
        return interact(agent, load_user_context(args.user), session_id=args.session, verbose=args.verbose)
    except BusinessError as exc:
        print(f"[{exc.code}] {exc.public_message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
