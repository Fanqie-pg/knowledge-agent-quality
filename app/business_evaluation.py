"""Business acceptance checks compare model actions with actual authorized SQLite state."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

from app.business_repository import SQLiteBusinessRepository
from app.dataset_governance import ROOT, read_jsonl, sha256
from app.evaluation_records import RunStore, CallBudget, attach_budget
from app.evaluation_campaign import provider_error_details
from app.identity import load_user_context
from app.intent_router import IntentRouter, AgentRoute
from app.mcp_tool_registry import build_mcp_tools
from app.routing_case_loader import load_routing_cases
from app.tool_agent import ToolAgent
from app.tool_registry import build_tools
from app.tracing import capture_trace
from app.unified_agent import UnifiedAgent, AgentRunStatus


def visible_number(text, value):
    numbers = re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?![A-Za-z0-9.])", str(text))
    return any(abs(float(number) - value) <= 1e-9 for number in numbers)


def evaluate_case(agent, case, repository, transport, expected_arguments):
    from app.tool_control_evaluation import database_snapshot
    state_before = database_snapshot(repository)
    key = f"batch2-{transport}-{case.get('intent',case['case_id'])}"
    with repository.connection() as connection:
        existing = connection.execute("SELECT ticket_id FROM tickets WHERE requester_id=? AND request_key=?",("E001",f"batch2-{transport}-ticket-one")).fetchone()
        before = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    ticket_id = existing["ticket_id"] if existing else "T-unavailable"
    query = case["query"].format(ticket_id=ticket_id)
    with capture_trace() as events:
        result = agent.run(query, request_key=key, user_context=load_user_context(case["actor"]))
    tool = result.tool_result
    records = tool.execution_records if tool else []
    calls = tool.decision.tool_calls if tool else []
    codes = [record.get("error_code") for record in records]
    reads = [event for event in events if event["stage"]=="business" and event["event"]=="protected_read"]
    writes = [event for event in events if event["stage"]=="business" and event["event"]=="committed_write"]
    with repository.connection() as connection:
        after = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        stored = connection.execute("SELECT * FROM tickets WHERE requester_id=? AND request_key=?",("E001",f"batch2-{transport}-ticket-one")).fetchall()
    checks={"route": result.route_decision is not None and result.route_decision.route is AgentRoute.TOOL,
            "flow_completed":result.status is AgentRunStatus.SUCCESS}
    expected=case["expect"]
    if calls:
        arguments={name:(value.format(ticket_id=ticket_id) if isinstance(value,str) else value) for name,value in expected_arguments.items()}
        normalized=[{key:(value.strip() if isinstance(value,str) else value) for key,value in call["args"].items()} for call in calls]
        checks["requested_tool_and_parameters"]=all(call["name"]==case["tool"] and params==arguments for call,params in zip(calls,normalized))
    else:
        checks["requested_tool_and_parameters"]=expected in {"denied","error"}
    if expected=="number":
        successful=[message for message in (tool.tool_messages if tool else []) if message.name==case["tool"] and message.status=="success"]
        checks["actual_tool_result"]=len(successful)==1 and abs(float(successful[0].content)-case["value"])<=1e-9
        checks["answer_number"]=visible_number(result.final_response.content,case["value"])
        checks["no_write"]=before==after
    elif expected in {"denied","error"}:
        refused_without_call=not records and any(word in str(result.final_response.content) for word in ("无法","不能","无权","权限","除数","除零"))
        checks["safe_failure"]=case["code"] in codes or refused_without_call
        checks["no_write"]=before==after and not writes
        if expected=="denied":
            checks["no_protected_read"]=not reads
    elif expected in {"created","replayed"}:
        checks["one_ticket"]=len(stored)==1
        row=dict(stored[0]) if stored else {}
        checks["database_contents"]=(row.get("requester_id")=="E001" and row.get("title")=="邮箱无法正常登录"
                                     and row.get("description")=="登录页面显示错误，请协助排查。" and row.get("priority")=="P2" and row.get("status")=="open")
        checks["receipt_in_answer"]=bool(row) and row["ticket_id"] in str(result.final_response.content)
        receipts=[record.get("write_receipt",{}) for record in records]
        checks["actual_receipt"]=any(receipt.get("ticket_id")==row.get("ticket_id") for receipt in receipts) and bool(row)
        if expected=="replayed":
            checks["reused_ticket"]=ticket_id==row.get("ticket_id") and not writes and before==after
    elif expected=="own_ticket":
        checks["actual_read"]=bool(reads) and any(message.name=="get_ticket" and message.status=="success" for message in (tool.tool_messages if tool else []))
        checks["receipt_in_answer"]=ticket_id!="T-unavailable" and ticket_id in str(result.final_response.content)
        checks["no_write"]=before==after
    else:
        raise ValueError("未知验收类型")
    return {"case_id":case["case_id"],"transport":transport,"actor":case["actor"],"query":query,
            "database_before":state_before,"database_after":database_snapshot(repository),
            "resolved_expected_arguments":{name:(value.format(ticket_id=ticket_id) if isinstance(value,str) else value) for name,value in expected_arguments.items()},
            "request_key":key,"status":"completed" if result.status is AgentRunStatus.SUCCESS else "error",
            "passed":all(checks.values()),"checks":checks,"answer":result.final_response.content,
            "execution_records":records,"database_ticket_count_before":before,"database_ticket_count_after":after,
            "model_tool_calls":calls,
            "answer_mode":tool.answer_mode if tool else None,"final_answer_error":tool.final_answer_error if tool else None,
            "trace":events,"failure":asdict(result.failure) if result.failure else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/batch2-business-live")
    parser.add_argument("--run-models",action="store_true")
    parser.add_argument("--max-calls",type=int,default=160)
    args=parser.parse_args()
    folder=ROOT/"data/evaluation/business_v1"
    cases=read_jsonl(folder/"agent_cases.jsonl")
    expected_arguments=json.loads((folder/"argument_expectations.json").read_text(encoding="utf-8"))
    routing=load_routing_cases(folder/"routing_cases.jsonl")
    if not args.run_models:
        print(json.dumps({"business_tasks":len(cases)*2,"routing_tasks":len(routing),"model_call_upper_budget":args.max_calls,"model_calls":0}))
        return 0
    store=RunStore(args.output,{"kind":"batch2_business_acceptance","cases_sha256":sha256(folder/"agent_cases.jsonl"),
                                "analysis_version":"1.1-explicit-parameter-checks","arguments_sha256":sha256(folder/"argument_expectations.json"),
                                "routing_sha256":sha256(folder/"routing_cases.jsonl"),"model":"deepseek-v4-flash",
                                "code":{path.name:sha256(path) for path in (ROOT/"app").glob("*.py")}})
    from app.agent_factory import build_chat_model,build_knowledge_retriever,build_rag_agent
    model=attach_budget(build_chat_model(),CallBudget(args.output/"calls.jsonl",args.max_calls))
    repository=SQLiteBusinessRepository(args.output/"business.sqlite3")
    repository.seed_demo()
    rag=build_rag_agent(model,build_knowledge_retriever())
    router=IntentRouter(model)
    seen={(row["case_id"],row.get("transport","routing")) for row in store.rows()}
    for case in routing:
        if (case.case_id,"routing") in seen:continue
        row={"case_id":case.case_id,"transport":"routing","status":"error","passed":False}
        try:
            decision=router.route(case.query)
            row.update(status="completed",query=case.query,expected_route=case.expected_route.value,
                       actual_route=decision.route.value,reason=decision.reason,passed=decision.route is case.expected_route)
        except Exception as exc:row.update(provider_error_details(exc))
        store.append(row)
        if row.get("error_http_status") in {401,402,403} or str(row.get("error_code", "")).lower() in {"arrearage", "insufficient_quota", "credit_balance_exhausted"}:
            raise RuntimeError("提供商权限或余额错误，已停止业务评测；原始记录已保存")
    for transport in ("local","mcp"):
        agents={}
        for case in cases:
            if (case["case_id"],transport) in seen:continue
            actor=transport  # one shared Agent per transport, never per user
            if actor not in agents:
                def request_tools(user, mode=transport):
                    return (build_tools(user_context=user, repository=repository) if mode == "local" else
                            build_mcp_tools(user_context=user, database_path=repository.path))
                agents[actor]=UnifiedAgent(router,ToolAgent(model,build_tools(repository=repository),tools_factory=request_tools),rag,model)
            try:
                row=evaluate_case(agents[actor],case,repository,transport,expected_arguments[case["case_id"]])
            except Exception as exc:
                row={"case_id":case["case_id"],"transport":transport,"status":"error","passed":False,**provider_error_details(exc)}
            store.append(row)
            if row.get("error_http_status") in {401,402,403} or str(row.get("error_code", "")).lower() in {"arrearage", "insufficient_quota", "credit_balance_exhausted"}:
                raise RuntimeError("提供商权限或余额错误，已停止业务评测；原始记录已保存")
    rows=store.rows()
    expected=len(cases)*2+len(routing)
    report={"expected":expected,"executed":len(rows),"passed":sum(row["passed"] for row in rows),
            "failed":sum(not row["passed"] for row in rows),"errors":sum(row["status"]=="error" for row in rows),
            "missing":expected-len(rows),"status":"passed" if len(rows)==expected and all(row["passed"] for row in rows) else "failed",
            "per_transport":{kind:{"count":sum(row["transport"]==kind for row in rows),"passed":sum(row["passed"] for row in rows if row["transport"]==kind)} for kind in ("routing","local","mcp")},
            "provider_calls":len(read_jsonl(args.output/"calls.jsonl")),"run_config":store.manifest["config"],
            "limitations":"合成工程验收各一次；拒绝的替代安全路径允许不调用工具。数据库与调用轨迹为业务完成依据，未声称生产覆盖率。"}
    store.report("summary",report)
    print(json.dumps({key:report[key] for key in ("status","expected","passed","failed","errors","per_transport","provider_calls")},ensure_ascii=False))
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":raise SystemExit(main())
