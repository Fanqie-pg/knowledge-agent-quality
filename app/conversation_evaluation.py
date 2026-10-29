"""Opt-in live evaluation of complete conversations and actual business state."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
from types import SimpleNamespace

from app.business_errors import BusinessError
from app.business_repository import SQLiteBusinessRepository
from app.conversation_agent import ConversationAgent
from app.conversation_store import ConversationStore
from app.evaluation_records import CallBudget, RunStore, attach_budget, provenance
from app.dataset_governance import sha256
from app.identity import load_user_context
from app.tool_agent import ToolAgent
from app.tool_registry import build_tools
from app.tracing import capture_trace


DATA = Path("data/evaluation/conversation_v1/cases.json")


def load_cases(path=DATA):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [c["case_id"] for c in data["cases"]]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("会话用例ID必须唯一且非空")
    for case in data["cases"]:
        if not case["turns"] or any(not isinstance(t.get("query"), str) or not t["query"].strip() for t in case["turns"]):
            raise ValueError("每个会话必须包含有效问题")
        for i, turn in enumerate(case["turns"]):
            if "repeat_turn" in turn and not 1 <= turn["repeat_turn"] <= i:
                raise ValueError("重试必须引用已有轮次")
    return data


def grade_turn(expected, result, repo, *, error=None, new_provider_calls=0):
    checks = {}
    if "error" in expected:
        return {"expected_host_error": error == expected["error"], "no_model_call": new_provider_calls == 0}
    if error or result is None:
        return {"unexpected_error": False}
    checks["status"] = result.status == expected.get("status", "completed")
    checks["cached"] = result.cached == expected.get("cached", False)
    if result.cached:
        checks["cached_no_provider_call"] = new_provider_calls == 0
    requested = [c["name"] for d in result.decisions for c in d["tool_calls"]]
    invoked = [r["tool_name"] for r in result.execution_records if r["invoked"]]
    checks["model_tool_choices"] = requested == expected["tools"]
    checks["actual_tools"] = invoked == expected["tools"]
    checks["tool_success"] = all(r["status"] == "success" for r in result.execution_records)
    checks["permission_boundary"] = all(r.get("business_authorized") is not False for r in result.execution_records)
    if "route" in expected:
        checks["route"] = result.route == expected["route"]
    if "stop_reason" in expected:
        checks["stop_reason"] = result.stop_reason == expected["stop_reason"]
    if "number" in expected:
        numbers = re.findall(r"(?<![A-Za-z0-9.])-?\d+(?:\.\d+)?(?![A-Za-z0-9.])", result.answer)
        checks["answer_number"] = any(abs(float(n) - expected["number"]) < 1e-9 for n in numbers)
    text = re.sub(r"\s+", "", result.answer)
    for word in expected.get("contains", []):
        checks["contains:" + word] = word in text
    for word in expected.get("forbidden", []):
        checks["forbidden:" + word] = word not in text
    if "contains_any" in expected:
        checks["clarification"] = any(w in text for w in expected["contains_any"])
    if "arguments" in expected:
        checks["arguments"] = [r["arguments"] for r in result.execution_records] == expected["arguments"]
    if "ticket_count" in expected:
        with repo.connection() as conn:
            checks["ticket_count"] = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == expected["ticket_count"]
    if expected.get("actual_ticket_id"):
        reads = [r for r in result.execution_records if r["tool_name"] == "get_ticket"]
        checks["actual_ticket_id"] = bool(reads) and all(
            repo.get_own_ticket(r["arguments"].get("ticket_id"), result.permission_snapshot["identity"]["employee_id"])
            and r["arguments"]["ticket_id"] in result.answer for r in reads)
    if expected.get("dependent_steps"):
        steps = [r["model_step"] for r in result.execution_records]
        checks["dependent_model_steps"] = len(steps) >= 2 and all(a < b for a, b in zip(steps, steps[1:]))
    if expected.get("citation"):
        from app.citation_validator import extract_citations
        refs = extract_citations(result.answer)
        available = {(c["doc_id"], c["chunk_id"]) for c in result.knowledge_evidence}
        checks["citation_structure"] = bool(refs) and all((r.doc_id, r.chunk_id) in available for r in refs)
    return checks


def run_story(case, directory, model, rag, budget, transport):
    repo = SQLiteBusinessRepository(directory / "business.sqlite3")
    repo.seed_demo()
    def factory(context):
        if transport == "local":
            return build_tools(user_context=context, repository=repo)
        from app.mcp_tool_registry import build_mcp_tools
        return build_mcp_tools(user_context=context, database_path=repo.path)
    tool_agent = ToolAgent(model, build_tools(repository=repo), tools_factory=factory)
    unified = SimpleNamespace(chat_model=model, tool_agent=tool_agent, rag_agent=rag)
    db = directory / "conversation.sqlite3"
    service = ConversationAgent(unified, ConversationStore(db), repo, **case.get("limits", {}))
    sessions, history_ids, steps = {}, [], []
    for index, expected in enumerate(case["turns"], 1):
        actor = load_user_context(expected.get("actor", "employee-e001"))
        name = expected.get("session", "main")
        if name not in sessions:
            sessions[name] = service.create_session(user_context=actor)
        if expected.get("restart"):
            service = ConversationAgent(unified, ConversationStore(db), repo, **case.get("limits", {}))
        if "set_leave" in expected:
            with repo.connection() as conn:
                conn.execute("UPDATE employees SET leave_balance=? WHERE employee_id=?", (expected["set_leave"], actor.employee_id))
        turn_id = history_ids[expected["repeat_turn"] - 1] if "repeat_turn" in expected else f"turn-{index}"
        history_ids.append(turn_id)
        before, result, error = budget.used, None, None
        from app.tool_control_evaluation import database_snapshot
        state_before = database_snapshot(repo)
        with capture_trace() as trace:
            try:
                result = service.run(expected["query"], session_id=sessions[name], turn_id=turn_id, user_context=actor)
            except BusinessError as exc:
                error = exc.code
            except Exception as exc:
                error = type(exc).__name__
        used = budget.used - before
        checks = grade_turn(expected, result, repo, error=error, new_provider_calls=used)
        steps.append({"turn": index, "expected": expected, "result": asdict(result) if result else None,
                      "database_before": state_before, "database_after": database_snapshot(repo),
                      "error": error, "checks": checks, "passed": all(checks.values()), "provider_calls": used, "trace": trace})
        print(f"{transport} {case['case_id']} turn {index}: {'passed' if all(checks.values()) else 'FAILED'}", flush=True)
        if budget.used >= budget.maximum:
            break
    return {"case_id": case["case_id"], "category": case["category"], "transport": transport, "turns": steps,
            "passed": len(steps) == len(case["turns"]) and all(s["passed"] for s in steps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--transport", choices=["local", "mcp", "both"], default="both")
    parser.add_argument("--max-calls", type=int, default=420)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--run-models", action="store_true", help="明确启用付费模型调用")
    args = parser.parse_args()
    if not args.run_models:
        parser.error("真实模型实验需要--run-models")
    data = load_cases()
    cases = [c for c in data["cases"] if not args.case_ids or c["case_id"] in args.case_ids]
    if not cases or (args.case_ids and set(args.case_ids) != {c["case_id"] for c in cases}):
        parser.error("会话用例ID无效")
    transports = ["local", "mcp"] if args.transport == "both" else [args.transport]
    run = RunStore(args.output, {"suite": "conversation_v1", "dataset_version": data["version"],
                               "data_sha256": sha256(DATA), "case_ids": [c["case_id"] for c in cases],
                               "transports": transports, "max_calls": args.max_calls, "provenance": provenance()})
    if run.rows():
        raise ValueError("此评测目录已有记录，请使用新的输出目录，保留原始实验")
    budget = CallBudget(args.output / "calls.jsonl", args.max_calls)
    from app.agent_factory import build_chat_model, build_knowledge_retriever, build_rag_agent
    model = attach_budget(build_chat_model(), budget)
    rag = build_rag_agent(model, build_knowledge_retriever()) if any("rag" in c["category"] for c in cases) else None
    for transport in transports:
        for case in cases:
            row = run_story(case, args.output / transport / case["case_id"], model, rag, budget, transport)
            run.append(row)
            if budget.used >= args.max_calls:
                break
        if budget.used >= args.max_calls:
            break
    rows = run.rows()
    total = len(cases) * len(transports)
    summary = {"expected_conversations": total, "completed_conversations": len(rows),
               "passed_conversations": sum(r["passed"] for r in rows),
               "expected_turns": sum(len(c["turns"]) for c in cases) * len(transports),
               "completed_turns": sum(len(r["turns"]) for r in rows),
               "passed_turns": sum(t["passed"] for r in rows for t in r["turns"]),
               "provider_attempts": budget.used,
               "failed": [{"transport": r["transport"], "case_id": r["case_id"], "turn": t["turn"],
                           "checks": [k for k, v in t["checks"].items() if not v]} for r in rows for t in r["turns"] if not t["passed"]]}
    run.report("summary", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed_conversations"] == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
