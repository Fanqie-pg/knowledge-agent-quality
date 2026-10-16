"""Bounded live security/calibration/basic regressions for identity permissions."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from app.calibration import SECURITY_FIELDS, calibration_inputs, judge_versions
from app.dataset_governance import ROOT, read_jsonl, sha256
from app.evaluation_campaign import execute_job, planned_jobs, score_task
from app.evaluation_records import CallBudget, RunStore, attach_budget, fingerprint, provenance
from app.tracing import capture_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["security", "calibration", "basics"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-calls", type=int, default=160)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers必须在1到4之间")
    samples = [s for s in read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl") if s["suite"] == "security"]
    if args.phase == "calibration":
        jobs = [{"sample": s, "trial_id": trial, "case_id": s["sample_id"], "suite": "security"} for trial in range(2) for s in samples]
    elif args.phase == "security":
        jobs = [job for job in planned_jobs() if job["suite"] == "security"]
    else:
        source = planned_jobs()
        route_jobs = [next(j for j in source if j["suite"] == "routing" and j["case"]["expected_route"] == route) for route in ("chat", "knowledge", "tool")]
        jobs = route_jobs + [j for j in source if j["suite"] == "rag" and j["variant"] == "rewrite" and j["case"]["case_id"] in {"RET-001", "RET-048", "RET-056"}]
    jobs = [{**job, "job_id": fingerprint(job)} for job in jobs]
    config = {"kind": "identity_permission_v2_" + args.phase, "jobs": jobs, "provenance": provenance(),
              "judge_versions": judge_versions(), "calibration_inputs_hash": fingerprint(calibration_inputs(samples)),
              "model": "deepseek-v4-flash", "workers": args.workers, "runner_sha256": sha256(Path(__file__))}
    store = RunStore(args.output, config)
    budget = CallBudget(args.output / "calls.jsonl", args.max_calls)
    from app.agent_factory import build_chat_model, build_knowledge_retriever, build_rag_agent
    from app.business_repository import SQLiteBusinessRepository
    from app.intent_router import IntentRouter
    from app.tool_agent import ToolAgent
    from app.tool_registry import build_tools
    from app.unified_agent import UnifiedAgent
    from app.judge_model import build_judge_model
    from app.security_judge import SecurityLLMJudge
    from app.llm_judge import LLMJudge
    security_judge = SecurityLLMJudge(attach_budget(build_judge_model("security"), budget)) if args.phase != "basics" else None
    rag_judge = LLMJudge(attach_budget(build_judge_model("rag"), budget)) if args.phase == "basics" else None
    runtime = None
    if args.phase != "calibration":
        model = attach_budget(build_chat_model(), budget)
        retriever = build_knowledge_retriever()
        router = IntentRouter(model)
        rag = build_rag_agent(model, retriever)
        repo = SQLiteBusinessRepository(args.output / "business.sqlite3")
        repo.seed_demo()
        tool = ToolAgent(model, build_tools(repository=repo), tools_factory=lambda user: build_tools(user_context=user, repository=repo))
        runtime = {"model": model, "retriever": retriever, "router": router, "rewrite": rag,
                   "unified": UnifiedAgent(router, tool, rag, model)}

    def run(job):
        if args.phase == "calibration":
            sample = job["sample"]
            grade = score_task({"score_id": job["job_id"], "source": "identity_v2_calibration", "suite": "security", "input": sample}, None, security_judge)
            expected = sample["suggested_labels"]
            grade.update(job_id=job["job_id"], case_id=sample["sample_id"], trial_id=job["trial_id"], expected_labels=expected,
                         agrees=grade["status"] == "completed" and all(grade["scores"][key] == expected[key] for key in SECURITY_FIELDS))
            return grade
        row = execute_job(job, runtime)
        row["passed"] = False
        if row["status"] == "completed":
            if job["suite"] == "routing":
                row["passed"] = row["actual_route"] == job["case"]["expected_route"]
            else:
                grade = score_task({"score_id": job["job_id"], "source": "identity_v2_live", "suite": job["suite"], "input": row}, rag_judge, security_judge)
                row["grade"] = grade
                row["passed"] = grade["status"] == "completed" and grade["passed"] and row.get("structural_passed", row.get("rule_passed", False))
                if job["suite"] == "security":
                    facts = row["permission_checks"]
                    row["passed"] &= row["route_satisfied"] and facts["model_decision_passed"] and facts["system_boundary_passed"]
                if grade["status"] != "completed":
                    row.update(status="error", **{key: grade.get(key) for key in ("error_type", "error_code", "error_http_status", "cause_type", "provider_error_type")})
        return row

    completed = {row["job_id"] for row in store.rows() if row["status"] == "completed"}
    pending = [job for job in jobs if job["job_id"] not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(pending), args.workers):
            futures = [pool.submit(run, job) for job in pending[offset:offset + args.workers]]
            stop = False
            for future in as_completed(futures):
                row = future.result()
                store.append(row)
                print(json.dumps({key: row.get(key) for key in ("case_id", "trial_id", "status", "passed", "agrees")}), flush=True)
                stop |= row.get("error_http_status") in {401, 402, 403} or row.get("error_code") in {"Arrearage", "insufficient_quota", "credit_balance_exhausted"} or row.get("error_type") == "CallBudgetExceeded"
            if stop:
                break
    rows = list({row["job_id"]: row for row in store.rows()}.values())
    success = lambda row: row["status"] == "completed" and (row.get("agrees", False) if args.phase == "calibration" else row["passed"])
    summary = {"phase": args.phase, "expected": len(jobs), "recorded": len(rows), "passed": sum(success(r) for r in rows),
               "errors": sum(r["status"] != "completed" for r in rows), "missing": len(jobs) - len(rows),
               "provider_calls": budget.used, "status": "passed" if len(rows) == len(jobs) and all(success(r) for r in rows) else "failed",
               "failures": [{"case_id": r.get("case_id"), "status": r["status"], "error_type": r.get("error_type"), "scores": r.get("scores"), "expected": r.get("expected_labels")} for r in rows if not success(r)],
               "limitations": "已知回归/合成校准；安全参考标签来自既有助手复核，不是独立人工认证。重复试次不视为独立问题。"}
    store.report("summary", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
