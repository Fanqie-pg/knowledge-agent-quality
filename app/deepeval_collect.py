"""Collect reproducible current sources for the unified DeepEval entry point."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from app.dataset_governance import ROOT, read_jsonl
from app.evaluation_records import CallBudget, RunStore, provenance


def collect_model_branches(output, suites):
    from app.agent_factory import build_chat_model, build_knowledge_retriever, build_rag_agent
    from app.evaluation_campaign import planned_jobs, execute_job
    from app.business_repository import SQLiteBusinessRepository
    from app.intent_router import IntentRouter
    from app.tool_agent import ToolAgent
    from app.tool_registry import build_tools
    from app.unified_agent import UnifiedAgent
    jobs = [j for j in planned_jobs() if j["suite"] in suites]
    store = RunStore(output, {"kind": "deepeval-current-agent-sources", "jobs": jobs, "provenance": provenance()})
    model, retriever = build_chat_model(), build_knowledge_retriever()
    repo = SQLiteBusinessRepository(output / "business.sqlite3")
    repo.seed_demo()
    router = IntentRouter(model)
    tools = ToolAgent(model, build_tools(repository=repo), tools_factory=lambda ctx: build_tools(user_context=ctx, repository=repo))
    rewrite = build_rag_agent(model, retriever)
    runtime = {"model": model, "retriever": retriever, "router": router, "rewrite": rewrite,
               "direct": build_rag_agent(model, retriever, rewrite_queries=False),
               "unified": UnifiedAgent(router, tools, rewrite, model)}
    completed = {r["job_id"] for r in store.rows()}
    for job in jobs:
        if job["job_id"] not in completed:
            row = execute_job(job, runtime)
            store.append(row)
            print(f"collect {job['suite']} {job['case']['case_id']} {row['status']}", flush=True)
            if row.get("error_type") == "CallBudgetExceeded" or row.get("error_http_status") in {401, 402, 403}:
                break
    rows = store.rows()
    report = {"expected": len(jobs), "recorded": len(rows), "completed": sum(r["status"] == "completed" for r in rows)}
    store.report("collection", report)
    return report["completed"] == len(jobs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--suites", nargs="+", choices=["rag", "routing", "security", "tools", "business", "conversation"],
                        default=["rag", "routing", "security", "tools", "business", "conversation"])
    args = parser.parse_args(argv)
    if not args.run_models: parser.error("Collection requires --run-models")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    CallBudget(output / "calls.jsonl", args.max_calls)
    os.environ.update(EVALUATION_MAX_CALLS=str(args.max_calls), EVALUATION_CALL_LEDGER=str(output / "calls.jsonl"),
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from app.deepeval_adapters import PROFILE
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    branches = set(args.suites) & {"rag", "routing", "security"}
    if branches:
        if not collect_model_branches(output / "model-branches", branches): return 1
        source = str(output / "model-branches/records.jsonl")
        for key in branches:
            if key == "rag": profile.update(rag_direct=source, rag_rewrite=source)
            else: profile[key] = source
    commands = {"tools": ["app.tool_control_evaluation", "--phase", "all", "--trials", "3", "--workers", "2"],
                "business": ["app.business_evaluation"], "conversation": ["app.conversation_evaluation", "--transport", "both"]}
    expected = {"tools": 120, "business": 38, "conversation": 32}
    for suite in [s for s in args.suites if s in commands]:
        folder = output / suite
        records = folder / "records.jsonl"
        if not records.exists():
            command = [sys.executable, "-m", *commands[suite], "--run-models", "--max-calls", str(args.max_calls), "--output", str(folder)]
            print("collect " + suite, flush=True)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            (output / (suite + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if not records.exists() or len(read_jsonl(records)) != expected[suite]:
            print("Collection incomplete; retain artifacts and use a new output directory for a fresh stateful run.")
            return 1
        profile[suite] = [str(records)] if suite == "conversation" else str(records)
    (output / "sources.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_profile": str(output / "sources.json"), "provider_attempts": len(read_jsonl(output / "calls.jsonl"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
