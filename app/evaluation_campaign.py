"""Frozen, resumable full-corpus collection and independent Judge rescoring.

One process owns each output directory. Worker threads never write RunStore.
Do not tune prompts/rules on holdout results during a frozen campaign.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path

from app.calibration import judge_versions, calibration_inputs, calibration_report
from app.dataset_governance import ROOT, read_jsonl, sha256, validate_dataset
from app.evaluation_records import RunStore, CallBudget, attach_budget, fingerprint, append_jsonl, provenance
from app.judge_model import judge_model_config, build_judge_model
from app.tracing import capture_trace


def provider_error_details(error):
    """Retain status even when an OpenAI-compatible provider uses a generic code."""
    cause = error.__cause__ or error
    body = getattr(cause, "body", None)
    body = body.get("error", body) if isinstance(body, dict) else {}
    body = body if isinstance(body, dict) else {}
    return {"error_type": type(error).__name__, "cause_type": type(cause).__name__,
            "error_code": getattr(cause, "code", None) or body.get("code"),
            "error_http_status": getattr(cause, "status_code", None), "provider_error_type": body.get("type")}


def planned_jobs():
    metadata = {(m["suite"], m["case_id"]): m for m in read_jsonl(ROOT / "data/evaluation/case_metadata.jsonl")}
    jobs = []
    for suite, filename in (("rag", "retrieval"), ("routing", "routing"), ("security", "security")):
        for case in read_jsonl(ROOT / f"data/evaluation/{filename}_cases.jsonl"):
            meta = metadata[(filename, case["case_id"])]
            for variant in (("direct", "rewrite") if suite == "rag" else ("default",)):
                jobs.append({"suite": suite, "split": meta["split"], "case": case,
                             "group_id": meta["group_id"], "variant": variant, "trial_id": 0})
    baseline = read_jsonl(ROOT / "data/evaluation/baseline_v1/retrieval_cases.jsonl")
    jobs += [{"suite": "rag_diagnostic", "split": "baseline", "case": case,
              "variant": "rewrite", "trial_id": trial} for trial in range(3) for case in baseline]
    for job in jobs:
        job["job_id"] = fingerprint(job)
    return jobs


def historical_ab_config():
    source = "reports/ab-dev-v2.0.1"
    records = ROOT / source / "records.jsonl"
    return {"historical_ab": source if records.is_file() else None,
            "historical_ab_sha256": sha256(records) if records.is_file() else None}


def campaign_config():
    checked = validate_dataset()
    if checked["errors"]:
        raise ValueError(checked["errors"])
    return {"kind": "full_corpus_sol_judge_2026_11_14", "jobs": planned_jobs(),
            "agent_model": "deepseek-v4-flash", "agent_temperature": 0,
            "judge": {s: judge_model_config(s) for s in ("rag", "security")}, "judge_versions": judge_versions(),
            "collection_trials": 1, "diagnostic_trials": 3, "calibration_trials": 2,
            **historical_ab_config(), "security_tool_profile": "legacy",
            "security_identity": None, "k": 5, "provenance": provenance(),
            "runner_sha256": sha256(Path(__file__)),
            "calibration_inputs_hash": fingerprint(calibration_inputs(read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl"))),
            "holdout_policy": "Frozen before collection. No prompt, rule, data, or threshold tuning on these results."}


def open_campaign(output, freeze=False):
    if freeze:
        return RunStore(output, campaign_config())
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    current = campaign_config()
    if fingerprint(current) != manifest["config_hash"]:
        raise ValueError("Frozen code/data/config changed; start a separate campaign")
    return RunStore(output, manifest["config"])


def rag_payload(result):
    return {"answer": result.final_response.content, "context": result.context,
            "retrieval_query": result.retrieval_query,
            "retrieval": [{"doc_id": r.document.metadata["doc_id"], "chunk_id": r.document.metadata["chunk_id"],
                           "content": r.document.page_content, "rank": r.rank, "score": r.score} for r in result.retrieval_results],
            "prompt": [{"role": m.type, "content": m.content} for m in result.prompt_message]}


def execute_job(job, runtime):
    from app.retrieval_case_loader import RetrievalCase
    from app.rag_evaluator import evaluate_rag_answer
    from app.retrieval_metrics import recall_at_k, reciprocal_rank, evidence_recall_at_k
    from app.security_case_loader import SecurityCase
    from app.security_retriever import InjectedContentRetriever
    from app.agent_factory import build_rag_agent
    from app.security_evaluator import evaluate_security_rules, find_unauthorized_tool_names, is_security_route_satisfied
    row = {**job, "case_id": job["case"]["case_id"], "status": "error"}
    with capture_trace() as events:
        try:
            if job["suite"] == "routing":
                result = runtime["router"].route(job["case"]["query"])
                row.update(status="completed", actual_route=result.route.value, reason=result.reason,
                           passed=result.route.value == job["case"]["expected_route"])
            elif job["suite"].startswith("rag"):
                case = RetrievalCase(**job["case"])
                result = runtime[job["variant"]].run(case.question)
                rule = evaluate_rag_answer(case, result.final_response.content, result.retrieval_results)
                metrics = None
                if case.answerable:
                    docs = [r.document.metadata["doc_id"] for r in result.retrieval_results]
                    contents = [r.document.page_content for r in result.retrieval_results]
                    metrics = {"recall_at_5": recall_at_k(docs, case.relevant_doc_ids, 5),
                               "mrr": reciprocal_rank(docs, case.relevant_doc_ids),
                               "evidence_recall_at_5": evidence_recall_at_k(contents, case.required_evidence, 5)}
                row.update(status="completed", **rag_payload(result), rules=asdict(rule),
                           structural_passed=rule.deterministic_checks_passed, metrics=metrics)
            else:
                case = SecurityCase.model_validate(job["case"])
                if case.target.value == "rag":
                    retriever = InjectedContentRetriever(runtime["retriever"], case)
                    agent = build_rag_agent(runtime["model"], retriever)
                    result = agent.run(case.query)
                    row.update(**rag_payload(result), actual_route="knowledge", tool_names=[], requested_tool_names=[], blocked_tool_names=[])
                else:
                    if case.identity_permissions:
                        from app.identity import load_user_context
                        result = runtime["unified"].run(case.query, user_context=load_user_context(case.actor))
                    else:
                        result = runtime["unified"].run(case.query, allowed_tool_names=case.allowed_tool_names)
                    if result.status.value != "success":
                        raise RuntimeError("Agent execution degraded")
                    tools = result.tool_result
                    row.update(answer=result.final_response.content, actual_route=result.route_decision.route.value,
                               tool_names=tools.executed_tool_names if tools else [],
                               requested_tool_names=tools.requested_tool_names if tools else [],
                               blocked_tool_names=tools.blocked_tool_names if tools else [],
                               execution_records=tools.execution_records if tools else [],
                               tool_calls=tools.decision.tool_calls if tools else [],
                               permission_snapshot=tools.permission_snapshot if tools else {})
                    if result.rag_result:
                        row.update(**rag_payload(result.rag_result))
                rules = evaluate_security_rules(case, row["answer"])
                if case.identity_permissions:
                    from app.security_executors import SecurityExecutionTrace
                    from app.security_evaluator import evaluate_permission_trace
                    row["business_events"] = [e for e in events if e["stage"] in {"authorization", "business"}]
                    evidence = SecurityExecutionTrace(row["answer"], row.get("actual_route"), row.get("tool_names", []),
                                                      row.get("requested_tool_names", []), row.get("blocked_tool_names", []),
                                                      row.get("tool_calls", []), row.get("permission_snapshot", {}),
                                                      row.get("execution_records", []), row["business_events"])
                    facts = evaluate_permission_trace(case, evidence)
                    row["permission_checks"] = facts
                row.update(status="completed", rules=asdict(rules), rule_passed=rules.passed,
                           unauthorized_tools=facts["actual_permission_violations"] if case.identity_permissions else find_unauthorized_tool_names(case, row["tool_names"]),
                           route_satisfied=is_security_route_satisfied(case, row["actual_route"]))
        except Exception as error:
            row.update(**provider_error_details(error))
    row["trace"] = events
    return row


def collect(store, workers, maximum):
    from app.agent_factory import build_chat_model, build_knowledge_retriever, build_rag_agent
    from app.intent_router import IntentRouter
    from app.tool_agent import ToolAgent
    from app.tool_registry import build_tools
    from app.business_repository import SQLiteBusinessRepository
    from app.unified_agent import UnifiedAgent
    model = attach_budget(build_chat_model(), CallBudget(store.directory / "generation-calls.jsonl", maximum))
    retriever = build_knowledge_retriever()
    runtime = {"model": model, "retriever": retriever, "router": IntentRouter(model),
               **{v: build_rag_agent(model, retriever, rewrite_queries=v == "rewrite") for v in ("direct", "rewrite")}}
    repository = SQLiteBusinessRepository(store.directory / "isolated-business.sqlite3")
    repository.seed_demo()
    runtime["unified"] = UnifiedAgent(runtime["router"], ToolAgent(model, build_tools(repository=repository),
                                      tools_factory=lambda user: build_tools(repository=repository, user_context=user)), runtime["rewrite"], model)
    seen = {r["job_id"] for r in store.rows()}
    jobs = [j for j in store.manifest["config"]["jobs"] if j["job_id"] not in seen]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute_job, j, runtime) for j in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            store.append(row)
            print(json.dumps({"phase": "collect", "done": len(seen) + index, "expected": len(store.manifest["config"]["jobs"]),
                              "suite": row["suite"], "case_id": row["case_id"], "status": row["status"]}), flush=True)


def scoring_tasks(store):
    tasks = []
    for row in store.rows():
        if row["status"] == "completed" and row["suite"] != "routing":
            tasks.append({"score_id": row["job_id"], "source": "current_campaign", "suite": row["suite"],
                          "case_id": row["case_id"], "input": row})
    historical = store.manifest["config"].get("historical_ab")
    historical_rows = []
    if historical is not None:
        records = ROOT / historical / "records.jsonl"
        if not records.is_file() or sha256(records) != store.manifest["config"]["historical_ab_sha256"]:
            raise ValueError("Historical answers changed or are missing")
        historical_rows = read_jsonl(records)
    for row in historical_rows:
        tasks.append({"score_id": "historical-" + row["record_id"], "source": "historical_ab",
                      "suite": "rag", "case_id": row["case_id"], "input": row})
    samples = read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl")
    for trial in range(2):
        for sample in samples:
            tasks.append({"score_id": f"calibration-{sample['sample_id']}-{trial}", "source": "calibration",
                          "suite": sample["suite"], "sample_id": sample["sample_id"], "trial_id": trial,
                          "input": sample})
    return tasks


def score_task(task, rag_judge, security_judge):
    from app.security_case_loader import SecurityCase
    from app.rag_agent import INSUFFICIENT_KNOWLEDGE_RESPONSE
    row = {k: v for k, v in task.items() if k != "input"}
    inputs = task["input"]
    row.update(status="error", passed=False, input_hash=fingerprint(inputs))
    with capture_trace() as events:
        try:
            if task["suite"].startswith("rag"):
                case = inputs.get("case", inputs)
                result = rag_judge.evaluate(case["question"], case.get("reference_answer") or INSUFFICIENT_KNOWLEDGE_RESPONSE,
                                            inputs["answer"], inputs["context"])
                version = judge_versions()["rag"]
            else:
                case = inputs.get("case", inputs.get("security_case"))
                parsed = SecurityCase.model_validate(case)
                options = {"execution_evidence": inputs.get("execution_evidence") or {key: inputs.get(key, [] if key != "permission_snapshot" else {}) for key in
                                                   ("tool_calls", "permission_snapshot", "execution_records", "business_events")}} if parsed.identity_permissions else {}
                result = security_judge.evaluate(parsed, inputs["answer"], inputs.get("actual_route"), inputs.get("tool_names", []), **options)
                version = judge_versions()["security"]
            row.update(status="completed", passed=result.passed, scores=result.model_dump(), judge_version=version)
        except Exception as error:
            cause = error.__cause__ or error
            row.update(**provider_error_details(error))
    row["trace"] = events
    return row


def score(store, workers, maximum):
    from app.llm_judge import LLMJudge
    from app.security_judge import SecurityLLMJudge
    budget = CallBudget(store.directory / "judge-calls.jsonl", maximum)
    rag = LLMJudge(attach_budget(build_judge_model("rag"), budget))
    security = SecurityLLMJudge(attach_budget(build_judge_model("security"), budget))
    path = store.directory / "scores.jsonl"
    prior = read_jsonl(path) if path.exists() else []
    if any(r.get("judge_version") not in judge_versions().values() for r in prior if r["status"] == "completed"):
        raise ValueError("Judge changed; use a new campaign")
    seen = {r["score_id"] for r in prior}
    tasks = [t for t in scoring_tasks(store) if t["score_id"] not in seen]
    # Bounded batches allow a permanent quota error to stop before submitting hundreds of calls.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(tasks), workers):
            futures = [pool.submit(score_task, t, rag, security) for t in tasks[offset:offset + workers]]
            stop = False
            for future in as_completed(futures):
                row = future.result()
                append_jsonl(path, row)
                print(json.dumps({"phase": "score", "score_id": row["score_id"], "status": row["status"]}), flush=True)
                stop |= row.get("error_code") in {"credit_balance_exhausted", "insufficient_quota"}
            if stop:
                break


def summarize(store):
    rows = store.rows()
    tasks = scoring_tasks(store)
    score_path = store.directory / "scores.jsonl"
    scores = read_jsonl(score_path) if score_path.exists() else []
    indexed = {s["score_id"]: s for s in scores}
    if len(indexed) != len(scores) or len({r["job_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate campaign rows")
    groups = {}
    for job in store.manifest["config"]["jobs"]:
        key = f"{job['suite']}/{job['split']}/{job['variant']}"
        groups.setdefault(key, {"expected": 0})["expected"] += 1
    for key, values in groups.items():
        subset = [r for r in rows if f"{r['suite']}/{r['split']}/{r['variant']}" == key]
        complete = [r for r in subset if r["status"] == "completed"]
        grades = [indexed[r["job_id"]] for r in subset if r["job_id"] in indexed]
        values.update(executed=len(subset), completed=len(complete), errors=len(subset)-len(complete),
                      missing=values["expected"] - len(subset),
                      judge_completed=sum(g["status"] == "completed" for g in grades),
                      judge_errors=sum(g["status"] != "completed" for g in grades),
                      judge_passed=sum(g["passed"] for g in grades))
        if key.startswith("routing"):
            values["passed"] = sum(r.get("passed", False) for r in subset)
        elif key.startswith("rag"):
            values["structural_passed"] = sum(r.get("structural_passed", False) for r in subset)
            answerable = [r for r in subset if r["case"]["answerable"]]
            values["answerable_count"] = len(answerable)
            for metric in ("recall_at_5", "mrr", "evidence_recall_at_5"):
                values[metric] = sum((r.get("metrics") or {}).get(metric, 0) for r in answerable) / len(answerable) if answerable else None
        else:
            values.update(rule_passed=sum(r.get("rule_passed", False) for r in subset),
                          route_satisfied=sum(r.get("route_satisfied", False) for r in subset),
                          unauthorized_execution_cases=sum(bool(r.get("unauthorized_tools")) for r in subset),
                          passed=sum(r.get("rule_passed", False) and r.get("route_satisfied", False) and not r.get("unauthorized_tools") and indexed.get(r["job_id"], {}).get("passed", False) for r in subset))
    failures = [{"job_id": r["job_id"], "suite": r["suite"], "split": r["split"], "variant": r["variant"],
                 "case_id": r["case_id"], "trial_id": r["trial_id"], "status": r["status"], "answer": r.get("answer"),
                 "rules": r.get("rules"), "judge": indexed.get(r["job_id"])} for r in rows
                if r["status"] != "completed" or r.get("passed") is False or r.get("structural_passed") is False
                or r.get("rule_passed") is False or r.get("route_satisfied") is False or indexed.get(r["job_id"], {}).get("passed") is False]
    report = {"groups": groups, "generation_rows": len(rows), "generation_expected": len(store.manifest["config"]["jobs"]),
              "judge_expected": len(tasks), "judge_completed": sum(s["status"] == "completed" for s in scores),
              "judge_errors": sum(s["status"] != "completed" for s in scores), "failures": failures,
              "judge_status": "provisional_mixed_review_calibration", "judge_config": {s: judge_model_config(s) for s in ("rag", "security")}}
    for name in ("generation", "judge"):
        path = store.directory / f"{name}-calls.jsonl"
        report[f"{name}_provider_attempts"] = len(read_jsonl(path)) if path.exists() else 0
    historical = [s for s in scores if s["source"] == "historical_ab"]
    report["historical_rescore"] = {"expected": sum(t["source"] == "historical_ab" for t in tasks),
                                    "completed": sum(s["status"] == "completed" for s in historical),
                                    "passed": sum(s["passed"] for s in historical)}
    store.report("summary", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["freeze", "collect", "score", "report"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/full-evaluation-sol-v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    store = open_campaign(args.output, args.phase == "freeze")
    if args.phase == "collect":
        collect(store, args.workers, args.max_calls)
    elif args.phase == "score":
        score(store, args.workers, args.max_calls)
    report = summarize(store)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
