"""Role-specific Qwen rescoring of immutable saved answers, with explicit lineage."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from app.calibration import calibration_report, judge_versions, RAG_FIELDS, SECURITY_FIELDS, label_passed
from app.dataset_governance import ROOT, read_jsonl, sha256
from app.evaluation_records import RunStore, CallBudget, attach_budget, fingerprint
from app.evaluation_campaign import scoring_tasks, score_task
from app.judge_model import build_judge_model, judge_model_config

GENERATION_FILES = ("agent_factory", "rag_agent", "query_rewriter", "retriever", "document_loader", "document_splitter",
                    "intent_router", "unified_agent", "tool_agent", "tool_registry", "business_service", "business_repository", "identity")


def source_store(directory):
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    store = RunStore(directory, manifest["config"])
    for name in GENERATION_FILES:
        relative = f"app/{name}.py"
        if sha256(ROOT / relative) != manifest["config"]["provenance"]["files"][relative]:
            raise ValueError(f"Agent generation implementation changed: {name}")
    for relative, digest in manifest["config"]["provenance"]["files"].items():
        if relative.startswith("data/") and sha256(ROOT / relative) != digest:
            raise ValueError("Source knowledge or dataset changed")
    if len(store.rows()) != 216 or any(r["status"] != "completed" for r in store.rows()):
        raise ValueError("Source campaign is incomplete")
    return store


def open_rescore(output, source):
    tasks = scoring_tasks(source)
    if len(tasks) != 540 or len({t["score_id"] for t in tasks}) != 540:
        raise ValueError("Unexpected or duplicate scoring tasks")
    config = {"kind": "qwen_dual_judge_rescore", "source_run_id": source.manifest["run_id"],
              "source_records_sha256": sha256(source.path), "source_manifest_sha256": sha256(source.directory / "run.json"),
              "judges": {s: judge_model_config(s) for s in ("rag", "security")}, "judge_versions": judge_versions(),
              "task_inputs": {t["score_id"]: fingerprint(t["input"]) for t in tasks},
              "samples_sha256": sha256(ROOT / "data/evaluation/calibration/samples.jsonl"),
              "reference_labels_sha256": sha256(ROOT / "data/evaluation/calibration/references.jsonl"),
              "implementation": {n: sha256(ROOT / "app" / n) for n in ("judge_rescore.py", "judge_model.py", "llm_judge.py", "security_judge.py", "calibration.py", "evaluation_campaign.py")},
              "generation_calls": 0, "label_sources": {"user_provided": 34, "assistant_generated": 26}}
    return RunStore(output, config), tasks


def run(store, tasks, phase, workers, maximum):
    from app.llm_judge import LLMJudge
    from app.security_judge import SecurityLLMJudge
    budget = CallBudget(store.directory / "calls.jsonl", maximum)
    rag = LLMJudge(attach_budget(build_judge_model("rag"), budget))
    security = SecurityLLMJudge(attach_budget(build_judge_model("security"), budget))
    seen = {r["score_id"] for r in store.rows()}
    pending = [t for t in tasks if t["score_id"] not in seen and (t["source"] == "calibration") == (phase == "calibration")]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(pending), workers):
            futures = [pool.submit(score_task, t, rag, security) for t in pending[offset:offset+workers]]
            stop = False
            for future in as_completed(futures):
                row = future.result()
                store.append(row)
                print(json.dumps({"phase": phase, "score_id": row["score_id"], "suite": row["suite"],
                                  "status": row["status"], "passed": row["passed"]}), flush=True)
                stop |= row.get("error_code") in {"credit_balance_exhausted", "insufficient_quota", "InvalidApiKey", "Arrearage", "CallBudgetExceeded"}
            if stop:
                break


def report(store, tasks):
    rows = store.rows()
    ids = [r["score_id"] for r in rows]
    expected = {t["score_id"] for t in tasks}
    if len(ids) != len(set(ids)) or not set(ids) <= expected:
        raise ValueError("Invalid grading records")
    for row in rows:
        if row["input_hash"] != store.manifest["config"]["task_inputs"][row["score_id"]]:
            raise ValueError("Scored input changed")
        suite = "rag" if row["suite"].startswith("rag") else "security"
        if row["status"] == "completed" and row["judge_version"] != store.manifest["config"]["judge_versions"][suite]:
            raise ValueError("Mixed evaluator versions")
    samples = read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl")
    references = read_jsonl(ROOT / "data/evaluation/calibration/references.jsonl")
    cal_rows = []
    for row in rows:
        if row["source"] != "calibration":
            continue
        result = {k: row[k] for k in ("sample_id", "suite", "trial_id", "status")}
        if row["status"] == "completed":
            fields = RAG_FIELDS if row["suite"] == "rag" else SECURITY_FIELDS
            result.update(labels={k: row["scores"][k] for k in fields}, reason=row["scores"]["reason"])
        cal_rows.append(result)
    calibration = calibration_report(samples, cal_rows, judge_versions(), trials=2, references=references)
    assistant_index = {r["sample_id"]: r for r in references if r["origin"] == "assistant_generated"}
    assistant_comparisons = [r for r in cal_rows if r["status"] == "completed" and r["sample_id"] in assistant_index]
    calibration["assistant_comparison"] = {"prediction_count": len(assistant_comparisons),
        "agreement": sum(label_passed(r["suite"], r["labels"]) == label_passed(r["suite"], assistant_index[r["sample_id"]]["labels"]) for r in assistant_comparisons) / len(assistant_comparisons) if assistant_comparisons else None,
        "independent_human_evidence": False}
    store.report("calibration", calibration)
    indexed = {r["score_id"]: r for r in rows}
    cohorts = {}
    for task in tasks:
        inputs = task["input"]
        if task["source"] == "current_campaign":
            key = f"{task['suite']}/{inputs['split']}/{inputs['variant']}"
        elif task["source"] == "historical_ab":
            key = "historical_ab/" + inputs["variant"]
        else:
            key = "calibration/" + task["suite"]
        cohort = cohorts.setdefault(key, {"expected": 0, "completed": 0, "errors": 0, "judge_passed": 0})
        cohort["expected"] += 1
        row = indexed.get(task["score_id"])
        if row:
            cohort["completed"] += row["status"] == "completed"
            cohort["errors"] += row["status"] != "completed"
            cohort["judge_passed"] += row["passed"]
        if task["source"] == "current_campaign" and task["suite"] == "security":
            cohort.setdefault("combined_passed", 0)
            cohort["combined_passed"] += bool(row and row["passed"] and inputs["rule_passed"] and inputs["route_satisfied"] and not inputs["unauthorized_tools"])
    summary = {"expected": len(tasks), "executed": len(rows), "completed": sum(r["status"] == "completed" for r in rows),
               "errors": sum(r["status"] != "completed" for r in rows), "missing": len(tasks) - len(rows),
               "cohorts": cohorts, "judges": store.manifest["config"]["judges"],
               "provider_attempts": len(read_jsonl(store.directory / "calls.jsonl")) if (store.directory / "calls.jsonl").exists() else 0,
               "generation_calls": 0, "review_completion_status": calibration["review_completion_status"],
               "human_certification": calibration["status"], "human_validation_agreement": calibration["agreement"],
               "interpretation": "New evaluator scores of frozen answers; limited independent human validation. No Agent quality gain implied."}
    store.report("summary", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["calibration", "regrade", "report"], required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/full-evaluation-sol-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/qwen-dual-judge-v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=1100)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1..4")
    source = source_store(args.source)
    store, tasks = open_rescore(args.output, source)
    if args.phase != "report":
        run(store, tasks, args.phase, args.workers, args.max_calls)
    summary = report(store, tasks)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
