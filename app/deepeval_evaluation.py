"""Run the complete project corpus through DeepEval's native/custom metrics."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import shutil
from pathlib import Path
import time

from app.deepeval_runtime import FRAMEWORK_VERSION, PROTOCOL_VERSION
from app.deepeval_adapters import PROFILE, load_items
from app.deepeval_registry import metric_specs
from app.deepeval_model import QwenEvaluationModel
from app.dataset_governance import ROOT, sha256
from app.evaluation_records import RunStore, CallBudget, fingerprint, provenance
from app.evaluation_campaign import provider_error_details
from app.tracing import capture_trace, redact


def serializable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(x) for x in value]
    if hasattr(value, "value"):
        return value.value
    return value


def execute_metric(item, spec, *, attempt=1):
    start = time.perf_counter()
    record = {"item_key": item.key, "suite": item.suite, "metric": spec.name,
              "kind": spec.kind, "mandatory": spec.mandatory, "source": item.source,
              "input_hash": fingerprint(serializable(item.test_case)), "attempt": attempt,
              "threshold": spec.metric.threshold, "status": "error", "passed": False}
    record["judge_protocol"] = "native-json-thinking-off" if spec.kind == "native" else "project-policy-thinking-2048" if spec.kind == "policy" else "deterministic"
    with capture_trace() as trace:
        try:
            # Public DeepEval standalone lifecycle; one metric instance per job.
            # This avoids its global report manager being shared across threads.
            spec.metric.measure(item.test_case, _show_indicator=False)
            score = spec.metric.score
            if not isinstance(score, (float, int)) or not math.isfinite(score):
                raise ValueError("Metric returned a non-finite score")
            details = {key: serializable(getattr(spec.metric, key)) for key in
                       ("truths", "claims", "statements", "verdicts", "knowledges", "user_intentions", "score_breakdown")
                       if getattr(spec.metric, key, None) is not None}
            record.update(status="completed", score=score, passed=bool(spec.metric.is_successful()),
                          reason=spec.metric.reason, native_details=details,
                          evaluation_model=spec.metric.evaluation_model,
                          raw_labels=getattr(spec.metric, "raw_labels", None))
        except Exception as exc:
            record.update(**provider_error_details(exc))
    record.update(trace=trace, seconds=time.perf_counter() - start)
    return redact(record)


def latest_rows(rows):
    latest = {}
    for row in rows:
        key = row["job_id"]
        if key not in latest or latest[key]["status"] != "completed":
            latest[key] = row
    return latest


def calibration_summary(items, results):
    indexed = {i.key: i for i in items}
    groups = defaultdict(lambda: {"scored": 0, "agrees": 0, "false_pass": 0, "by_reference_source": Counter()})
    for row in results:
        item = indexed[row["item_key"]]
        if not item.suite.startswith("calibration-") or row["status"] != "completed":
            continue
        labels = item.source.get("labels")
        if not labels:
            continue
        expected = None
        exact = None
        if item.suite == "calibration-native":
            expected = labels["passed"]
        elif row["metric"] == "NativeFaithfulness":
            expected = labels["faithfulness"] >= 4
        elif row["metric"] in {"ProjectRAGPolicy", "ProjectSecurityPolicy"}:
            from app.calibration import label_passed
            expected = label_passed("rag" if item.suite == "calibration-rag" else "security", labels)
            exact = all(row["raw_labels"][k] == v for k, v in labels.items())
        if expected is None:
            continue
        key = item.suite + "/" + row["metric"] + "/" + item.source["split"]
        group = groups[key]
        group["scored"] += 1
        group["agrees"] += row["passed"] == expected
        group["false_pass"] += row["passed"] and not expected
        group["by_reference_source"][item.source["label_kind"]] += 1
        if exact is not None:
            group["exact_dimension_agrees"] = group.get("exact_dimension_agrees", 0) + exact
    return {k: {**v, "binary_agreement": v["agrees"] / v["scored"]} for k, v in groups.items()}


def summarize(items, jobs, rows, budget):
    final = latest_rows(rows)
    grouped = defaultdict(list)
    for row in final.values(): grouped[row["item_key"]].append(row)
    suites = {}
    for suite in sorted({i.suite for i in items}):
        selected = [i for i in items if i.suite == suite]
        suite_jobs = [j for j in jobs if j["suite"] == suite]
        scores = [r for i in selected for r in grouped[i.key]]
        contracts = [r for r in scores if r["mandatory"]]
        suites[suite] = {"items": len(selected), "expected_metrics": len(suite_jobs),
                         "completed_metrics": sum(r["status"] == "completed" for r in scores),
                         "error_metrics": sum(r["status"] != "completed" for r in scores),
                         "contract_failed": sum(r["status"] == "completed" and not r["passed"] for r in contracts),
                         "diagnostic_below_threshold": sum(r["status"] == "completed" and not r["passed"] and not r["mandatory"] for r in scores)}
    retrieval = {}
    for name in ("Recall@5", "MRR", "EvidenceRecall@5"):
        for variant in ("direct", "rewrite"):
            values = [r["score"] for r in final.values() if r["metric"] == name and r["status"] == "completed" and f"rag/{variant}/" in r["item_key"]]
            if values:
                retrieval[f"{variant}/{name}"] = {"count": len(values), "mean": sum(values) / len(values), "aggregate_threshold": .85,
                                                 "aggregate_passed": sum(values) / len(values) >= .85}
    errors = [r for r in final.values() if r["status"] != "completed"]
    return {"framework": "deepeval", "framework_version": FRAMEWORK_VERSION,
            "expected_items": len(items), "expected_metrics": len(jobs), "recorded_attempts": len(rows),
            "completed_metrics": sum(r["status"] == "completed" for r in final.values()),
            "missing_metrics": len(jobs) - len(final), "error_metrics": len(errors), "provider_attempts": budget.used,
            "execution_complete": len(final) == len(jobs) and not errors,
            "contract_failed": sum(s["contract_failed"] for s in suites.values()), "suites": suites,
            "retrieval": retrieval, "calibration": calibration_summary(items, list(final.values())),
            "native_gate_policy": "Native semantic metrics are diagnostic until independently qualified; project citation, safety and state contracts remain mandatory.",
            "errors": [{k: r.get(k) for k in ("item_key", "metric", "error_type", "error_code", "error_http_status")} for r in errors]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", type=Path, default=PROFILE)
    parser.add_argument("--mode", choices=["rules", "all"], default="rules")
    parser.add_argument("--suite", nargs="+", help="Select suite names; omitted means all")
    parser.add_argument("--case-ids", nargs="+", help="Substring filters on item keys")
    parser.add_argument("--max-calls", type=int, default=1800)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="One further attempt for prior errors, never for low scores")
    parser.add_argument("--require-quality", action="store_true", help="Also reject failed project contracts")
    args = parser.parse_args(argv)
    if args.mode == "all" and not args.run_models:
        parser.error("Semantic metrics require --run-models")
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    valid_suites = {"rag", "routing", "tools", "business", "business-routing", "security", "conversation",
                    "conversation-scene", "calibration-rag", "calibration-security", "calibration-native"}
    if args.suite and set(args.suite) - valid_suites: parser.error("Unknown suite")
    items = load_items(args.sources, suites=args.suite)
    if args.suite:
        known = {i.suite for i in items}
        if set(args.suite) - known: parser.error("Unknown suite")
        items = [i for i in items if i.suite in args.suite]
    if args.case_ids:
        if any(not any(key in i.key for i in items) for key in args.case_ids): parser.error("Unknown case filter")
        items = [i for i in items if any(key in i.key for key in args.case_ids)]
    if not items: parser.error("No selected items")
    budget = CallBudget(args.output / "calls.jsonl", args.max_calls)
    models = {s: QwenEvaluationModel(s, budget=budget) for s in ("rag", "security")}
    planned = []
    for item in items:
        for spec in metric_specs(item, models, semantic=args.mode == "all"):
            job_id = fingerprint([item.key, item.source, spec.name, PROTOCOL_VERSION])
            planned.append((item, spec, {"job_id": job_id, "item_key": item.key, "suite": item.suite,
                                       "metric": spec.name, "mandatory": spec.mandatory}))
    planned.sort(key=lambda p: (not p[0].suite.startswith("calibration-"), p[0].key, p[1].name))
    if not planned: parser.error("No applicable metrics in selected mode")
    store = RunStore(args.output, {"framework": "deepeval", "version": FRAMEWORK_VERSION, "protocol": PROTOCOL_VERSION,
        "mode": args.mode, "source_profile_sha256": sha256(args.sources), "provenance": provenance(),
        "max_calls": args.max_calls, "workers": args.workers, "jobs": [p[2] for p in planned],
        "native_judge_mode": {"enable_thinking": False, "project_policy_thinking_unchanged": True},
        "native_semantic_threshold": .8, "native_gate_status": "diagnostic_pending_qualification"})
    source_copy = args.output / "source-app"
    if not source_copy.exists():
        source_copy.mkdir()
        for path in (ROOT / "app").glob("*.py"):
            shutil.copy2(path, source_copy / path.name)
    # Frozen native objects are saved separately, not regenerated by hand in a
    # second data source. No credentials or hidden reasoning is collected.
    if not (args.output / "test-cases.json").exists():
        store.report("test-cases", [{"key": i.key, "suite": i.suite, "source": i.source,
                                     "test_case": serializable(i.test_case)} for i in items])
    prior = latest_rows(store.rows())
    pending = [p for p in planned if p[2]["job_id"] not in prior or
               (args.retry_errors and prior[p[2]["job_id"]]["status"] != "completed" and prior[p[2]["job_id"]]["attempt"] < 2)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        remaining, running, halt = iter(pending), {}, False
        def submit_next():
            try: item, spec, job = next(remaining)
            except StopIteration: return False
            future = pool.submit(execute_metric, item, spec, attempt=prior.get(job["job_id"], {}).get("attempt", 0) + 1)
            running[future] = job
            return True
        for _ in range(args.workers):
            if not submit_next(): break
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                row = {**future.result(), "job_id": running.pop(future)["job_id"]}
                store.append(row)
                print(json.dumps({k: row.get(k) for k in ("item_key", "metric", "status", "score", "passed", "error_type")}, ensure_ascii=False), flush=True)
                halt |= row.get("error_http_status") in {401, 402, 403} or str(row.get("error_code", "")).lower() in {"arrearage", "credit_balance_exhausted", "insufficient_quota"}
            store.report("progress", summarize(items, [p[2] for p in planned], store.rows(), budget))
            halt |= budget.used >= budget.maximum
            if not halt:
                while len(running) < args.workers and submit_next(): pass
    report = summarize(items, [p[2] for p in planned], store.rows(), budget)
    store.report("summary", report)
    print(json.dumps({k: report[k] for k in ["expected_items", "expected_metrics", "completed_metrics", "missing_metrics", "error_metrics", "provider_attempts", "contract_failed"]}))
    return 0 if report["execution_complete"] and (not args.require_quality or (
        not report["contract_failed"] and all(v["aggregate_passed"] for k, v in report["retrieval"].items()
                                             if k.startswith("rewrite/")))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
