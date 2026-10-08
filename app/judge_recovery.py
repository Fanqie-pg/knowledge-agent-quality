"""Resume after account recovery in a new run, reusing verified successes only."""
import argparse
import json
from pathlib import Path

from app.dataset_governance import ROOT, sha256, read_jsonl
from app.evaluation_records import RunStore, fingerprint
from app.judge_rescore import source_store, open_rescore, run, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["calibration", "regrade", "report"], required=True)
    parser.add_argument("--reuse", type=Path, default=ROOT / "reports/qwen-dual-judge-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/qwen-dual-judge-recovery-v1")
    parser.add_argument("--max-calls", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.output.resolve() == args.reuse.resolve():
        raise ValueError("Recovery requires a different directory and 1..4 workers")
    source = source_store(ROOT / "reports/full-evaluation-sol-v1")
    original, tasks = open_rescore(args.reuse, source)
    old_rows = original.rows()
    config = {**original.manifest["config"], "kind": "qwen_dual_judge_recovery",
              "reuse_source_run_id": original.manifest["run_id"], "reuse_source_rows_sha256": sha256(original.path),
              "source_provider_attempts": len(read_jsonl(original.directory / "calls.jsonl")),
              "source_error_records": sum(r["status"] != "completed" for r in old_rows),
              "recovery_implementation": sha256(Path(__file__))}
    store = RunStore(args.output, config)
    seen = {r["score_id"] for r in store.rows()}
    for row in old_rows:
        if row["status"] != "completed" or row["score_id"] in seen:
            continue
        if row["input_hash"] != config["task_inputs"][row["score_id"]]:
            raise ValueError("Saved grading input changed")
        suite = "rag" if row["suite"].startswith("rag") else "security"
        if row["judge_version"] != config["judge_versions"][suite]:
            raise ValueError("Saved grading model or rubric changed")
        store.append({**row, "reused": True, "origin_score_run_id": row["run_id"], "origin_score_hash": fingerprint(row)})
        seen.add(row["score_id"])
    if args.phase != "report":
        run(store, tasks, args.phase, args.workers, args.max_calls)
    summary = report(store, tasks)
    summary.update(reused_scores=sum(r.get("reused", False) for r in store.rows()),
                   new_scores=sum(not r.get("reused", False) for r in store.rows()),
                   source_provider_attempts=config["source_provider_attempts"],
                   source_error_records=config["source_error_records"])
    store.report("summary", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
