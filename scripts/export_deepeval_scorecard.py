"""Export metric definitions/results separately from calibration and API failures."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dataset_governance import read_jsonl, sha256
from app.deepeval_evaluation import latest_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output file")
    summary = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))
    if not summary["execution_complete"]:
        parser.error("Final scorecard requires all metric jobs to complete")
    attempts = read_jsonl(args.run / "records.jsonl")
    rows = list(latest_rows(attempts).values())
    assert len(rows) == summary["expected_metrics"] and all(r["status"] == "completed" for r in rows)
    grouped = defaultdict(list)
    for row in rows:
        if row["suite"].startswith("calibration-"):
            continue
        group = row["suite"]
        if group == "rag":
            group += "/" + row["item_key"].split("/")[1]
        grouped[(group, row["metric"])].append(row)
    metrics = []
    for (group, name), selected in sorted(grouped.items()):
        metrics.append({"group": group, "metric": name, "count": len(selected),
                        "passed": sum(r["passed"] for r in selected),
                        "mean": sum(r["score"] for r in selected) / len(selected),
                        "thresholds": sorted({r["threshold"] for r in selected}),
                        "kinds": sorted({r["kind"] for r in selected}),
                        "mandatory_count": sum(r["mandatory"] for r in selected),
                        "below_threshold": [{k: r.get(k) for k in ("item_key", "score", "reason", "input_hash")}
                                            for r in selected if not r["passed"]]})
    usage = Counter()
    responses = 0
    for row in attempts:
        for event in row["trace"]:
            if event["stage"] == "judge_provider" and event["event"] == "response":
                responses += 1
                for key, value in (event["data"].get("usage") or {}).items():
                    if isinstance(value, int):
                        usage[key] += value
    lineage_file = args.run / "lineage.json"
    inherited = json.loads(lineage_file.read_text(encoding="utf-8"))["inherited_provider_attempts"] if lineage_file.exists() else 0
    rewrite_retrieval_ok = all(v["aggregate_passed"] for k, v in summary["retrieval"].items() if k.startswith("rewrite/"))
    rag_joint = {}
    for variant in ("direct", "rewrite"):
        rag_rows = defaultdict(dict)
        for row in rows:
            if row["item_key"].startswith("rag/" + variant + "/"):
                rag_rows[row["item_key"]][row["metric"]] = row
        assert len(rag_rows) == 60
        passed = {key: record["RAGStructure"]["passed"] and record["ProjectRAGPolicy"]["passed"]
                  for key, record in rag_rows.items()}
        rag_joint[variant] = {"count": len(passed), "passed": sum(passed.values()),
                              "failed_items": [k for k, value in passed.items() if not value],
                              "definition": "RAGStructure AND ProjectRAGPolicy; retrieval aggregate reported separately"}
    output = {"framework": summary["framework"], "framework_version": summary["framework_version"],
              "generator_sha256": sha256(Path(__file__)),
              "records_sha256": sha256(args.run / "records.jsonl"),
              "summary_sha256": sha256(args.run / "summary.json"),
              "execution_complete": True,
              "project_contract_status": "passed" if not summary["contract_failed"] and rewrite_retrieval_ok else "failed",
              "semantic_native_policy": summary["native_gate_policy"],
              "expected_items": summary["expected_items"], "expected_metrics": len(rows),
              "provider_attempts_including_inherited": summary["provider_attempts"],
              "inherited_provider_attempts": inherited,
              "new_provider_attempts_after_resume": summary["provider_attempts"] - inherited,
              "provider_responses_with_metadata": responses,
              "observed_usage": dict(usage),
              "historical_errors_retained": [{k: r.get(k) for k in ("item_key", "metric", "attempt", "error_type", "error_code", "error_http_status")}
                                             for r in attempts if r["status"] != "completed"],
              "retrieval": summary["retrieval"], "rag_joint_contract": rag_joint,
              "calibration": summary["calibration"], "metrics": metrics,
              "interpretation": "Counts are metric measurements, not independent questions. Native semantic scores stay diagnostic; calibration negatives excluded from Agent metric tables. Failed/unknown calls need not provide usage; observed tokens are not an invoice."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("execution_complete", "project_contract_status", "expected_metrics", "provider_attempts_including_inherited")}))


if __name__ == "__main__":
    main()
