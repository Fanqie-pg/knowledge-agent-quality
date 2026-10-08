"""Run reproducible offline or frozen-v1 live gates and verify JUnit completeness."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET

from app.dataset_governance import ROOT, validate_dataset
from app.evaluation_records import provenance
from app.tracing import redact


def inspect_junit(path, *, returncode=0, expected_suites=()):
    path = Path(path)
    if not path.is_file():
        return {"status": "failed", "error": "missing_junit", "executed": 0}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {"status": "failed", "error": "invalid_junit", "executed": 0}
    cases = root.findall(".//testcase")
    counts = {"executed": len(cases), "failed": sum(c.find("failure") is not None for c in cases),
              "errors": sum(c.find("error") is not None for c in cases),
              "skipped": sum(c.find("skipped") is not None for c in cases)}
    counts["passed"] = counts["executed"] - counts["failed"] - counts["errors"] - counts["skipped"]
    missing = [suite for suite in expected_suites if not any(suite in c.get("classname", "") for c in cases)]
    return {"status": "passed" if cases and returncode == 0 and not missing and not any(counts[k] for k in ("failed", "errors", "skipped")) else "failed",
            "returncode": returncode, **counts, "missing_suites": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["offline", "live", "framework"])
    parser.add_argument("--output", type=Path, default=Path("reports/gate"))
    parser.add_argument("--max-calls", type=int, default=240)
    parser.add_argument("--sources", type=Path, help="DeepEval source profile")
    parser.add_argument("--run-models", action="store_true", help="Explicitly enable DeepEval semantic Judge calls")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "framework":
        if not args.run_models:
            parser.error("framework gate requires --run-models")
        command = [sys.executable, "-m", "app.deepeval_evaluation", "--mode", "all", "--run-models",
                   "--require-quality", "--output", str(output), "--max-calls", str(args.max_calls)]
        if args.sources: command += ["--sources", str(args.sources)]
        return subprocess.call(command, cwd=ROOT)
    try:
        data = validate_dataset()
    except (ValueError, KeyError, OSError) as exc:
        data = {"errors": [type(exc).__name__]}
    if data["errors"]:
        (output / "gate.json").write_text(json.dumps({"status": "failed", "error": "invalid_dataset", "errors": data["errors"]}, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "gate.md").write_text("门禁失败：数据校验未通过，本次未执行模型或测试。\n", encoding="utf-8")
        print("Dataset validation failed")
        return 1
    command = [sys.executable, "-m", "pytest", "tests", "-q", "--tb=short", "--strict-markers", "--junitxml=" + str(output / "junit.xml"),
               "--basetemp=" + str(output / ("tmp-" + uuid.uuid4().hex)), "--trace-output=" + str(output / "traces.jsonl"),
               "-o", "cache_dir=" + str(output / "pytest-cache"), "-o", "junit_logging=all"]
    if args.mode == "offline":
        command += ["-m", "not evaluation"]
        expected = ("test_mcp_server", "test_mcp_client", "test_dataset_governance", "test_business_security")
    else:
        command += ["-m", "quality_gate", "--run-evaluation", "--strict-quality-gate"]
        expected = ("test_retrieval_quality_gate_evaluation", "test_intent_router_evaluation", "test_rag_end_to_end_evaluation",
                    "test_security_quality_gate_evaluation", "test_mcp_unified_agent_evaluation")
    # Do not trust a previous successful JUnit file if collection fails this time.
    junit = output / "junit.xml"
    if junit.exists():
        junit.unlink()
    trace = output / "traces.jsonl"
    if trace.exists():
        trace.unlink()
    environment = os.environ.copy()
    if args.mode == "live":
        if args.max_calls <= 0:
            raise ValueError("max-calls必须大于0")
        environment.update(EVALUATION_MAX_CALLS=str(args.max_calls), EVALUATION_CALL_LEDGER=str(output / "calls.jsonl"))
    process = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", errors="replace", capture_output=True, env=environment)
    log = redact(process.stdout + process.stderr)
    (output / "pytest.log").write_text(log, encoding="utf-8")
    print(log)
    report = {**inspect_junit(junit, returncode=process.returncode, expected_suites=expected),
              "mode": args.mode, "time": datetime.now(timezone.utc).isoformat(), "command": command,
              "dataset_version": "baseline_v1" if args.mode == "live" else data["dataset_version"],
              "dataset_integrity_passed": not data["errors"], "provenance": provenance()}
    report["max_calls"] = args.max_calls if args.mode == "live" else None
    ledger = output / "calls.jsonl"
    report["provider_calls"] = len(ledger.read_text(encoding="utf-8").splitlines()) if ledger.exists() else 0
    (output / "gate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "gate.md").write_text(f"{args.mode} gate: {report['status']}\n\n执行 {report['executed']}；通过 {report.get('passed',0)}；失败 {report.get('failed',0)}；错误 {report.get('errors',0)}；跳过 {report.get('skipped',0)}。\n\n真实模型门禁冻结在 v1 数据与既有阈值；v2 仍为待人工复核的能力评测集。\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
