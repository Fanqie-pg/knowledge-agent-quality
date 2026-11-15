"""Carry frozen evidence across the single, verified retrieval-gate exit-code fix.

No scoring function, metric, prompt, data, or Judge setting may change. This is
deliberately narrower than a general cross-version cache.
"""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.dataset_governance import ROOT, read_jsonl, sha256
from app.deepeval_adapters import load_items
from app.deepeval_evaluation import serializable, summarize
from app.evaluation_records import RunStore, fingerprint, provenance


GATE_RETURN = '''return 0 if report["execution_complete"] and (not args.require_quality or (
    not report["contract_failed"] and all(v["aggregate_passed"] for k, v in report["retrieval"].items()
                                         if k.startswith("rewrite/")))) else 1'''


def validate_gate_only_change(old_source, new_source):
    old, new = ast.parse(old_source), ast.parse(new_source)
    old_main = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    new_main = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert isinstance(old_main.body[-1], ast.Return)
    assert ast.dump(new_main.body[-1]) == ast.dump(ast.parse(GATE_RETURN).body[0]), "Unexpected exit-code patch"
    new_main.body[-1] = deepcopy(old_main.body[-1])
    assert ast.dump(old) == ast.dump(new), "Code changed beyond the final CLI return expression"


def verify_compatibility(parent):
    run = json.loads((parent / "run.json").read_text(encoding="utf-8"))
    assert run["config_hash"] == fingerprint(run["config"])
    for name, digest in run["config"]["provenance"]["files"].items():
        if name.startswith("app/"):
            assert sha256(parent / "source-app" / Path(name).name) == digest
        if name == "app/deepeval_evaluation.py":
            validate_gate_only_change((parent / "source-app/deepeval_evaluation.py").read_text(encoding="utf-8"),
                                      (ROOT / name).read_text(encoding="utf-8"))
        else:
            assert sha256(ROOT / name) == digest, name
    items = load_items()
    frozen = json.loads((parent / "test-cases.json").read_text(encoding="utf-8"))
    assert frozen == [{"key": i.key, "suite": i.suite, "source": i.source,
                       "test_case": serializable(i.test_case)} for i in items]
    return run, items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new directory; never overwrite parent or existing runs")
    parent_run, items = verify_compatibility(args.parent)
    config = deepcopy(parent_run["config"])
    config["provenance"] = provenance()
    store = RunStore(args.output, config)
    (args.output / "source-app").mkdir()
    for path in (ROOT / "app").glob("*.py"):
        shutil.copy2(path, args.output / "source-app" / path.name)
    shutil.copy2(args.parent / "test-cases.json", args.output / "test-cases.json")
    shutil.copy2(args.parent / "calls.jsonl", args.output / "calls.jsonl")
    rows = read_jsonl(args.parent / "records.jsonl")
    for index, row in enumerate(rows):
        store.append({**row, "parent_run_id": parent_run["run_id"], "parent_record_index": index,
                      "parent_record_sha256": fingerprint(row),
                      "reuse_reason": "Only final CLI retrieval gate return changed; scoring AST and frozen inputs identical"})
    calls = len(read_jsonl(args.output / "calls.jsonl"))
    summary = summarize(items, config["jobs"], store.rows(), SimpleNamespace(used=calls))
    store.report("summary", summary)
    store.report("progress", summary)
    store.report("lineage", {"parent": str(args.parent.resolve().relative_to(ROOT)),
                            "parent_run_id": parent_run["run_id"],
                            "parent_records_sha256": sha256(args.parent / "records.jsonl"),
                            "parent_calls_sha256": sha256(args.parent / "calls.jsonl"),
                            "reused_records": len(rows), "inherited_provider_attempts": calls,
                            "new_provider_attempts_by_migration": 0,
                            "compatibility": "Only final main() return expression changed; all scoring AST, other app/data hashes and frozen inputs checked identical"})
    print(json.dumps({"reused_records": len(rows), "inherited_provider_attempts": calls,
                      "new_provider_attempts": 0}))


if __name__ == "__main__":
    main()
