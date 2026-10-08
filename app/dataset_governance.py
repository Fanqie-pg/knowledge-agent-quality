"""Offline corpus integrity, evidence grounding, split and coverage checks."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("retrieval", "routing", "security")


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def sha256(path):
    # Git checks text out with different newlines on Windows/Linux.
    # Hash canonical UTF-8/LF content rather than platform-specific bytes.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def normalize(text):
    return "".join(text.casefold().split())


def validate_dataset(root=ROOT, *, require_reviewed=False):
    from app.retrieval_case_loader import load_retrieval_cases
    from app.routing_case_loader import load_routing_cases
    from app.security_case_loader import load_security_cases
    root = Path(root).resolve()
    folder = root / "data/evaluation"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    loaders = {"retrieval": load_retrieval_cases, "routing": load_routing_cases, "security": load_security_cases}
    errors, coverage = [], {}
    documents = {p.stem: normalize(p.read_text(encoding="utf-8")) for p in (root / "data/knowledge").glob("*.md")}
    metadata = read_jsonl(folder / "case_metadata.jsonl")
    lookup = {}
    for item in metadata:
        key = (item.get("suite"), item.get("case_id"))
        if key in lookup:
            errors.append(f"duplicate metadata: {key}")
        lookup[key] = item
    expected_keys = set()
    for suite in SUITES:
        path = folder / f"{suite}_cases.jsonl"
        cases = loaders[suite](path)
        rows = read_jsonl(path)
        if len(cases) != manifest["expected_counts"][suite]:
            errors.append(f"{suite}: count differs from manifest")
        groups, questions = {}, set()
        dimensions = defaultdict(Counter)
        for case, row in zip(cases, rows):
            key = (suite, case.case_id)
            expected_keys.add(key)
            meta = lookup.get(key)
            if meta is None:
                errors.append(f"missing metadata: {key}")
                continue
            for field in ("group_id", "scenario", "risk", "source_type"):
                if not isinstance(meta.get(field), str) or not meta[field].strip():
                    errors.append(f"{key}: invalid {field}")
            if meta.get("risk") not in {"low", "medium", "high"}:
                errors.append(f"{key}: invalid risk")
            if meta.get("source_type") not in {"legacy_imported", "assistant_authored", "human_authored", "model_generated"}:
                errors.append(f"{key}: invalid source_type")
            refs = meta.get("source_refs")
            if not isinstance(refs, list) or any(not isinstance(r, str) or not r.strip() for r in refs):
                errors.append(f"{key}: invalid source_refs")
            split, group = meta.get("split"), meta.get("group_id")
            if split not in {"dev", "holdout"}:
                errors.append(f"{key}: invalid split")
            if group in groups and groups[group] != split:
                errors.append(f"{suite}: semantic group crosses splits: {group}")
            groups[group] = split
            question = normalize(row.get("question", row.get("query", "")))
            if question in questions:
                errors.append(f"{suite}: duplicate question: {case.case_id}")
            questions.add(question)
            for field in ("split", "scenario", "risk", "source_type"):
                dimensions[field][str(meta.get(field))] += 1
            for field in ("category", "difficulty", "answerable", "expected_route", "attack_type", "target"):
                if field in row:
                    dimensions[field][str(row[field])] += 1
            if suite == "retrieval":
                if set(meta.get("source_refs", [])) != set(case.relevant_doc_ids):
                    errors.append(f"{key}: source_refs mismatch")
                source_texts = [documents.get(doc_id, "") for doc_id in case.relevant_doc_ids]
                for doc_id in case.relevant_doc_ids:
                    dimensions["documents"][doc_id] += 1
                    if doc_id not in documents:
                        errors.append(f"{key}: unknown document {doc_id}")
                for evidence in case.required_evidence:
                    if not any(normalize(evidence) in text for text in source_texts):
                        errors.append(f"{key}: evidence absent from named sources: {evidence}")
                if not case.answerable and not meta.get("unanswerable_reason"):
                    errors.append(f"{key}: unanswerable reason required")
        if dict(dimensions["split"]) != manifest["expected_splits"][suite]:
            errors.append(f"{suite}: split counts differ from manifest")
        coverage[suite] = {"count": len(cases), "semantic_groups": len(groups),
                           "dimensions": {k: dict(v) for k, v in dimensions.items()}}
    if set(lookup) != expected_keys:
        errors.append("metadata contains missing or unexpected case IDs")
    for relative, expected in manifest["sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            errors.append(f"file hash mismatch: {relative}")
    if require_reviewed:
        errors.append("independent certification is not declared by this dataset schema")
    return {"dataset_version": manifest["dataset_version"], "status": "failed" if errors else "valid",
            "coverage": coverage, "errors": errors}


def write_coverage(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"数据集 {report['dataset_version']}：{report['status']}",
             "", "| 数据集 | 样本 | 语义组 | dev | holdout |", "|---|---:|---:|---:|---:|"]
    for name, suite in report["coverage"].items():
        split = suite["dimensions"]["split"]
        lines.append(f"| {name} | {suite['count']} | {suite['semantic_groups']} | {split.get('dev', 0)} | {split.get('holdout', 0)} |")
        lines.extend(["", f"{name} 覆盖明细：", "", "```json", json.dumps(suite["dimensions"], ensure_ascii=False, indent=2), "```", ""])
    lines += ["校验错误：", *report["errors"]] if report["errors"] else ["校验错误：无。数据有效性不等同于外部认证。"]
    (directory / "coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/dataset"))
    parser.add_argument("--require-reviewed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = validate_dataset(require_reviewed=args.require_reviewed)
    write_coverage(report, args.output)
    print(json.dumps({"status": report["status"], "errors": report["errors"], "counts": {k: v["count"] for k, v in report["coverage"].items()}}, ensure_ascii=False))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
