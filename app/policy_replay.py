"""Replay a new scoring policy on frozen answers without claiming new generation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from langchain_core.documents import Document

from app.calibration import judge_versions
from app.dataset_governance import ROOT, read_jsonl, sha256
from app.evaluation_records import RunStore, CallBudget, append_jsonl, attach_budget, fingerprint
from app.rag_agent import INSUFFICIENT_KNOWLEDGE_RESPONSE
from app.rag_evaluator import RAG_RULE_VERSION, evaluate_rag_answer
from app.retrieval_case_loader import load_retrieval_cases
from app.retriever import RetrievalResult


def replay_rules(store, source_rows, cases):
    seen = {row["record_id"] for row in store.rows()}
    for original in source_rows:
        case = cases[original["case_id"]]
        if (original["question"] != case.question or original["answerable"] != case.answerable
                or original.get("reference_answer") != case.reference_answer):
            raise ValueError("保存的回答与当前用例定义不同，不能混合重评")
        if original["record_id"] in seen:
            continue
        if original["status"] != "completed":
            raise ValueError("来源有生成错误，必须另行保留，不能冒充完整回答重评")
        retrieved = [RetrievalResult(Document(page_content=item["content"], metadata={
            "doc_id": item["doc_id"], "chunk_id": item["chunk_id"]}), item["score"], item["rank"])
            for item in original["retrieval"]]
        rule = evaluate_rag_answer(case, original["answer"], retrieved)
        store.append({**{key: original[key] for key in ("record_id", "case_id", "variant", "trial_id", "question", "answerable",
                                                        "reference_answer", "answer", "context")},
                      "source_run_id": original["run_id"], "source_origin_run_id": original.get("origin_run_id"),
                      "original_rules": original["rules"], "original_deterministic_passed": original["deterministic_passed"],
                      "rules": asdict(rule), "deterministic_passed": rule.deterministic_checks_passed,
                      "generation_reused": True, "new_generation": False})


def score_refusals(store, judge):
    path = store.directory / "refusal_scores.jsonl"
    version = store.manifest["config"]["judge_version"]
    prior = read_jsonl(path) if path.exists() else []
    if any(row["judge_version"] != version for row in prior):
        raise ValueError("评分协议已变化，请使用新的目录")
    seen = {row["record_id"] for row in prior}
    for row in store.rows():
        if row["answerable"] or row["record_id"] in seen:
            continue
        result = {"record_id": row["record_id"], "case_id": row["case_id"], "variant": row["variant"],
                  "trial_id": row["trial_id"], "judge_version": version, "status": "error", "passed": False}
        try:
            grade = judge.evaluate(row["question"], INSUFFICIENT_KNOWLEDGE_RESPONSE, row["answer"], row["context"])
            result.update(status="completed", passed=grade.passed, scores=grade.model_dump())
        except Exception as exc:
            result["error_type"] = type(exc).__name__
        append_jsonl(path, result)
        if result.get("error_type") == "CallBudgetExceeded":
            break


def summarize(store):
    rows = store.rows()
    path = store.directory / "refusal_scores.jsonl"
    scores = read_jsonl(path) if path.exists() else []
    expected_ids = {row["record_id"] for row in rows if not row["answerable"]}
    scored_ids = [row["record_id"] for row in scores]
    if len(scored_ids) != len(set(scored_ids)) or not set(scored_ids) <= expected_ids:
        raise ValueError("重评包含重复或非计划评分")
    variants = {}
    for variant in sorted({row["variant"] for row in rows}):
        subset = [row for row in rows if row["variant"] == variant]
        variants[variant] = {"records": len(subset),
                             "old_structural_pass_rate": sum(row["original_deterministic_passed"] for row in subset) / len(subset),
                             "new_structural_pass_rate": sum(row["deterministic_passed"] for row in subset) / len(subset)}
    completed = [row for row in scores if row["status"] == "completed"]
    semantic_complete = set(scored_ids) == expected_ids and len(completed) == len(scores)
    report = {"analysis_kind": "frozen_answer_policy_replay", "rule_version": RAG_RULE_VERSION,
              "source_records": len(rows), "new_generation_calls": 0, "variants": variants,
              "structural_changed_records": sum(row["deterministic_passed"] != row["original_deterministic_passed"] for row in rows),
              "structural_failed_records": sum(not row["deterministic_passed"] for row in rows),
              "semantic_scope": "only_the_saved_unanswerable_records",
              "semantic_expected": len(expected_ids), "semantic_scored": len(scores),
              "semantic_missing": len(expected_ids - set(scored_ids)), "semantic_errors": len(scores) - len(completed),
              "semantic_passed": sum(row["passed"] for row in completed),
              "semantic_complete": semantic_complete,
              "semantic_failures": [row for row in scores if not row["passed"]],
              "provider_calls": len(read_jsonl(store.directory / "calls.jsonl")) if (store.directory / "calls.jsonl").exists() else 0,
              "judge_status": "provisional_not_human_calibrated", "run_config": store.manifest["config"],
              "limitations": "结构规则通过不证明背景事实有据；语义评分仅覆盖保存的不可回答记录。回答由旧版提示词生成，规则变更收益不能解释为新模型输出质量提升。"}
    store.report("summary", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/ab-dev-v2.0.1")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/policy-v2-replay")
    parser.add_argument("--score-refusals", action="store_true")
    parser.add_argument("--max-calls", type=int, default=90)
    args = parser.parse_args()
    source = json.loads((args.source / "run.json").read_text(encoding="utf-8"))
    source_rows = read_jsonl(args.source / "records.jsonl")
    definitions = {case.case_id: case for case in load_retrieval_cases(ROOT / "data/evaluation/retrieval_cases.jsonl")}
    for case_id, value in source["config"]["case_fingerprints"].items():
        if case_id not in definitions or fingerprint(asdict(definitions[case_id])) != value:
            raise ValueError("来源用例已改变，不能重用旧回答")
    for name, value in source["config"]["provenance"]["files"].items():
        if name.startswith("data/knowledge/") and sha256(ROOT / name) != value:
            raise ValueError("知识事实已变化，不能沿用旧证据重评")
    expected = {(case, variant, trial) for case in source["config"]["case_ids"]
                for variant in source["config"]["variants"] for trial in range(source["config"]["trials"])}
    if len(source_rows) != len(expected) or {(r["case_id"], r["variant"], r["trial_id"]) for r in source_rows} != expected:
        raise ValueError("来源试次不完整或重复")
    store = RunStore(args.output, {"kind": "frozen_answer_policy_replay", "source_run_id": source["run_id"],
                                  "source_records_sha256": sha256(args.source / "records.jsonl"),
                                  "rule_version": RAG_RULE_VERSION, "judge_version": judge_versions()["rag"],
                                  "rule_implementation_sha256": sha256(Path(__file__).with_name("rag_evaluator.py")),
                                  "citation_implementation_sha256": sha256(Path(__file__).with_name("citation_validator.py")),
                                  "implementation_sha256": sha256(Path(__file__))})
    replay_rules(store, source_rows, definitions)
    if args.score_refusals:
        from app.judge_model import build_judge_model
        from app.llm_judge import LLMJudge
        model = attach_budget(build_judge_model(), CallBudget(args.output / "calls.jsonl", args.max_calls))
        score_refusals(store, LLMJudge(model))
    report = summarize(store)
    print(json.dumps({key: report[key] for key in ("source_records", "structural_changed_records", "structural_failed_records",
                                                  "semantic_scored", "semantic_expected", "semantic_passed", "semantic_errors")}, ensure_ascii=False))
    return 0 if not args.score_refusals or report["semantic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
