"""Paired query-rewrite experiment: collect once, score saved answers separately."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import random
import hashlib
from pathlib import Path

from app.dataset_governance import ROOT, read_jsonl, validate_dataset
from app.evaluation_records import RunStore, CallBudget, CallBudgetExceeded, attach_budget, fingerprint, provenance
from app.tracing import capture_trace

VARIANTS = ("direct", "rewrite")
ANALYSIS_VERSION = fingerprint({"source": Path(__file__).read_text(encoding="utf-8")})


def load_cases(split):
    from app.retrieval_case_loader import load_retrieval_cases
    metadata = {r["case_id"]: r for r in read_jsonl(ROOT / "data/evaluation/case_metadata.jsonl") if r["suite"] == "retrieval"}
    return [(c, metadata[c.case_id]) for c in load_retrieval_cases(ROOT / "data/evaluation/retrieval_cases.jsonl")
            if metadata[c.case_id]["split"] == split]


def experiment_config(split="dev", trials=3):
    if trials <= 0:
        raise ValueError("trials必须大于0")
    cases = load_cases(split)
    if not cases:
        raise ValueError("实验用例不能为空")
    return {"kind": "rag_rewrite_ab", "split": split, "trials": trials,
            "case_ids": [c.case_id for c, _ in cases], "variants": list(VARIANTS), "k": 5,
            "answerable_case_ids": [c.case_id for c, _ in cases if c.answerable],
            "case_fingerprints": {c.case_id: fingerprint(asdict(c)) for c, _ in cases},
            "model": "deepseek-v4-flash", "temperature": 0, "timeout": 30,
            "max_attempts": 2, "chunk_size": 500, "chunk_overlap": 50,
            "embedding": "BAAI/bge-small-zh-v1.5", "embedding_device": "cpu",
            "current_default": "rewrite", "provenance": provenance()}


def reuse_unchanged(store, source_directory, source_cases_path, cases):
    """Explicit input-verified reuse when correcting a draft corpus; preserve lineage."""
    source_directory, source_cases_path = Path(source_directory), Path(source_cases_path)
    source_manifest = json.loads((source_directory / "run.json").read_text(encoding="utf-8"))
    previous, current = source_manifest["config"], store.manifest["config"]
    for key in ("model", "temperature", "timeout", "max_attempts", "chunk_size", "chunk_overlap", "embedding", "embedding_device", "k", "trials", "variants"):
        if previous[key] != current[key]:
            raise ValueError(f"复用条件不满足：{key}")
    from app.dataset_governance import sha256
    def matches(path, expected):
        return expected in {sha256(path), hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    sources = previous["provenance"]["files"]
    if not matches(source_cases_path, sources["data/evaluation/retrieval_cases.jsonl"]):
        raise ValueError("旧数据快照与原运行版本不一致")
    for name in ("rag_agent", "query_rewriter", "retriever", "document_loader", "document_splitter", "resilient_model", "agent_factory"):
        relative = f"app/{name}.py"
        if not matches(ROOT / relative, sources[relative]):
            raise ValueError(f"RAG执行代码变更，不能复用：{name}")
    for relative in sources:
        if relative.startswith("data/knowledge/") and not matches(ROOT / relative, sources[relative]):
            raise ValueError("知识库变更，不能复用")
    old_cases = {c["case_id"]: c for c in read_jsonl(source_cases_path)}
    allowed = {c.case_id for c, _ in cases if fingerprint(old_cases.get(c.case_id)) == fingerprint(asdict(c))}
    seen = {(r["case_id"], r["variant"], r["trial_id"]) for r in store.rows()}
    imported = set()
    for row in read_jsonl(source_directory / "records.jsonl"):
        key = (row["case_id"], row["variant"], row["trial_id"])
        if row["case_id"] not in allowed or row["status"] != "completed":
            continue
        imported.add(row["record_id"])
        if key not in seen:
            store.append({**row, "origin_run_id": row["run_id"], "origin_config_hash": source_manifest["config_hash"], "reused": True})
            seen.add(key)
    # Only reuse scores from the same Judge on verified identical saved answers.
    summary_path = source_directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("scored") == summary.get("expected"):
            reuse_saved_scores(store, source_directory)


def reuse_saved_scores(store, source_directory):
    """Reuse grades only when the actual stored question/context/answer are identical."""
    from app.calibration import judge_versions
    from app.evaluation_records import append_jsonl
    source_directory = Path(source_directory)
    source_rows = {r["record_id"]: r for r in read_jsonl(source_directory / "records.jsonl")}
    source_manifest = json.loads((source_directory / "run.json").read_text(encoding="utf-8"))
    scores_path, target = source_directory / "scores.jsonl", store.directory / "scores.jsonl"
    if not scores_path.exists():
        return
    target_rows = {r["record_id"]: r for r in store.rows()}
    version = judge_versions()["rag"]
    seen = {r["record_id"] for r in read_jsonl(target)} if target.exists() else set()
    fields = ("question", "reference_answer", "context", "answer", "answerable")
    for score in read_jsonl(scores_path):
        key = score["record_id"]
        old, new = source_rows.get(key), target_rows.get(key)
        if key in seen or not old or not new or new.get("origin_run_id") != source_manifest["run_id"]:
            continue
        if score["status"] == "completed" and score["judge_version"] == version and all(old.get(k) == new.get(k) for k in fields):
            append_jsonl(target, {**score, "origin_run_id": source_manifest["run_id"], "reused": True})
            seen.add(key)


def wilson(successes, count):
    if count == 0:
        return None
    z, p = 1.96, successes / count
    center = (p + z*z / (2*count)) / (1 + z*z/count)
    radius = z * ((p*(1-p)/count + z*z/(4*count*count)) ** .5) / (1 + z*z/count)
    return [max(0, center-radius), min(1, center+radius)]


def collect(store, cases, agents):
    from app.rag_evaluator import evaluate_rag_answer
    from app.retrieval_metrics import recall_at_k, reciprocal_rank, evidence_recall_at_k
    trials = store.manifest["config"]["trials"]
    keys = {(r["case_id"], r["variant"], r["trial_id"]) for r in store.rows()}
    for trial in range(trials):
        for index, (case, meta) in enumerate(cases):
            # Interleave A/B order without modifying model sampling parameters.
            order = VARIANTS if (trial + index) % 2 == 0 else VARIANTS[::-1]
            for variant in order:
                key = (case.case_id, variant, trial)
                if key in keys:
                    continue
                row = {"case_id": case.case_id, "group_id": meta["group_id"], "variant": variant, "trial_id": trial,
                       "question": case.question, "answerable": case.answerable, "reference_answer": case.reference_answer,
                       "status": "error", "deterministic_passed": False}
                stop = False
                with capture_trace() as events:
                    try:
                        result = agents[variant].run(case.question)
                        answer = result.final_response.content
                        rule = evaluate_rag_answer(case, answer, result.retrieval_results)
                        ids = [r.document.metadata["doc_id"] for r in result.retrieval_results]
                        contents = [r.document.page_content for r in result.retrieval_results]
                        metrics = None
                        if case.answerable:
                            metrics = {"recall_at_5": recall_at_k(ids, case.relevant_doc_ids, 5),
                                       "reciprocal_rank": reciprocal_rank(ids, case.relevant_doc_ids),
                                       "evidence_recall_at_5": evidence_recall_at_k(contents, case.required_evidence, 5)}
                        row.update(status="completed", answer=answer, context=result.context,
                                   retrieval_query=result.retrieval_query, metrics=metrics,
                                   deterministic_passed=rule.deterministic_checks_passed, rules=asdict(rule),
                                   retrieval=[{"doc_id": r.document.metadata["doc_id"], "chunk_id": r.document.metadata["chunk_id"],
                                               "score": r.score, "rank": r.rank, "content": r.document.page_content} for r in result.retrieval_results],
                                   usage=getattr(result.final_response, "usage_metadata", None))
                    except Exception as exc:
                        row["error_type"] = type(exc).__name__
                        stop = isinstance(exc, CallBudgetExceeded)
                row["trace"] = events
                row["record_id"] = fingerprint({k: row[k] for k in ("case_id", "variant", "trial_id")})
                store.append(row)
                keys.add(key)
                if stop:
                    return


def score_saved(store, judge, judge_version):
    """Scoring errors are persisted, not silently interpreted as answer failures/successes."""
    path = store.directory / "scores.jsonl"
    prior = read_jsonl(path) if path.exists() else []
    if any(r["judge_version"] != judge_version for r in prior):
        raise ValueError("Judge版本发生变化，请使用独立评分目录/归档旧评分后重新评分")
    seen = {r["record_id"] for r in prior}
    from app.evaluation_records import append_jsonl
    from app.rag_agent import INSUFFICIENT_KNOWLEDGE_RESPONSE
    for row in store.rows():
        if row["status"] != "completed" or row["record_id"] in seen:
            continue
        result = {"record_id": row["record_id"], "judge_version": judge_version, "status": "error", "passed": False}
        try:
            grade = judge.evaluate(row["question"], row["reference_answer"] or INSUFFICIENT_KNOWLEDGE_RESPONSE,
                                   row["answer"], row["context"])
            result.update(status="completed", passed=grade.passed, scores=grade.model_dump())
        except Exception as exc:
            result["error_type"] = type(exc).__name__
        append_jsonl(path, result)
        if result.get("error_type") == "CallBudgetExceeded":
            break


def summarize(store, calibration_path=None):
    config = store.manifest["config"]
    rows = store.rows()
    expected = {(c, v, t) for c in config["case_ids"] for v in VARIANTS for t in range(config["trials"])}
    keys = [(r["case_id"], r["variant"], r["trial_id"]) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("重复或非计划内运行记录")
    score_path = store.directory / "scores.jsonl"
    score_rows = read_jsonl(score_path) if score_path.exists() else []
    scores = {r["record_id"]: r for r in score_rows}
    if len(scores) != len(score_rows) or not set(scores) <= {r["record_id"] for r in rows if r["status"] == "completed"}:
        raise ValueError("重复或非计划Judge记录")
    calibrated = False
    if calibration_path and scores:
        from app.calibration import certified_judge
        versions = {s["judge_version"] for s in score_rows}
        calibrated = len(versions) == 1 and certified_judge(calibration_path, next(iter(versions)))
    variants = {}
    for variant in VARIANTS:
        subset = [r for r in rows if r["variant"] == variant]
        denominator = len(config["case_ids"]) * config["trials"]
        variants[variant] = {"expected": denominator, "executed": len(subset),
                             "completed": sum(r["status"] == "completed" for r in subset),
                             "errors": sum(r["status"] == "error" for r in subset),
                             "deterministic_pass_rate": sum(r["deterministic_passed"] for r in subset) / denominator,
                             "judge_pass_rate_provisional": sum(scores.get(r["record_id"], {}).get("passed", False) for r in subset) / denominator}
        stable = sum(all(any(r["case_id"] == case_id and r["trial_id"] == trial and r["status"] == "completed" and r["deterministic_passed"] for r in subset) for trial in range(config["trials"])) for case_id in config["case_ids"])
        variants[variant].update(case_all_trials_passed=stable, case_all_trials_denominator=len(config["case_ids"]),
                                 case_all_trials_wilson_95_interval=wilson(stable, len(config["case_ids"])))
        for metric in ("recall_at_5", "reciprocal_rank", "evidence_recall_at_5"):
            answerable = [r for r in subset if r["answerable"]]
            metric_denominator = len(config["answerable_case_ids"]) * config["trials"]
            variants[variant][metric] = sum((r.get("metrics") or {}).get(metric, 0) for r in answerable) / metric_denominator if metric_denominator else None
    cases = defaultdict(dict)
    for row in rows:
        cases[row["case_id"]].setdefault(row["variant"], []).append(row)
    details, groups = [], defaultdict(list)
    for case_id, by_variant in cases.items():
        if any(len(by_variant.get(v, [])) != config["trials"] for v in VARIANTS):
            continue
        means = {v: sum(r["deterministic_passed"] for r in by_variant[v]) / config["trials"] for v in VARIANTS}
        delta = means["rewrite"] - means["direct"]
        groups[by_variant["direct"][0]["group_id"]].append(delta)
        details.append({"case_id": case_id, "delta": delta, "classification": "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"})
    interval = None
    if groups:
        rng = random.Random(20261112)
        values = [sum(v) / len(v) for v in groups.values()]
        samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(2000))
        interval = [samples[49], samples[1949]]
    complete = set(keys) == expected and all(r["status"] == "completed" for r in rows)
    report = {"run_id": store.manifest["run_id"], "status": "complete" if complete else "incomplete",
              "expected": len(expected), "executed": len(rows), "missing": len(expected - set(keys)),
              "judge_status": "human_calibrated" if calibrated else "provisional_not_human_calibrated", "scored": len(scores),
              "judge_missing": len(expected) - len(scores),
              "scoring_status": "complete" if len(scores) == len(expected) and all(s["status"] == "completed" for s in scores.values()) else "incomplete",
              "reused_records": sum(r.get("reused", False) for r in rows),
              "new_records": sum(not r.get("reused", False) for r in rows),
              "provider_calls_this_directory": len(read_jsonl(store.directory / "calls.jsonl")) if (store.directory / "calls.jsonl").exists() else 0,
              "judge_errors": sum(r["status"] == "error" for r in scores.values()), "variants": variants,
              "paired_case_results": details, "semantic_group_count": len(groups),
              "deterministic_delta_group_bootstrap_95_interval": interval,
              "semantic_group_mean_delta": sum(sum(v)/len(v) for v in groups.values()) / len(groups) if groups else None,
              "analysis_version": ANALYSIS_VERSION,
              "interpretation": "质量结论需结合人工校准；区间跨零不宣称提升。检索错误、生成错误和Judge错误单列。"}
    store.report("summary", report)
    lines = [f"实验 {report['run_id']}：{report['status']}", f"执行 {len(rows)}/{len(expected)}；缺失 {report['missing']}。",
             "Judge状态：" + report["judge_status"], "", "| 组别 | 完成 | 错误 | 规则通过率 | Judge通过率（以校准状态为准） |", "|---|---:|---:|---:|---:|"]
    for name, v in variants.items():
        lines.append(f"| {name} | {v['completed']} | {v['errors']} | {v['deterministic_pass_rate']:.4f} | {v['judge_pass_rate_provisional']:.4f} |")
    lines += ["", f"按语义组配对重采样的差值区间：{interval}", "", "逐例变化：", "```json", json.dumps(details, ensure_ascii=False, indent=2), "```"]
    (store.directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ab-dev"))
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=600)
    parser.add_argument("--run-models", action="store_true")
    parser.add_argument("--score", action="store_true", help="Only grade saved answers; never regenerate them")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--reuse-cases", type=Path)
    parser.add_argument("--reuse-scores-from", type=Path)
    args = parser.parse_args()
    if args.report_only or args.score:
        manifest = json.loads((args.output / "run.json").read_text(encoding="utf-8"))
        store = RunStore(args.output, manifest["config"])
    else:
        checked = validate_dataset()
        if checked["errors"]:
            raise ValueError(checked["errors"])
        config = experiment_config(args.split, args.trials)
        estimate = len(config["case_ids"]) * args.trials * 3
        if not args.run_models:
            print(json.dumps({"mode": "dry_run", "cases": len(config["case_ids"]), "jobs": len(config["case_ids"]) * args.trials * 2,
                              "generation_calls_without_retries": estimate, "maximum_with_retries": estimate * 2,
                              "additional_judge_calls_without_retries": len(config["case_ids"]) * args.trials * 2}, ensure_ascii=False))
            return 0
        store = RunStore(args.output, config)
        if args.reuse_from:
            if args.reuse_cases is None:
                raise ValueError("复用旧运行需要--reuse-cases提供其原始数据快照")
            reuse_unchanged(store, args.reuse_from, args.reuse_cases, load_cases(args.split))
    if args.run_models and not args.report_only:
        from app.agent_factory import build_chat_model, build_knowledge_retriever, build_rag_agent
        # Validate credentials before loading embeddings or consuming any provider calls.
        if args.score:
            from app.llm_judge import LLMJudge
            from app.judge_model import build_judge_model
            from app.calibration import judge_versions
            model = attach_budget(build_judge_model(), CallBudget(args.output / "calls.jsonl", args.max_calls))
            score_saved(store, LLMJudge(model), judge_versions()["rag"])
        else:
            model = attach_budget(build_chat_model(), CallBudget(args.output / "calls.jsonl", args.max_calls))
            retriever = build_knowledge_retriever()
            agents = {v: build_rag_agent(model, retriever, rewrite_queries=v == "rewrite") for v in VARIANTS}
            collect(store, load_cases(args.split), agents)
    if args.reuse_scores_from:
        reuse_saved_scores(store, args.reuse_scores_from)
    report = summarize(store, args.calibration_report)
    print(json.dumps({k: report[k] for k in ("status", "expected", "executed", "missing", "scoring_status", "judge_status")}, ensure_ascii=False))
    successful = report["status"] == "complete" and (not args.score or report["scoring_status"] == "complete")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
