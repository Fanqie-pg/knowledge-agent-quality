"""Compare native Faithfulness aggregation using frozen provider JSON, with zero API calls.

This is a diagnostic counterfactual, not a new Judge evaluation or a replacement
for the parent report. Every completed Faithfulness result is included.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import deepeval_runtime
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from app.deepeval_evaluation import latest_rows, serializable
from app.deepeval_runtime import LLMTestCase, FRAMEWORK_VERSION
from app.evaluation_records import fingerprint
from app.dataset_governance import sha256, read_jsonl


class FrozenResponses(DeepEvalBaseLLM):
    """Serve only the original typed final JSON; no network-capable fallback."""
    def __init__(self, responses, model_name):
        self.responses = list(responses)
        self.model_name = model_name
        self.prompts = []
        super().__init__(model=model_name)

    def load_model(self):
        return None

    def get_model_name(self):
        return self.model_name

    def generate(self, prompt, schema=None):
        if not self.responses or schema is None:
            raise ValueError("No frozen response for this request")
        response = self.responses.pop(0)
        if schema.__name__ != response["schema"]:
            raise ValueError("Frozen response schema/order mismatch")
        self.prompts.append(fingerprint([prompt, schema.model_json_schema()]))
        text = response["content"].strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[8:-3].strip()
        return schema.model_validate_json(text)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)


def compare(row, case):
    if row["metric"] != "NativeFaithfulness" or row["status"] != "completed":
        raise ValueError("Only completed native Faithfulness is replayable")
    if fingerprint(case) != row["input_hash"]:
        raise ValueError("Frozen test input hash mismatch")
    responses = [e["data"] for e in row["trace"]
                 if e["stage"] == "deepeval_judge" and e["event"] == "raw_response"]
    if not responses:
        raise ValueError("Original structured provider responses are missing")
    scores, prompt_hashes = [], []
    for strict in (False, True):
        model = FrozenResponses(responses, row["evaluation_model"])
        metric = FaithfulnessMetric(model=model, threshold=row["threshold"],
                                    penalize_ambiguous_claims=strict,
                                    include_reason=False, async_mode=False, eval_mode="llm")
        metric.measure(LLMTestCase.model_validate(case), _show_indicator=False)
        if model.responses:
            raise ValueError("Original provider responses were not fully consumed")
        if any(serializable(getattr(metric, key)) != row["native_details"][key]
               for key in ("truths", "claims", "verdicts")):
            raise ValueError("Replayed extraction/verdicts differ from the parent")
        scores.append(metric.score)
        prompt_hashes.append(model.prompts)
    if scores[0] != row["score"] or prompt_hashes[0] != prompt_hashes[1]:
        raise ValueError("Original score or generated prompts could not be reproduced")
    return {"job_id": row["job_id"], "item_key": row["item_key"],
            "suite": row["suite"], "input_hash": row["input_hash"],
            "original_score": scores[0], "strict_aggregation_score": scores[1],
            "original_passed": row["passed"], "strict_aggregation_passed": scores[1] >= row["threshold"],
            "threshold": row["threshold"], "prompt_hashes": prompt_hashes[0],
            "original_verdicts_reused": True, "new_provider_attempts": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a new output file; original comparisons are immutable")
    run = json.loads((args.parent / "run.json").read_text(encoding="utf-8"))
    if run["config"]["version"] != FRAMEWORK_VERSION:
        parser.error("Installed DeepEval version differs from parent")
    cases = {c["key"]: c for c in json.loads((args.parent / "test-cases.json").read_text(encoding="utf-8"))}
    selected = [r for r in latest_rows(read_jsonl(args.parent / "records.jsonl")).values()
                if r["metric"] == "NativeFaithfulness" and r["status"] == "completed"]
    results = [compare(r, cases[r["item_key"]]["test_case"]) for r in selected]
    calibration = {}
    for suite in ("calibration-rag", "calibration-native"):
        for split in ("calibration", "validation"):
            rows = [r for r in results if r["suite"] == suite and cases[r["item_key"]]["source"]["split"] == split]
            if not rows:
                continue
            expected = lambda r: (cases[r["item_key"]]["source"]["labels"]["faithfulness"] >= 4
                                   if suite == "calibration-rag" else cases[r["item_key"]]["source"]["labels"]["passed"])
            calibration[suite + "/" + split] = {mode: {
                "scored": len(rows), "agrees": sum(r[mode + "_passed"] == expected(r) for r in rows),
                "false_pass": sum(r[mode + "_passed"] and not expected(r) for r in rows)}
                for mode in ("original", "strict_aggregation")}
    output = {"framework_version": FRAMEWORK_VERSION, "parent_run_id": run["run_id"],
              "parent_records_sha256": sha256(args.parent / "records.jsonl"),
              "parent_test_cases_sha256": sha256(args.parent / "test-cases.json"),
              "scope": "All completed Faithfulness jobs in the frozen parent; pending/error jobs excluded and not claimed complete.",
              "interpretation": "Counterfactual aggregation only. Original JSON judgments and prompts unchanged; not a new model evaluation or replacement gate.",
              "completed_faithfulness": len(results), "new_provider_attempts": 0,
              "calibration": calibration, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("completed_faithfulness", "new_provider_attempts", "calibration")}))


if __name__ == "__main__":
    main()
