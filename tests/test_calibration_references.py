from copy import deepcopy
import json

import pytest

from app.calibration import (
    calibration_inputs, calibration_report, export_reference_samples, load_references,
    validate_references,
)
from app.dataset_governance import ROOT, read_jsonl


def reference_data():
    return read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl"), load_references()


def test_neutral_references_cover_all_samples_without_changing_label_origin():
    samples, references = reference_data()
    validate_references(samples, references)
    assert {row["sample_id"] for row in references} == {row["sample_id"] for row in samples}
    assert sum(row["origin"] == "user_provided" for row in references) == 34
    assert sum(row["origin"] == "assistant_generated" for row in references) == 26
    assert all("review_status" not in sample and "human_labels" not in sample for sample in samples)


def test_reference_input_binding_rejects_changed_answers():
    samples, references = reference_data()
    samples[0]["answer"] += " changed"
    with pytest.raises(ValueError, match="输入已变化"):
        validate_references(samples, references)


@pytest.mark.parametrize("mutation", [
    {"origin": "unknown"}, {"labels": {}}, {"suite": "unknown"}, {"human_labels": {}},
])
def test_invalid_reference_schema_is_not_accepted(mutation):
    samples, references = reference_data()
    reference = {**references[0], **mutation}
    with pytest.raises(ValueError):
        validate_references(samples, [reference])


def test_missing_references_do_not_promote_suggested_scores():
    samples, _ = reference_data()
    report = calibration_report(samples, [], {}, references=[])
    assert report["reference_count"] == 0 and report["reference_missing"] == len(samples)
    assert report["status"] == "pending" and report["agreement"] is None


def test_mixed_reference_origins_do_not_grant_independent_qualification():
    samples, references = reference_data()
    predictions = [{"sample_id": row["sample_id"], "suite": row["suite"], "trial_id": 0,
                    "status": "completed", "labels": deepcopy(row["labels"])} for row in references]
    report = calibration_report(samples, predictions, {}, references=references)
    assert report["reference_missing"] == 0 and report["reference_count"] == 60
    assert report["reference_origin_counts"] == {"user_provided": 34, "assistant_generated": 26}
    assert report["status"] == "pending"


def test_reference_export_contains_scores_and_origins_without_process_metadata(tmp_path):
    samples, references = reference_data()
    before = deepcopy(calibration_inputs(samples))
    path = tmp_path / "samples.md"
    export_reference_samples(samples, path, references)
    text = path.read_text(encoding="utf-8")
    assert "user_provided" in text and "assistant_generated" in text
    assert "参考分" in text and "证据理由" in text
    assert all(word not in text for word in ("review_status", "human_labels", "reviewer", "人工", "复核"))
    assert calibration_inputs(samples) == before


def test_calibration_cli_loads_neutral_references_without_model_calls(tmp_path, monkeypatch):
    from app import calibration
    monkeypatch.setattr("sys.argv", ["calibration", "--output", str(tmp_path)])
    monkeypatch.setattr(calibration, "judge_versions", lambda: {"rag": "v", "security": "v"})
    monkeypatch.setattr(calibration, "run_judges", lambda *args: pytest.fail("Unexpected model execution"))
    assert calibration.main() == 1
    report = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert report["reference_count"] == 60 and report["reference_missing"] == 0
    assert report["scored"] == 0 and report["status"] == "pending"
    assert (tmp_path / "samples.md").exists()


def test_reference_coverage_cannot_turn_incomplete_predictions_into_certification(tmp_path, monkeypatch):
    from app import calibration
    monkeypatch.setattr("sys.argv", ["calibration", "--output", str(tmp_path)])
    monkeypatch.setattr(calibration, "judge_versions", lambda: {"rag": "v", "security": "v"})
    calibration.main()
    path = tmp_path / "calibration.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["status"] = "passed"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert not calibration.certified_judge(path, "v")
