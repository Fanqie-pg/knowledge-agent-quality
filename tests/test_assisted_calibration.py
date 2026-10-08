from copy import deepcopy

import pytest

from app.calibration import (
    calibration_inputs, calibration_report, export_review, validate_assistant_reviews, validate_samples,
)
from app.dataset_governance import ROOT, read_jsonl
from app.evaluation_records import fingerprint


def pending_sample():
    sample = deepcopy(read_jsonl(ROOT / "data/evaluation/calibration/samples.jsonl")[0])
    sample.update(split="validation", human_labels=None, review_status="pending_human_review")
    sample.pop("reviewer", None)
    return sample


def assistant_review(sample):
    return {"sample_id": sample["sample_id"], "suite": sample["suite"], "reviewer_kind": "assistant",
            "labels": sample["suggested_labels"].copy(),
            "reason": "Compared answer against supplied evidence.",
            "sample_inputs_hash": fingerprint(calibration_inputs([sample])[0])}


def test_assistant_completion_never_certifies_human_calibration():
    sample = pending_sample()
    review = assistant_review(sample)
    prediction = {"sample_id": sample["sample_id"], "trial_id": 0, "status": "completed", "labels": review["labels"]}
    report = calibration_report([sample], [prediction], {}, assistant_reviews=[review])
    assert report["review_completion_status"] == "completed_with_assistance"
    assert report["review_coverage"] == 1 and report["review_uncovered"] == 0
    assert report["human_review_completed"] == 0 and report["human_review_pending"] == 1
    assert report["agreement"] is None and report["status"] == "pending"
    assert report["validation_evaluated"] == 0


def test_reference_labels_do_not_require_personal_reviewer_metadata():
    sample = pending_sample()
    review = assistant_review(sample)
    validate_assistant_reviews([sample], [review])
    sample.update(human_labels=review["labels"].copy(), review_status="reviewed")
    validate_samples([sample])
    assert "reviewer" not in sample and "reviewer" not in review
    assert "reviewed_at" not in review


def test_assistant_review_does_not_follow_modified_answer():
    sample = pending_sample()
    review = assistant_review(sample)
    sample["answer"] = "A different answer."
    with pytest.raises(ValueError, match="输入已变化"):
        validate_assistant_reviews([sample], [review])


@pytest.mark.parametrize("mutation", [
    {"reviewer_kind": "human"}, {"human_labels": {}}, {"review_status": "reviewed"}, {"reason": " "},
])
def test_assistant_sidecar_cannot_impersonate_human_review(mutation):
    sample = pending_sample()
    review = {**assistant_review(sample), **mutation}
    with pytest.raises(ValueError):
        validate_assistant_reviews([sample], [review])


def test_unknown_or_duplicate_assistant_reviews_are_rejected():
    sample = pending_sample()
    review = assistant_review(sample)
    with pytest.raises(ValueError, match="未知或重复"):
        validate_assistant_reviews([sample], [review, review])
    with pytest.raises(ValueError, match="未知或重复"):
        validate_assistant_reviews([sample], [{**review, "sample_id": "unknown"}])


def test_human_export_preserves_both_annotation_sources(tmp_path):
    sample = pending_sample()
    review = assistant_review(sample)
    before = deepcopy(sample)
    sample.update(human_labels={**review["labels"], "correctness": 4}, review_status="reviewed", reviewer="human-reviewer")
    review.update(reviewer="test-assistant", reviewed_at="private-review-date")
    assert calibration_inputs([sample]) == calibration_inputs([before])
    output = tmp_path / "review.md"
    export_review([sample], output, [review])
    text = output.read_text(encoding="utf-8")
    assert '人工评分：' in text and '助手评分（非人工标签）：' in text
    assert 'human-reviewer' not in text and 'test-assistant' not in text
    assert 'private-review-date' not in text
    assert '"correctness": 4' in text and '"correctness": 5' in text
    assert sample["human_labels"]["correctness"] == 4
