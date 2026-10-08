import json
from pathlib import Path
import shutil

import pytest

from app.dataset_governance import ROOT, read_jsonl, validate_dataset
from app.dataset_governance import sha256
from app.retrieval_case_loader import RetrievalCase, load_retrieval_cases

pytestmark = pytest.mark.unit


def test_controlled_corpus_has_evidence_and_isolated_groups():
    report = validate_dataset()
    assert report["errors"] == []
    assert {k: v["count"] for k, v in report["coverage"].items()} == {"retrieval": 60, "routing": 36, "security": 30}
    assert "human_review_complete" not in report
    assert report["coverage"]["retrieval"]["semantic_groups"] < 60


def copy_corpus(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    return tmp_path


def test_changed_knowledge_invalidates_frozen_manifest(tmp_path):
    root = copy_corpus(tmp_path)
    path = root / "data/knowledge/KB-HR-001.md"
    path.write_text(path.read_text(encoding="utf-8") + "\n变更\n", encoding="utf-8")
    assert any("hash mismatch" in e for e in validate_dataset(root)["errors"])


def test_text_hash_is_identical_on_windows_and_linux(tmp_path):
    windows, linux = tmp_path / "windows.md", tmp_path / "linux.md"
    windows.write_bytes("中文\r\nsecond\r\n".encode("utf-8"))
    linux.write_bytes("中文\nsecond\n".encode("utf-8"))
    assert sha256(windows) == sha256(linux)


def test_semantic_paraphrases_cannot_cross_splits(tmp_path):
    root = copy_corpus(tmp_path)
    path = root / "data/evaluation/case_metadata.jsonl"
    rows = read_jsonl(path)
    next(r for r in rows if r["case_id"] == "RET-007")["split"] = "holdout"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert any("semantic group crosses splits" in e for e in validate_dataset(root)["errors"])


def test_schema_does_not_claim_independent_certification():
    assert any("certification is not declared" in e for e in validate_dataset(require_reviewed=True)["errors"])


def test_current_metadata_keeps_coverage_without_process_fields():
    rows = read_jsonl(ROOT / "data/evaluation/case_metadata.jsonl")
    assert all("review_status" not in row and "reviewer" not in row for row in rows)
    report = validate_dataset()
    assert report["errors"] == []
    assert all("review_status" not in suite["dimensions"] for suite in report["coverage"].values())


@pytest.mark.parametrize("override", [
    {"answerable": "false"}, {"required_evidence": []}, {"reference_answer": " "},
    {"relevant_doc_ids": ["KB-TEST", "KB-TEST"]}, {"difficulty": "unknown"},
    {"answerable": False}, {"question": " "},
])
def test_retrieval_case_rejects_inconsistent_or_untyped_labels(override):
    fields = {"case_id": "TEST", "question": "问题", "relevant_doc_ids": ["KB-TEST"], "reference_answer": "答案",
              "answerable": True, "category": "single_doc", "difficulty": "easy", "required_evidence": ["证据"]}
    with pytest.raises(ValueError):
        RetrievalCase(**(fields | override))


def test_duplicate_retrieval_ids_are_rejected(tmp_path):
    row = read_jsonl(ROOT / "data/evaluation/retrieval_cases.jsonl")[0]
    path = tmp_path / "cases.jsonl"
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="重复case_id"):
        load_retrieval_cases(path)
