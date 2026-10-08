import pytest

pytestmark = pytest.mark.unit


from app.retrieval_metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank, evidence_recall_at_k


def test_recall_at_k_returns_one():
    score = recall_at_k(
        retrieved_doc_ids=[
            "KB-FAQ-001",
            "KB-API-001",
        ],
        relevant_doc_ids=["KB-FAQ-001"],
        k=1,
    )

    assert score == 1.0

def test_recall_at_k_returns_partial_score():
    score = recall_at_k(
        retrieved_doc_ids=[
            "KB-PRODUCT-001",
            "KB-FAQ-001",
        ],
        relevant_doc_ids=[
            "KB-PRODUCT-001",
            "KB-SOP-001",
        ],
        k=2,
    )

    assert score == 0.5

def test_recall_at_k_returns_zero():
    score = recall_at_k(
        retrieved_doc_ids=[
            "KB-FAQ-001",
            "KB-API-001",
            "KB-HR-001",
            "KB-SOP-001",
        ],
        relevant_doc_ids=["KB-SOP-001"],
        k=3,
    )
    assert score == 0.0


def test_recall_at_k_does_not_count_duplicate_document_twice():
    score = recall_at_k(
        retrieved_doc_ids=[
            "KB-FAQ-001",
            "KB-FAQ-001",
            "KB-API-001",
        ],
        relevant_doc_ids=["KB-FAQ-001"],
        k=3,
    )

    assert score == 1.0


@pytest.mark.parametrize(
    (
        "retrieved_doc_ids",
        "relevant_doc_ids",
        "expected_score",
    ),
    [
        (
            ["KB-FAQ-001", "KB-API-001"],
            ["KB-FAQ-001"],
            1.0
        ),
        (
            ["KB-API-001", "KB-FAQ-001"],
            ["KB-FAQ-001"],
            0.5,
        ),
        (
            ["KB-API-001", "KB-HR-001", "KB-FAQ-001"],
            ["KB-FAQ-001"],
            pytest.approx(1 / 3),
        ),
        (
            ["KB-API-001", "KB-HR-001"],
            ["KB-FAQ-001"],
            0.0,
        ),
    ]
)
def test_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids, expected_score):
    assert reciprocal_rank(retrieved_doc_ids, relevant_doc_ids) == expected_score


def test_mean_reciprocal_rank_returns_average_score():
    score = mean_reciprocal_rank(
        retrieved_doc_ids_by_case=[
            ["KB-FAQ-001", "KB-API-001"],
            ["KB-API-001", "KB-HR-001"],
            ["KB-HR-001", "KB-API-001"],
        ],
        relevant_doc_ids_by_case=[
            ["KB-FAQ-001"],
            ["KB-HR-001"],
            ["KB-SOP-001"],
        ],
    )

    assert score == pytest.approx(0.5)


def test_mean_reciprocal_rank_rejects_mismatched_case_count():
    with pytest.raises(ValueError, match="检索结果与相关文档数量不一致"):
        mean_reciprocal_rank(
            retrieved_doc_ids_by_case=[
                ["KB-FAQ-001"],
                ["KB-HR-001"],
            ],
            relevant_doc_ids_by_case=[
                ["KB-FAQ-001"],
            ],
        )


def test_evidence_recall_at_k_detects_missing_fact():
    score = evidence_recall_at_k(
        retrieved_contents=[
            "P1 工单应在 15 分钟内首次响应。",
            "P1 工单的首次响应时间要求仍为 15 分钟以内。",
            "首次响应表示已经开始跟进。",
        ],
        required_evidence=[
            "15 分钟内首次响应",
            "至少每 30 分钟更新一次处理进度",
        ],
        k=3,
    )

    assert score == 0.5