import pytest

pytestmark = pytest.mark.unit

from langchain_core.documents import Document

from app.retrieval_case_loader import RetrievalCase
from app.retrieval_evaluator import evaluate_retrieval_case, evaluate_retrieval_cases
from app.retriever import RetrievalResult


class MappingFakeRetriever:

    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def retrieve(self, query, k=3):
        self.calls.append((query, k))
        return self.results_by_query[query][:k]


class FakeRetriever:

    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_k = None

    def retrieve(self, query, k=3):
        self.last_query = query
        self.last_k = k
        return self.results[:k]


def test_evaluate_retrieval_case_calculates_metrics():
    wrong_document = Document(
        id="KB-API-001-CHUNK-000",
        page_content="接口说明",
        metadata={"doc_id": "KB-API-001"},
    )
    relevant_document = Document(
        id="KB-FAQ-001-CHUNK-000",
        page_content="同一个账号连续输入错误密码5次后，系统会锁定账号。",
        metadata={"doc_id": "KB-FAQ-001"},
    )

    retriever = FakeRetriever(
        [
            RetrievalResult(
                document=wrong_document,
                score=0.7,
                rank=1,
            ),
            RetrievalResult(
                document=relevant_document,
                score=0.6,
                rank=2,
            ),
        ]
    )
    case = RetrievalCase(
        case_id="RET-TEST-001",
        question="账号被锁定怎么办？",
        relevant_doc_ids=["KB-FAQ-001"],
        reference_answer="连续输错五次后锁定30分钟。",
        answerable=True,
        category="single_doc",
        difficulty="easy",
        required_evidence=["连续输入错误密码 5 次"],
    )

    result = evaluate_retrieval_case(retriever, case, k=2)

    assert retriever.last_query == case.question
    assert retriever.last_k == 2
    assert result.retrieved_doc_ids == [
        "KB-API-001",
        "KB-FAQ-001",
    ]
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 0.5
    assert result.evidence_recall_at_k == 1.0


def test_evaluate_retrieval_cases_aggregates_answerable_cases():
    faq_document = Document(
        page_content="账号会在30分钟后自动解锁。",
        metadata={"doc_id": "KB-FAQ-001"},
    )
    api_document = Document(
        page_content="调用 POST /tickets 创建工单。",
        metadata={"doc_id": "KB-API-001"},
    )
    faq_result = RetrievalResult(
        document=faq_document,
        score=0.8,
        rank=1,
    )
    api_result = RetrievalResult(
        document=api_document,
        score=0.7,
        rank=1,
    )

    first_case = RetrievalCase(
        case_id="RET-001",
        question="账号如何解锁？",
        relevant_doc_ids=["KB-FAQ-001"],
        reference_answer="等待30分钟。",
        answerable=True,
        category="single_doc",
        difficulty="easy",
        required_evidence=["账号会在 30 分钟后自动解锁"],
    )

    second_case = RetrievalCase(
        case_id="RET-002",
        question="接口如何创建工单？",
        relevant_doc_ids=["KB-API-001"],
        reference_answer="调用POST接口。",
        answerable=True,
        category="single_doc",
        difficulty="easy",
        required_evidence=["POST /tickets"],
    )

    unanswerable_case = RetrievalCase(
        case_id="RET-003",
        question="食堂今天吃什么？",
        relevant_doc_ids=[],
        reference_answer=None,
        answerable=False,
        category="unanswerable",
        difficulty="easy",
        required_evidence=[]
    )

    retriever = MappingFakeRetriever(
        {
            first_case.question: [faq_result, api_result],
            second_case.question: [faq_result, api_result],
        }
    )

    summary = evaluate_retrieval_cases(
        retriever,
        [
            first_case,
            second_case,
            unanswerable_case,
        ],
        k=2,
    )

    assert summary.evaluated_case_count == 2
    assert summary.skipped_unanswerable_count == 1
    assert summary.mean_recall_at_k == 1.0
    assert summary.mrr == 0.75
    assert len(summary.case_results) == 2

    assert summary.mean_evidence_recall_at_k == 1.0