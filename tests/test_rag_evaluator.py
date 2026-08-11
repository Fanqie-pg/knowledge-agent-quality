from langchain_core.documents import Document

from app.rag_evaluator import evaluate_rag_answer
from app.retrieval_case_loader import RetrievalCase
from app.retriever import RetrievalResult


def test_evaluate_rag_answer_accepts_answer_with_grounded_citation():
    document = Document(
        id="KB-FAQ-001-CHUNK-000",
        page_content="账号锁定30分钟。",
        metadata={
            "doc_id": "KB-FAQ-001",
            "chunk_id": "KB-FAQ-001-CHUNK-000",
        },
    )
    retrieval_result = RetrievalResult(
        document=document,
        score=0.8,
        rank=1,
    )
    case = RetrievalCase(
        case_id="RET-TEST-001",
        question="账号锁定多久？",
        relevant_doc_ids=["KB-FAQ-001"],
        reference_answer="锁定30分钟。",
        answerable=True,
        category="single_doc",
        difficulty="easy",
        required_evidence=["锁定 30 分钟"],
    )
    answer = (
        "账号会锁定30分钟。"
        "[来源：KB-FAQ-001/"
        "KB-FAQ-001-CHUNK-000]"
    )

    result = evaluate_rag_answer(
        case,
        answer,
        [retrieval_result],
    )

    assert result.refused is False
    assert result.refusal_correct is True
    assert result.citation_presence_correct is True
    assert result.citations_grounded is True
    assert result.deterministic_checks_passed is True


def test_evaluate_rag_answer_accepts_correct_refusal():
    case = RetrievalCase(
        case_id="RET-TEST-002",
        question="食堂今天吃什么？",
        relevant_doc_ids=[],
        reference_answer=None,
        answerable=False,
        category="unanswerable",
        difficulty="easy",
        required_evidence=[],
    )
    result = evaluate_rag_answer(
        case,
        "知识库中没有足够信息回答该问题。",
        [],
    )
    assert result.refused is True
    assert result.refusal_correct is True
    assert result.citation_presence_correct is True
    assert result.citations_grounded is True
    assert result.deterministic_checks_passed is True