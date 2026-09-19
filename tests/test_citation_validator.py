import pytest

pytestmark = pytest.mark.unit

from langchain_core.documents import Document

from app.citation_validator import Citation, extract_citations, validate_citations
from app.retriever import RetrievalResult


def test_extract_citations_returns_structured_citations():
    answer = (
        "P1工单应在15分钟内首次响应。"
        "[来源：KB-PRODUCT-001/KB-PRODUCT-001-CHUNK-000]\n"
        "处理期间至少每30分钟更新一次进展。"
        "[来源：KB-SOP-001/KB-SOP-001-CHUNK-001]"
    )

    citations = extract_citations(answer)

    assert citations == [
        Citation(
            doc_id="KB-PRODUCT-001",
            chunk_id="KB-PRODUCT-001-CHUNK-000",
        ),
        Citation(
            doc_id="KB-SOP-001",
            chunk_id="KB-SOP-001-CHUNK-001",
        ),
    ]


def test_extract_citations_returns_empty_list_when_missing():
    citations = extract_citations(
        "知识库中没有足够信息回答该问题。"
    )

    assert citations == []


def build_retrieval_result():
    document = Document(
        id="KB-PRODUCT-001-CHUNK-000",
        page_content="P1工单应在15分钟内首次响应。",
        metadata={
            "doc_id": "KB-PRODUCT-001",
            "chunk_id": "KB-PRODUCT-001-CHUNK-000",
        },
    )

    return RetrievalResult(
        document=document,
        score=0.8,
        rank=1,
    )


def test_validate_citations_accepts_retrieved_source():
    retrieval_result = build_retrieval_result()
    answer = (
        "P1工单应在15分钟内首次响应。"
        "[来源：KB-PRODUCT-001/"
        "KB-PRODUCT-001-CHUNK-000]"
    )

    result = validate_citations(
        answer,
        [retrieval_result],
    )

    assert result.has_citations is True
    assert result.all_citations_grounded is True
    assert result.invalid_citations == []


def test_validate_citations_rejects_unretrieved_source():
    retrieval_result = build_retrieval_result()
    answer = (
        "P1工单应在15分钟内首次响应。"
        "[来源：KB-FAKE-001/"
        "KB-FAKE-001-CHUNK-000]"
    )

    result = validate_citations(
        answer,
        [retrieval_result],
    )

    assert result.has_citations is True
    assert result.all_citations_grounded is False
    assert result.invalid_citations == [
        Citation(
            doc_id="KB-FAKE-001",
            chunk_id="KB-FAKE-001-CHUNK-000",
        )
    ]