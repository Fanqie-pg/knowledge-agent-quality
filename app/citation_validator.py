import re
from dataclasses import dataclass
from app.retriever import RetrievalResult


CITATION_PATTERN = re.compile(
    r"\[来源：([^/\]]+)/([^\]]+)\]"
)


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str


def extract_citations(answer: str) -> list[Citation]:
    matches = CITATION_PATTERN.findall(answer)

    return [
        Citation(doc_id=doc_id, chunk_id=chunk_id) for doc_id, chunk_id in matches
    ]

@dataclass
class CitationValidationResult:
    citations: list[Citation]
    invalid_citations: list[Citation]
    has_citations: bool
    all_citations_grounded: bool


def validate_citations(
    answer: str,
    retrieval_results: list[RetrievalResult],
) -> CitationValidationResult:
    citations = extract_citations(answer)

    allowed_sources = {
        (
            result.document.metadata["doc_id"],
            result.document.metadata["chunk_id"],
        )
        for result in retrieval_results
    }

    invalid_citations = [
        citation
        for citation in citations
        if (
            citation.doc_id,
            citation.chunk_id,
        ) not in allowed_sources
    ]

    return CitationValidationResult(
        citations=citations,
        invalid_citations=invalid_citations,
        has_citations=bool(citations),
        all_citations_grounded=not invalid_citations,
    )