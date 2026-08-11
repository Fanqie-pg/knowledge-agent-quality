from dataclasses import dataclass

from app.citation_validator import validate_citations
from app.rag_agent import INSUFFICIENT_KNOWLEDGE_RESPONSE
from app.retrieval_case_loader import RetrievalCase
from app.retriever import RetrievalResult


@dataclass
class RAGAnswerEvaluationResult:
    case_id: str
    answer: str
    refused: bool
    refusal_correct: bool
    citation_presence_correct: bool
    citations_grounded: bool
    deterministic_checks_passed: bool


def evaluate_rag_answer(
    case: RetrievalCase,
    answer:str,
    retrieval_results:list[RetrievalResult],
) -> RAGAnswerEvaluationResult:
    refused = INSUFFICIENT_KNOWLEDGE_RESPONSE in answer

    refusal_correct = refused == (not case.answerable)

    citation_validation = validate_citations(answer, retrieval_results)

    if case.answerable:
        citation_presence_correct = citation_validation.has_citations
    else:
        citation_presence_correct = not citation_validation.has_citations

    citations_grounded = citation_validation.all_citations_grounded

    deterministic_checks_passed = all(
        [
            refusal_correct,
            citation_presence_correct,
            citations_grounded,
        ]
    )

    return RAGAnswerEvaluationResult(
        case_id=case.case_id,
        answer=answer,
        refused=refused,
        refusal_correct=refusal_correct,
        citation_presence_correct=citation_presence_correct,
        citations_grounded=citations_grounded,
        deterministic_checks_passed=deterministic_checks_passed,
    )