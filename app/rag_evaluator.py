from dataclasses import dataclass
import re

from app.citation_validator import validate_citations
from app.rag_agent import INSUFFICIENT_KNOWLEDGE_RESPONSE
from app.retrieval_case_loader import RetrievalCase
from app.retriever import RetrievalResult

RAG_RULE_VERSION = "2.1.3-evidence-absence-refusal"


def is_insufficient_knowledge_answer(answer: str) -> bool:
    """Recognize the explicit refusal protocol without requiring its last words.

    This is a constrained structural recognizer, not a semantic truth check. A
    sentence must begin with the refusal, so negation and quoted instructions
    do not count just because they contain the same words.
    """
    normalized = re.sub(r"[ \t\r*_`]+", "", answer)
    # Scope and source type compose: e.g. 本次 + 提供的 + 知识库 + 资料中.
    subject = r"(?:知识库中|在?(?:本次|当前|现有|所|已)?(?:提供|召回)?的?(?:知识库)?(?:资料|上下文)中)"
    explicit = bool(re.search(
        r"(?:^|[。！？\n；，])(?:因此|所以|故)?" + subject + r"没有足够信息回答",
        normalized,
    ))
    absence_and_refusal = bool(re.search(
        r"(?:^|[。！？\n；])" + subject + r"没有[^。！？\n；]{0,180}(?:信息|说明|依据)[，,](?:因此|所以|故)?(?:我|我们)?(?:无法|不能)(?:确认|回答|给出)",
        normalized,
    ))
    return explicit or absence_and_refusal


@dataclass
class RAGAnswerEvaluationResult:
    case_id: str
    answer: str
    refused: bool
    refusal_correct: bool
    citation_presence_correct: bool
    citations_grounded: bool
    deterministic_checks_passed: bool
    rule_version: str = RAG_RULE_VERSION


def evaluate_rag_answer(
    case: RetrievalCase,
    answer:str,
    retrieval_results:list[RetrievalResult],
) -> RAGAnswerEvaluationResult:
    refused = is_insufficient_knowledge_answer(answer)

    refusal_correct = refused == (not case.answerable)

    citation_validation = validate_citations(answer, retrieval_results)

    if case.answerable:
        citation_presence_correct = citation_validation.has_citations
    else:
        # Refusals may cite supported background. Existence in the retrieved set
        # is checked below; semantic support and invented facts require Judge/human
        # review. This structural result must not be called factual correctness.
        citation_presence_correct = True

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
