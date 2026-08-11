from dataclasses import dataclass

from app.retrieval_case_loader import RetrievalCase
from app.retrieval_metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank, evidence_recall_at_k
from app.retriever import KnowledgeRetriever


@dataclass
class RetrievalCaseResult:
    case_id: str
    question: str
    retrieved_doc_ids: list[str]
    recall_at_k: float
    reciprocal_rank: float
    relevant_doc_ids: list[str]
    required_evidence: list[str]
    evidence_recall_at_k: float


@dataclass
class RetrievalEvaluationSummary:
    k: int
    evaluated_case_count: int
    skipped_unanswerable_count: int
    mean_recall_at_k: float
    mrr: float
    case_results: list[RetrievalCaseResult]
    mean_evidence_recall_at_k: float


def evaluate_retrieval_case(retriever: KnowledgeRetriever, case: RetrievalCase, k: int = 3) -> RetrievalCaseResult:
    if not case.answerable:
        raise ValueError("不可回答用例不能计算Recall和RR")

    retrieval_results = retriever.retrieve(case.question, k=k)

    retrieved_doc_ids = [
        result.document.metadata["doc_id"]
        for result in retrieval_results
    ]
    retrieved_contents = [
        result.document.page_content
        for result in retrieval_results
    ]

    return RetrievalCaseResult(
        case_id=case.case_id,
        question=case.question,
        retrieved_doc_ids=retrieved_doc_ids,
        recall_at_k=recall_at_k(retrieved_doc_ids, case.relevant_doc_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_doc_ids, case.relevant_doc_ids),
        relevant_doc_ids=case.relevant_doc_ids,
        required_evidence=case.required_evidence,
        evidence_recall_at_k=evidence_recall_at_k(
            retrieved_contents,
            case.required_evidence,
            k,
        ),
    )


def evaluate_retrieval_cases(retriever: KnowledgeRetriever, cases:list[RetrievalCase], k: int = 3) -> RetrievalEvaluationSummary:
    if k <= 0:
        raise ValueError("k必须大于0")

    answerable_cases = [
        case for case in cases if case.answerable
    ]

    if not answerable_cases:
        raise ValueError("没有可用于检索评测的用例")

    case_results = [
        evaluate_retrieval_case(retriever, case, k)
        for case in answerable_cases
    ]

    mean_recall = sum(result.recall_at_k for result in case_results) / len(case_results)

    mrr = mean_reciprocal_rank(
        retrieved_doc_ids_by_case=[result.retrieved_doc_ids for result in case_results],
        relevant_doc_ids_by_case=[case.relevant_doc_ids for case in answerable_cases]
    )

    mean_evidence_recall = sum(
        result.evidence_recall_at_k for result in case_results
    ) / len(case_results)

    return RetrievalEvaluationSummary(
        k=k,
        evaluated_case_count=len(answerable_cases),
        skipped_unanswerable_count=(len(cases) - len(answerable_cases)),
        mean_recall_at_k= mean_recall,
        mrr=mrr,
        case_results=case_results,
        mean_evidence_recall_at_k=mean_evidence_recall,
    )