import os
import json
from dataclasses import asdict

import pytest
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from app.document_loader import load_knowledge_documents
from app.document_splitter import split_documents_recursive
from app.llm_judge import LLMJudge
from app.query_rewriter import QueryRewriter
from app.rag_agent import RAGAgent, INSUFFICIENT_KNOWLEDGE_RESPONSE
from app.rag_evaluator import evaluate_rag_answer
from app.retrieval_case_loader import load_retrieval_cases
from app.retriever import build_in_memory_retriever
from app.tracing import emit


pytestmark = pytest.mark.evaluation


@pytest.fixture(scope="module")
def rag_evaluation_system():
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")

    from app.agent_factory import build_chat_model, build_knowledge_retriever, build_rag_agent
    chat_model = build_chat_model()
    agent = build_rag_agent(chat_model, build_knowledge_retriever())

    judge = LLMJudge()

    return agent, judge

def test_rag_end_to_end_answers_multi_document_question(rag_evaluation_system):
    agent, judge = rag_evaluation_system
    cases = load_retrieval_cases(
        "data/evaluation/baseline_v1/retrieval_cases.jsonl"
    )
    case = next(
        case for case in cases
        if case.case_id == "RET-006"
    )

    run_result = agent.run(case.question)

    deterministic_result = evaluate_rag_answer(
        case=case,
        answer=run_result.final_response.content,
        retrieval_results=run_result.retrieval_results,
    )

    assert case.reference_answer is not None

    judge_result = judge.evaluate(
        question=case.question,
        reference_answer=case.reference_answer,
        answer=run_result.final_response.content,
        context=run_result.context,
    )

    retrieved_doc_ids = {
        result.document.metadata["doc_id"]
        for result in run_result.retrieval_results
    }

    assert run_result.query == case.question
    assert run_result.retrieval_query.strip()
    assert set(case.relevant_doc_ids) <= retrieved_doc_ids
    assert deterministic_result.deterministic_checks_passed
    assert judge_result.passed, judge_result.model_dump()


@pytest.mark.quality_gate
def test_rag_quality_gate_over_evaluation_dataset(rag_evaluation_system):
    agent, judge = rag_evaluation_system
    cases = load_retrieval_cases(
        "data/evaluation/baseline_v1/retrieval_cases.jsonl"
    )

    deterministic_results = []
    judge_results = []
    case_details = []

    for case in cases:
        run_result = agent.run(case.question)

        deterministic_result = evaluate_rag_answer(
            case=case,
            answer=run_result.final_response.content,
            retrieval_results=run_result.retrieval_results,
        )

        emit("rag_evaluation", "answer_record", case_id=case.case_id, question=case.question,
             answer=run_result.final_response.content, context=run_result.context,
             retrieval_query=run_result.retrieval_query, rules=asdict(deterministic_result),
             retrieval=[{"doc_id":item.document.metadata["doc_id"],"chunk_id":item.document.metadata["chunk_id"]} for item in run_result.retrieval_results])

        reference_answer = (
            case.reference_answer
            if case.reference_answer is not None
            else INSUFFICIENT_KNOWLEDGE_RESPONSE
        )

        judge_result = judge.evaluate(
            question=case.question,
            reference_answer=reference_answer,
            answer=run_result.final_response.content,
            context=run_result.context,
        )

        deterministic_results.append(deterministic_result)
        judge_results.append(judge_result)
        emit("rag_evaluation", "judge_record", case_id=case.case_id, scores=judge_result.model_dump(), passed=judge_result.passed)

        case_details.append(
            {
                "case_id": case.case_id,
                "deterministic_passed": deterministic_result.deterministic_checks_passed,
                "judge_passed": judge_result.passed,
                "judge_scores": judge_result.model_dump(),
                "answer": run_result.final_response.content,
            }
        )

    case_count = len(cases)

    deterministic_pass_rate = sum(
        result.deterministic_checks_passed
        for result in deterministic_results
    ) / case_count

    judge_pass_rate = sum(
        result.passed for result in judge_results
    ) / case_count

    mean_correctness = sum(
        result.correctness for result in judge_results
    ) / case_count

    mean_completeness = sum(
        result.completeness for result in judge_results
    ) / case_count

    mean_faithfulness = sum(
        result.faithfulness for result in judge_results
    ) / case_count

    mean_citation_support = sum(
        result.citation_support for result in judge_results
    ) / case_count

    summary = {
        "case_count": case_count,
        "deterministic_pass_rate": deterministic_pass_rate,
        "judge_pass_rate": judge_pass_rate,
        "mean_correctness": mean_correctness,
        "mean_completeness": mean_completeness,
        "mean_faithfulness": mean_faithfulness,
        "mean_citation_support": mean_citation_support,
        "case_details": case_details,
    }

    print(json.dumps(summary,ensure_ascii=False))

    assert deterministic_pass_rate == 1.0, summary
    assert judge_pass_rate >= 0.9, summary
    assert mean_correctness >= 4.5, summary
    assert mean_completeness >= 4.5, summary
    assert mean_faithfulness >= 4.5, summary
    assert mean_citation_support >= 4.5, summary
