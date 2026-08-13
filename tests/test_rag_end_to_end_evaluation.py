import os

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


pytestmark = pytest.mark.evaluation


@pytest.fixture(scope="module")
def rag_evaluation_system():
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")

    chat_model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )

    documents = load_knowledge_documents("data/knowledge")
    chunks = split_documents_recursive(
        documents,
        chunk_size=500,
        chunk_overlap=50,
    )

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    retriever = build_in_memory_retriever(chunks, embedding)

    agent = RAGAgent(
        chat_model=chat_model,
        retriever=retriever,
        k=5,
        query_rewriter=QueryRewriter(chat_model),
    )

    judge = LLMJudge(chat_model)

    return agent, judge

def test_rag_end_to_end_answers_multi_document_question(rag_evaluation_system):
    agent, judge = rag_evaluation_system
    cases = load_retrieval_cases(
        "data/evaluation/retrieval_cases.jsonl"
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
        "data/evaluation/retrieval_cases.jsonl"
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

    assert deterministic_pass_rate == 1.0, summary
    assert judge_pass_rate >= 0.9, summary
    assert mean_correctness >= 4.5, summary
    assert mean_completeness >= 4.5, summary
    assert mean_faithfulness >= 4.5, summary
    assert mean_citation_support >= 4.5, summary