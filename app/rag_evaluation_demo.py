import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from app.rag_agent import RAGAgent, INSUFFICIENT_KNOWLEDGE_RESPONSE
from app.rag_evaluator import evaluate_rag_answer
from app.document_splitter import split_documents_recursive
from app.retrieval_case_loader import load_retrieval_cases
from app.document_loader import load_knowledge_documents
from app.retriever import build_in_memory_retriever
from app.query_rewriter import QueryRewriter
from app.llm_judge import LLMJudge


def main():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY未配置")

    chat_model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    query_rewriter = QueryRewriter(chat_model)

    documents = load_knowledge_documents("data/knowledge")
    chunks = split_documents_recursive(documents, chunk_size=500, chunk_overlap= 50)
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    retriever = build_in_memory_retriever(chunks, embedding)
    agent = RAGAgent(chat_model, retriever, k=5, query_rewriter=query_rewriter)

    judge = LLMJudge(chat_model)
    judge_results = []

    cases = load_retrieval_cases(
        "data/evaluation/retrieval_cases.jsonl"
    )
    evaluation_results = []

    for case in cases:
        run_result = agent.run(case.question)

        evaluation_result = evaluate_rag_answer(
            case=case,
            answer=run_result.final_response.content,
            retrieval_results=run_result.retrieval_results,
        )
        evaluation_results.append(evaluation_result)

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
        judge_results.append(judge_result)

        print(f"用例：{case.case_id}")
        print(f"问题：{case.question}")
        print(f"预期可回答：{case.answerable}")
        print(
            "召回来源：",
            [
                result.document.id
                for result in run_result.retrieval_results
            ],
        )
        print(f"回答：{evaluation_result.answer}")
        print(f"是否拒答：{evaluation_result.refused}")
        print(
            "拒答判断正确："
            f"{evaluation_result.refusal_correct}"
        )
        print(
            "引用存在性正确："
            f"{evaluation_result.citation_presence_correct}"
        )
        print(
            "引用均来自召回结果："
            f"{evaluation_result.citations_grounded}"
        )
        print(
            "确定性检查通过："
            f"{evaluation_result.deterministic_checks_passed}"
        )
        print(f"原始问题：{run_result.query}")
        print(f"检索问题：{run_result.retrieval_query}")

        print(f"正确性：{judge_result.correctness}")
        print(f"完整性：{judge_result.completeness}")
        print(f"忠实度：{judge_result.faithfulness}")
        print(f"引用支持度：{judge_result.citation_support}")
        print(f"Judge通过：{judge_result.passed}")
        print(f"Judge理由：{judge_result.reason}")

        print("-" * 70)

    case_count = len(evaluation_results)

    refusal_accuracy = sum(result.refusal_correct for result in evaluation_results) / case_count

    citation_presence_accuracy = sum(result.citation_presence_correct for result in evaluation_results) / case_count

    citation_grounding_rate = sum(result.citations_grounded for result in evaluation_results) / case_count

    deterministic_pass_rate = sum( result.deterministic_checks_passed for result in evaluation_results) / case_count

    print(f"评测用例数：{case_count}")
    print(f"拒答判断准确率：{refusal_accuracy:.4f}")
    print(
        "引用存在性准确率："
        f"{citation_presence_accuracy:.4f}"
    )
    print(
        "引用来源合法率："
        f"{citation_grounding_rate:.4f}"
    )
    print(
        "确定性检查通过率："
        f"{deterministic_pass_rate:.4f}"
    )

    judge_case_count = len(judge_results)

    mean_correctness = sum(
        result.correctness for result in judge_results
    ) / judge_case_count

    mean_completeness = sum(
        result.completeness for result in judge_results
    ) / judge_case_count

    mean_faithfulness = sum(
        result.faithfulness for result in judge_results
    ) / judge_case_count

    mean_citation_support = sum(
        result.citation_support for result in judge_results
    ) / judge_case_count

    judge_pass_rate = sum(
        result.passed for result in judge_results
    ) / judge_case_count

    print(f"平均正确性：{mean_correctness:.4f}")
    print(f"平均完整性：{mean_completeness:.4f}")
    print(f"平均忠实度：{mean_faithfulness:.4f}")
    print(
        f"平均引用支持度："
        f"{mean_citation_support:.4f}"
    )
    print(f"Judge通过率：{judge_pass_rate:.4f}")

if __name__ == "__main__":
    main()