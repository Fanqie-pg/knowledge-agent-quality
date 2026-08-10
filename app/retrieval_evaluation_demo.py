from langchain_huggingface import HuggingFaceEmbeddings

from app.document_loader import load_knowledge_documents
from app.document_splitter import split_documents_recursive
from app.retrieval_case_loader import load_retrieval_cases
from app.retrieval_evaluator import evaluate_retrieval_cases
from app.retriever import build_in_memory_retriever

def main():
    documents = load_knowledge_documents(
        "data/knowledge"
    )
    chunks = split_documents_recursive(
        documents,
        chunk_size=500,
        chunk_overlap=50,
    )
    cases = load_retrieval_cases(
        "data/evaluation/retrieval_cases.jsonl"
    )

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    retriever = build_in_memory_retriever(
        chunks,
        embedding,
    )
    summary = evaluate_retrieval_cases(
        retriever,
        cases,
        k=3,
    )

    print(f"原始文档数：{len(documents)}")
    print(f"切分块数：{len(chunks)}")
    print(f"评测用例数：{summary.evaluated_case_count}")
    print(f"跳过不可回答用例数：{summary.skipped_unanswerable_count}")
    print("-" * 60)

    for result in summary.case_results:
        print(f"用例：{result.case_id}")
        print(f"问题：{result.question}")
        print(f"期望文档：{result.relevant_doc_ids}")
        print(f"召回文档：{result.retrieved_doc_ids}")
        print(f"Recall@{summary.k}：{result.recall_at_k:.4f}")
        print(f"RR：{result.reciprocal_rank:.4f}")
        print("-" * 60)

    print(f"Mean Recall@{summary.k}：{summary.mean_recall_at_k:.4f}")
    print(f"MRR：{summary.mrr:.4f}")

if __name__ == "__main__":
    main()