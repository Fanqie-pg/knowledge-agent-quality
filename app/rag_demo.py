import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from app.document_loader import load_knowledge_documents
from app.document_splitter import split_documents_recursive
from app.rag_agent import RAGAgent
from app.retriever import build_in_memory_retriever
from app.query_rewriter import QueryRewriter


def main():
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY未配置")

    documents = load_knowledge_documents("data/knowledge")

    chunks = split_documents_recursive(documents, chunk_size=500, chunk_overlap=50)

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    retriever = build_in_memory_retriever(chunks, embedding=embedding)

    chat_model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    query_rewriter = QueryRewriter(chat_model)
    agent = RAGAgent(chat_model, retriever, k=5, query_rewriter=query_rewriter)

    questions = [
        "登录口令连续写错多少回会被封？多长时间后自己恢复？",
    ]

    for question in questions:
        result = agent.run(question)

        print(f"用户问题：{question}")
        print("找回结果：")
        for retrieval_result in result.retrieval_results:
            document = retrieval_result.document
            print(
                f"  rank={retrieval_result.rank}, "
                f"score={retrieval_result.score}, "
                f"doc_id={document.metadata['doc_id']}, "
                f"chunk_id={document.metadata['chunk_id']}"
            )

        print("最终回答：", result.final_response.content)
        print(f"检索问题：{result.retrieval_query}")
        print("-" * 70)


if __name__ == "__main__":
    main()