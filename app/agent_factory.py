import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from app.document_loader import load_knowledge_documents
from app.document_splitter import split_documents_recursive
from app.intent_router import IntentRouter
from app.query_rewriter import QueryRewriter
from app.rag_agent import RAGAgent
from app.retriever import build_in_memory_retriever, KnowledgeRetriever
from app.tool_agent import ToolAgent
from app.tool_registry import build_tools
from app.unified_agent import UnifiedAgent
from app.resilient_model import RetryingChatModel


def build_unified_agent() -> UnifiedAgent:
    chat_model = build_chat_model()
    retriever = build_knowledge_retriever()

    router = IntentRouter(chat_model)

    tool_agent = ToolAgent(chat_model, build_tools())

    rag_agent = RAGAgent(
        chat_model=chat_model,
        retriever=retriever,
        k=5,
        query_rewriter=QueryRewriter(chat_model),
    )
    return UnifiedAgent(
        router=router,
        tool_agent=tool_agent,
        rag_agent=rag_agent,
        chat_model=chat_model,
    )


def build_chat_model() -> RetryingChatModel:
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY未配置"
        )

    base_chat_model = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0,
        timeout=30,
        max_retries=0,
    )

    return RetryingChatModel(
        model=base_chat_model,
        max_attempts=2,
        retry_delay_seconds=0.5,
        retryable_exceptions=(
            TimeoutError,
            ConnectionError,
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        ),
    )


def build_knowledge_retriever() -> KnowledgeRetriever:
    documents = load_knowledge_documents(
        "data/knowledge"
    )
    chunks = split_documents_recursive(
        documents,
        chunk_size=500,
        chunk_overlap=50,
    )

    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return build_in_memory_retriever(
        chunks,
        embedding,
    )