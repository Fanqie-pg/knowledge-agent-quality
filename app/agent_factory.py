"""集中组装模型、知识检索器和 UnifiedAgent 的运行依赖。"""

import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from langchain_core.tools import StructuredTool

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
from app.business_repository import SQLiteBusinessRepository
from app.mcp_tool_registry import build_mcp_tools


def build_unified_agent(tools: list[StructuredTool] | None = None, *, rewrite_queries: bool = True, user_context=None, repository=None, tool_profile="current", transport="local") -> UnifiedAgent:
    """构建统一 Agent；允许注入本地工具或 MCP 适配后的工具。"""
    chat_model = build_chat_model()
    retriever = build_knowledge_retriever()

    router = IntentRouter(chat_model)

    if transport not in {"local", "mcp"}:
        raise ValueError("未知工具接入方式")
    request_tools = None
    if tools is None:
        repository = repository or SQLiteBusinessRepository()
        tools = build_tools(repository=repository, profile=tool_profile)
        def request_tools(context):
            if transport == "local":
                return build_tools(user_context=context, repository=repository, profile=tool_profile)
            return build_mcp_tools(user_context=context, database_path=repository.path, profile=tool_profile)
    tool_agent = ToolAgent(chat_model, tools, user_context=user_context, tools_factory=request_tools)

    rag_agent = build_rag_agent(chat_model, retriever, rewrite_queries=rewrite_queries)
    return UnifiedAgent(
        router=router,
        tool_agent=tool_agent,
        rag_agent=rag_agent,
        chat_model=chat_model,
        user_context=user_context,
    )


def build_rag_agent(chat_model, retriever, *, rewrite_queries=True, k=5):
    """Runtime and experiments share identical RAG assembly."""
    return RAGAgent(chat_model, retriever, k=k,
                    query_rewriter=QueryRewriter(chat_model) if rewrite_queries else None)


def build_conversation_agent(*, store=None, repository=None, transport="local", **limits):
    """Share model and agents across users; pass trusted identity on every turn."""
    from app.conversation_agent import ConversationAgent
    from app.conversation_store import ConversationStore
    repository = repository or SQLiteBusinessRepository()
    unified = build_unified_agent(repository=repository, transport=transport)
    return ConversationAgent(unified, store or ConversationStore(), repository, **limits)


def build_chat_model() -> RetryingChatModel:
    """创建 DeepSeek 客户端，并在项目边界统一配置有限重试。"""
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

    model = RetryingChatModel(
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
    if os.getenv("EVALUATION_MAX_CALLS"):
        from app.evaluation_records import CallBudget, attach_budget
        budget = CallBudget(os.environ["EVALUATION_CALL_LEDGER"], int(os.environ["EVALUATION_MAX_CALLS"]))
        model = attach_budget(model, budget)
    return model


def build_knowledge_retriever() -> KnowledgeRetriever:
    """加载并切分本地知识库，使用中文 Embedding 构建内存检索器。"""
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
