from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.rag_agent import format_retrieval_context, RAGAgent
from app.retriever import RetrievalResult


class FakeQueryRewriter:

    def __init__(self, rewritten_query):
        self.rewritten_query = rewritten_query
        self.last_query = None

    def rewrite(self, query):
        self.last_query = query
        return self.rewritten_query


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.last_query = None
        self.last_k = None

    def retrieve(self, query, k=3):
        self.last_query = query
        self.last_k = k
        return self.results[:k]


class FakeChatModel:

    def __init__(self, response):
        self.response = response
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return self.response


def test_format_retrieval_context_includes_content_and_sources():
    first_document = Document(
        id="KB-FAQ-001-CHUNK-000",
        page_content="连续输错5次后，账号锁定30分钟。",
        metadata={
            "doc_id": "KB-FAQ-001",
            "chunk_id": "KB-FAQ-001-CHUNK-000",
        },
    )
    second_document = Document(
        id="KB-FAQ-001-CHUNK-001",
        page_content="密码重置链接只能使用一次。",
        metadata={
            "doc_id": "KB-FAQ-001",
            "chunk_id": "KB-FAQ-001-CHUNK-001",
        },
    )

    retrieval_results = [
        RetrievalResult(
            document=first_document,
            score=0.8,
            rank=1,
        ),
        RetrievalResult(
            document=second_document,
            score=0.7,
            rank=2,
        ),
    ]

    context = format_retrieval_context(retrieval_results)

    assert ("[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-000") in context
    assert "连续输错5次后，账号锁定30分钟。" in context
    assert ("[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-001") in context
    assert "密码重置链接只能使用一次。" in context
    assert "0.8" not in context
    assert "0.7" not in context


def test_rag_agent_retrieves_context_and_calls_model():
    document = Document(
        id="KB-FAQ-001-CHUNK-000",
        page_content="连续输错5次后，账号锁定30分钟。",
        metadata={
            "doc_id": "KB-FAQ-001",
            "chunk_id": "KB-FAQ-001-CHUNK-000",
        },
    )
    retrieval_result = RetrievalResult(
        document=document,
        score=0.8,
        rank=1,
    )

    retriever = FakeRetriever([retrieval_result])
    expected_response = AIMessage(
        content=(
            "连续输错5次后，账号会锁定30分钟。"
            "[来源：KB-FAQ-001]"
        )
    )
    chat_model = FakeChatModel(expected_response)
    agent = RAGAgent(chat_model, retriever, k=1)
    result = agent.run("账号输错几次会锁定？")

    assert retriever.last_query == "账号输错几次会锁定？"
    assert retriever.last_k == 1

    assert len(chat_model.last_messages) == 2
    assert isinstance(chat_model.last_messages[0], SystemMessage)
    assert isinstance(chat_model.last_messages[1], HumanMessage)

    human_content = chat_model.last_messages[1].content
    assert "连续输错5次后，账号锁定30分钟。" in human_content
    assert "账号输错几次会锁定？" in human_content
    assert "<knowledge>" in human_content
    assert "</knowledge>" in human_content

    assert result.retrieval_results == [retrieval_result]
    assert result.final_response is expected_response


def test_rag_agent_uses_rewritten_query_only_for_retrieval():
    original_query = (
        "登录口令连续写错多少回会被封？"
        "多长时间后自己恢复？"
    )
    rewritten_query = (
        "账号连续输入错误密码几次会被锁定，"
        "多久自动解锁？"
    )
    document = Document(
        id="KB-FAQ-001-CHUNK-000",
        page_content="连续输错5次后，账号锁定30分钟。",
        metadata={
            "doc_id": "KB-FAQ-001",
            "chunk_id": "KB-FAQ-001-CHUNK-000",
        },
    )
    retrieval_result = RetrievalResult(
        document=document,
        score=0.8,
        rank=1,
    )
    retriever = FakeRetriever([retrieval_result])
    rewriter = FakeQueryRewriter(rewritten_query)

    chat_model = FakeChatModel(AIMessage(content="测试回答"))

    agent = RAGAgent(
        chat_model=chat_model,
        retriever=retriever,
        k=1,
        query_rewriter=rewriter,
    )

    result = agent.run(original_query)
    assert rewriter.last_query == original_query
    assert retriever.last_query == rewritten_query
    assert result.query == original_query
    assert result.retrieval_query == rewritten_query

    human_content = chat_model.last_messages[1].content
    assert original_query in human_content
    assert rewritten_query not in human_content