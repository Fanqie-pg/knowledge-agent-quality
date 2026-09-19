import pytest

pytestmark = pytest.mark.integration

import logging
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.retriever import KnowledgeRetriever, build_in_memory_retriever, RetrievalUnavailableError


class FakeVectorStore:

    def __init__(self, matches):
        self.matches = matches
        self.last_query = None
        self.last_k = None

    def similarity_search_with_score(self, query, k):
        self.last_query = query
        self.last_k = k
        return self.matches[:k]


def test_retrieve_returns_ranked_result():
    first_document = Document(
        id="KB-FAQ-001-CHUNK-000",
        page_content="账号连续输错五次后会被锁定。",
        metadata={"doc_id": "KB-FAQ-001"},
    )
    second_document = Document(
        id="KB-API-001-CHUNK-004",
        page_content="接口返回通用错误码。",
        metadata={"doc_id": "KB-API-001"},
    )

    fake_store = FakeVectorStore(
        [
            (first_document, 0.6373),
            (second_document, 0.5433),
        ]
    )
    retriever = KnowledgeRetriever(fake_store)

    results = retriever.retrieve(
        "账号被锁定怎么办？",
        k=2,
    )

    assert fake_store.last_query == "账号被锁定怎么办？"
    assert fake_store.last_k == 2

    assert len(results) == 2

    assert results[0].document is first_document
    assert results[0].score == 0.6373
    assert results[0].rank == 1

    assert results[1].document is second_document
    assert results[1].score == 0.5433
    assert results[1].rank == 2

@pytest.mark.parametrize(
    "invalid_query",
    [None, "", " ", "\t", "\n"],
)

def test_retrieve_rejects_empty_query(invalid_query):
    fake_store = FakeVectorStore([])
    retriever = KnowledgeRetriever(fake_store)

    with pytest.raises(ValueError, match="query不能为空"):
        retriever.retrieve(invalid_query)

    assert fake_store.last_query is None


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_retrieve_rejects_invalid_k(invalid_k):
    fake_store = FakeVectorStore([])
    retriever = KnowledgeRetriever(fake_store)

    with pytest.raises(ValueError, match="k必须大于0"):
        retriever.retrieve("有效问题", invalid_k)

    assert fake_store.last_query is None


def test_build_in_memory_retriever_rejects_empty_documents():
    embedding = DeterministicFakeEmbedding(size=8)

    with pytest.raises(ValueError, match="待索引文档不能为空"):
        build_in_memory_retriever([], embedding)


def test_build_in_memory_retriever_indexes_documents():
    document = Document(
        id="KB-TEST-001-CHUNK-000",
        page_content="账号锁定规则",
        metadata={"doc_id": "KB-TEST-001"},
    )
    embedding = DeterministicFakeEmbedding(size=8)

    retriever = build_in_memory_retriever([document], embedding)
    results = retriever.retrieve("账号锁定", k=1)

    assert len(results) == 1
    assert results[0].document.id ==  "KB-TEST-001-CHUNK-000"
    assert results[0].rank == 1


class FailingVectorStore:

    def similarity_search_with_score(self, query, k):
        raise ConnectionError("vector-store-internal-secret")


def test_retrieve_converts_infrastructure_error(caplog):
    retriever = KnowledgeRetriever(FailingVectorStore())

    with caplog.at_level(logging.ERROR, logger="app.retriever"):
        with pytest.raises(
            RetrievalUnavailableError,
            match="知识库检索服务暂时不可用",
        ) as exc_info:
            retriever.retrieve("P1工单多久响应？")

    assert "ConnectionError" in caplog.text
    assert "vector-store-internal-secret" not in caplog.text
    assert "vector-store-internal-secret" not in str(exc_info.value)
