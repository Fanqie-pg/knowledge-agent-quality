import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


logger = logging.getLogger(__name__)


class RetrievalUnavailableError(RuntimeError):
    """知识库检索基础设施暂时不可用。"""


@dataclass
class RetrievalResult:
    document: Document
    score: float
    rank: int


class KnowledgeRetriever:

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query不能为空")

        if k <= 0:
            raise ValueError("k必须大于0")

        try:
            matches = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error(
                "知识库检索失败：error_type=%s",
                type(exc).__name__,
            )
            raise RetrievalUnavailableError(
                "知识库检索服务暂时不可用"
            ) from exc
        return [
            RetrievalResult(
                document=document,
                score=float(score),
                rank=rank,
            )
            for rank, (document, score) in enumerate(
                matches,
                start=1,
            )
        ]


def build_in_memory_retriever(documents: list[Document], embedding: Embeddings) -> KnowledgeRetriever:
    if not documents:
        raise ValueError("待索引文档不能为空")

    vector_store = InMemoryVectorStore(embedding=embedding)
    vector_store.add_documents(documents)

    return KnowledgeRetriever(vector_store)