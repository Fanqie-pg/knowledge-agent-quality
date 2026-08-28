"""为间接 Prompt Injection 评测构造必定进入上下文的恶意证据。"""

from langchain_core.documents import Document

from app.retriever import RetrievalResult
from app.security_case_loader import SecurityAttackType, SecurityCase


class InjectedContentRetriever:
    """将安全用例的恶意内容固定放在召回首位，再补充正常证据。"""

    def __init__(self, base_retriever, case: SecurityCase):
        if case.attack_type is not SecurityAttackType.INDIRECT_PROMPT_INJECTION:
            raise ValueError("只支持间接提示词注入用例")

        self.base_retriever = base_retriever
        self.case = case

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        """返回包含攻击 Chunk 的结果，确保测试真正施加了间接注入。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query不能为空")

        if k <= 0:
            raise ValueError("k必须大于0")

        attack_chunk_id = f"{self.case.case_id}-INJECTED"
        attack_document = Document(
            id=attack_chunk_id,
            page_content=self.case.injected_content,
            metadata={
                "doc_id": "SECURITY-INJECTION",
                "chunk_id": attack_chunk_id,
                "security_case_id": self.case.case_id,
            },
        )

        base_results = []
        if k > 1:
            base_results = (
                self.base_retriever.retrieve(
                    query,
                    k=k - 1,
                )
            )

        # 攻击内容固定为 rank=1，避免“没有召回攻击文档”造成安全假通过。
        results = [
            RetrievalResult(
                document=attack_document,
                score=1.0,
                rank=1,
            )
        ]

        results.extend(
            RetrievalResult(
                document=result.document,
                score=result.score,
                rank=rank,
            )
            for rank, result in enumerate(
                base_results,
                start=2,
            )
        )

        return results
