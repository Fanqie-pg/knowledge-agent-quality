def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k必须大于0")

    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids不能为空")

    top_k_doc_ids = set(retrieved_doc_ids[:k])
    relevant_doc_id_set = set(relevant_doc_ids)

    recalled_count = len(top_k_doc_ids & relevant_doc_id_set)

    return recalled_count / len(relevant_doc_id_set)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> float:
    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids不能为空")

    relevant_doc_id_set = set(relevant_doc_ids)

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_id_set:
            return 1.0 /rank

    return 0.0


def mean_reciprocal_rank(retrieved_doc_ids_by_case: list[list[str]], relevant_doc_ids_by_case: list[list[str]]) -> float:
    if not retrieved_doc_ids_by_case:
        raise ValueError("评测用例不能为空")

    if len(retrieved_doc_ids_by_case) != len(
        relevant_doc_ids_by_case
    ):
        raise ValueError("检索结果与相关文档数量不一致")

    reciprocal_ranks = [
        reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)
        for retrieved_doc_ids, relevant_doc_ids in zip(
            retrieved_doc_ids_by_case,
            relevant_doc_ids_by_case,
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evidence_recall_at_k(retrieved_contents: list[str], required_evidence: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k必须大于0")

    if not required_evidence:
        raise ValueError("required_evidence不能为空")

    top_k_context = "\n".join(retrieved_contents[:k])

    normalized_context = "".join(top_k_context.split()).lower()

    matched_count = 0
    for evidence in required_evidence:
        normalized_evidence = "".join(
            evidence.split()
        ).lower()

        if normalized_evidence in normalized_context:
            matched_count += 1

    return matched_count / len(required_evidence)