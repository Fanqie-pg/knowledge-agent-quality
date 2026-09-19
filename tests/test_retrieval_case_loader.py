import pytest

pytestmark = pytest.mark.unit

from app.retrieval_case_loader import load_retrieval_cases


def test_load_retrieval_cases_loads_controlled_dataset():
    cases = load_retrieval_cases(
        "data/evaluation/retrieval_cases.jsonl"
    )

    assert len(cases) == 60
    assert cases[0].case_id == "RET-001"
    assert cases[0].question == "P1工单需要在多长时间内首次响应？"
    assert cases[0].relevant_doc_ids == ["KB-PRODUCT-001"]
    assert cases[0].answerable is True

    answerable_cases = [
        case for case in cases if case.answerable
    ]
    unanswerable_cases = [
        case for case in cases if not case.answerable
    ]

    assert len(answerable_cases) == 45
    assert len(unanswerable_cases) == 15
    assert unanswerable_cases[0].relevant_doc_ids == []