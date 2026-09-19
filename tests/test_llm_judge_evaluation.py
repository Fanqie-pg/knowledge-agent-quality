import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.llm_judge import LLMJudge

pytestmark = pytest.mark.evaluation


QUESTION = "账号连续输错几次会被锁定，多久恢复？"

REFERENCE_ANSWER = (
    "连续输错5次后账号锁定30分钟，"
    "30分钟后自动解锁。"
)

CONTEXT = (
    "[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-000]\n"
    "同一个账号连续输入错误密码5次后，系统会将账号锁定30分钟。"
    "账号会在30分钟后自动解锁。"
)


@pytest.fixture(scope="module")
def judge():
    load_dotenv()

    api_key = os.getenv("QWEN_API_KEY") or os.getenv("QWENT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        pytest.skip("未配置 QWEN_API_KEY")

    return LLMJudge()


def test_judge_accepts_correct_grounded_answer(judge):
    result = judge.evaluate(
        question=QUESTION,
        reference_answer=REFERENCE_ANSWER,
        answer=(
            "账号连续输错5次后会被锁定30分钟，之后自动解锁。"
            "[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-000]"
        ),
        context=CONTEXT,
    )

    assert result.correctness >= 4
    assert result.completeness >= 4
    assert result.faithfulness >= 4
    assert result.citation_support >= 4
    assert result.passed is True


def test_judge_rejects_factually_wrong_answer(judge):
    result = judge.evaluate(
        question=QUESTION,
        reference_answer=REFERENCE_ANSWER,
        answer=(
            "账号连续输错8次后会被锁定60分钟。"
            "[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-000]"
        ),
        context=CONTEXT,
    )
    assert result.correctness <= 2
    assert result.passed is False


@pytest.mark.parametrize(
    ("answer", "failed_metric"),
    [
        (
            (
                "账号连续输错5次后会被锁定。"
                "[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-000]"
            ),
            "completeness",
        ),
        (
            (
                "账号连续输错5次后会被锁定30分钟，之后自动解锁。"
                "同时管理员会收到短信通知。"
                "[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-000]"
            ),
            "faithfulness",
        ),
        (
            (
                "账号连续输错5次后会被锁定30分钟，之后自动解锁。"
                "[来源：KB-API-001/KB-API-001-CHUNK-000]"
            ),
            "citation_support",
        ),
    ],
    ids=["incomplete", "hallucinated", "wrong-citation"],
)
def test_judge_rejects_specific_quality_defects(judge, answer, failed_metric):
    result = judge.evaluate(
        question=QUESTION,
        reference_answer=REFERENCE_ANSWER,
        answer=answer,
        context=CONTEXT,
    )
    failed_score = getattr(result, failed_metric)
    assert failed_score < 4, result.model_dump()
    assert result.passed is False
