import pytest
from pydantic import ValidationError

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.llm_judge import JudgeResult, LLMJudge

pytestmark = pytest.mark.unit


def test_judge_result_passes_when_all_scores_reach_threshold():
    result = JudgeResult(
        correctness=5,
        completeness=4,
        faithfulness=5,
        citation_support=4,
        reason="答案正确完整，并且有上下文和引用支持。",
    )

    assert result.passed is True


def test_judge_result_fails_when_any_score_is_below_threshold():
    result = JudgeResult(
        correctness=5,
        completeness=3,
        faithfulness=5,
        citation_support=5,
        reason="答案遗漏了一项关键信息。",
    )
    assert result.passed is False


@pytest.mark.parametrize("invalid_score", [0, 6])
def test_judge_result_rejects_score_outside_range(invalid_score):
    with pytest.raises(ValidationError):
        JudgeResult(
            correctness=invalid_score,
            completeness=5,
            faithfulness=5,
            citation_support=5,
            reason="测试非法分数。",
        )


class FakeJudgeModel:

    def __init__(self, content):
        self.content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(content=self.content)


def test_llm_judge_builds_prompt_and_parses_result():
    model = FakeJudgeModel(
        """
        {
          "correctness": 5,
          "completeness": 4,
          "faithfulness": 5,
          "citation_support": 5,
          "reason": "答案正确，证据充分。"
        }
        """
    )
    judge = LLMJudge(model)

    result = judge.evaluate(
        question="账号锁定多久？",
        reference_answer="账号锁定30分钟。",
        answer=(
            "账号锁定30分钟。"
            "[来源：KB-FAQ-001/KB-FAQ-001-CHUNK-000]"
        ),
        context="账号连续输错5次后锁定30分钟。",
    )

    assert result.correctness == 5
    assert result.completeness == 4
    assert result.passed is True

    assert len(model.last_messages) == 2
    assert isinstance(model.last_messages[0], SystemMessage)
    assert isinstance(model.last_messages[1], HumanMessage)

    prompt = model.last_messages[1].content
    assert "账号锁定多久？" in prompt
    assert "账号锁定30分钟。" in prompt
    assert "<knowledge>" in prompt


def test_llm_judge_rejects_non_json_response():
    model = FakeJudgeModel("这个回答很好。")
    judge = LLMJudge(model)

    with pytest.raises(ValueError, match="Judge返回内容不是有效JSON"):
        judge.evaluate(
            question="测试问题",
            reference_answer="参考答案",
            answer="实际回答",
            context="召回上下文",
        )


def test_llm_judge_parses_markdown_json_code_block():
    model = FakeJudgeModel(
        """
        ```json
        {
          "correctness": 5,
          "completeness": 5,
          "faithfulness": 4,
          "citation_support": 5,
          "reason": "答案受到上下文支持。"
        }
        ```
        """
    )
    judge = LLMJudge(model)

    result = judge.evaluate(
        question="测试问题",
        reference_answer="参考答案",
        answer="实际回答",
        context="召回上下文",
    )
    assert result.correctness == 5
    assert result.faithfulness == 4
    assert result.passed is True