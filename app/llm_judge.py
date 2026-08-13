import json

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    citation_support: int = Field(ge=1, le=5)
    reason: str

    @property
    def passed(self) -> bool:
        return all(
            score >= 4
            for score in [
                self.correctness,
                self.completeness,
                self.faithfulness,
                self.citation_support,
            ]
        )


JUDGE_SYSTEM_PROMPT = """
你是严格的RAG回答质量评审员。

请根据用户问题、参考答案、召回上下文和实际回答进行评分。

评分维度：
1. correctness：实际回答的事实是否正确。
2. completeness：实际回答是否覆盖问题要求的关键信息。
3. faithfulness：实际回答是否完全受到召回上下文支持。
4. citation_support：回答中的引用是否真正支持相应结论。

每项使用1到5的整数评分：
1表示严重不符合，5表示完全符合。

重要评分规则：

- 参考答案只能用于判断正确性和完整性，不能作为事实依据。
- 召回上下文是判断faithfulness和citation_support的唯一事实依据。
- 如果回答包含召回上下文没有明确支持、也不能直接推导出的具体事实，
  即使回答的其他部分正确，faithfulness也不得超过3分。
- 如果引用来源不能支持回答中的某个具体结论，
  citation_support不得超过3分。
- 不得因为回答整体看起来合理，就忽略其中无依据的新增事实。

只输出一个JSON对象，不要输出Markdown代码块或额外文字。
JSON必须包含：
correctness、completeness、faithfulness、
citation_support、reason。
""".strip()


class LLMJudge:

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def evaluate(self, question: str, reference_answer: str, answer: str, context: str) -> JudgeResult:
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"用户问题：\n{question}\n\n"
                    f"参考答案：\n{reference_answer}\n\n"
                    f"召回上下文：\n"
                    f"<knowledge>\n{context}\n</knowledge>\n\n"
                    f"实际回答：\n{answer}"
                )
            ),
        ]
        response = self.chat_model.invoke(messages)

        return parse_judge_response(response.content)


def parse_judge_response(content: str) -> JudgeResult:
    cleaned_content = content.strip()

    if cleaned_content.startswith("```"):
        lines = cleaned_content.splitlines()

        if lines[0].strip() in {"```json", "```"}:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_content = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned_content)
    except json.JSONDecodeError as error:
        raise ValueError("Judge返回内容不是有效JSON") from error

    return JudgeResult.model_validate(data)