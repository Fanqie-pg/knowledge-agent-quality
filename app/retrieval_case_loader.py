import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalCase:
    case_id: str
    question: str
    relevant_doc_ids: list[str]
    reference_answer: str | None
    answerable: bool
    category: str
    difficulty: str
    required_evidence: list[str]

    def __post_init__(self):
        for name in ("case_id", "question", "category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name}不能为空")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError("difficulty不合法")
        if not isinstance(self.answerable, bool):
            raise ValueError("answerable必须是布尔值")
        for name in ("relevant_doc_ids", "required_evidence"):
            value = getattr(self, name)
            if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
                raise ValueError(f"{name}必须是非空字符串列表")
            if len(value) != len(set(value)):
                raise ValueError(f"{name}存在重复值")
        if self.answerable:
            if not self.relevant_doc_ids or not self.required_evidence or not isinstance(self.reference_answer, str) or not self.reference_answer.strip():
                raise ValueError("可回答用例必须提供文档、证据和参考答案")
        elif self.relevant_doc_ids or self.required_evidence or self.reference_answer is not None:
            raise ValueError("不可回答用例不能提供参考答案或正向证据")


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        raise ValueError("检索评测文件不存在")

    cases = []

    with file_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第{line_number}行不是有效JSON") from error

            try:
                case = RetrievalCase(**data)
            except (TypeError, ValueError) as error:
                raise ValueError(f"第{line_number}行字段不合法：{error}") from error
            if any(existing.case_id == case.case_id for existing in cases):
                raise ValueError(f"发现重复case_id：{case.case_id}")
            cases.append(case)

        if not cases:
            raise ValueError("检索评测文件不能为空")

        return cases
