import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalCase:
    case_id: str
    question: str
    relevant_doc_ids: list[str]
    reference_answer: str
    answerable: bool
    category: str
    difficulty: str


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

            cases.append(RetrievalCase(**data))

        if not cases:
            raise ValueError("检索评测文件不能为空")

        return cases