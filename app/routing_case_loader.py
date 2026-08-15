import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.intent_router import AgentRoute


class RoutingCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_route: AgentRoute
    category: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]


def load_routing_cases(path: str | Path) -> list[RoutingCase]:
    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        raise ValueError("路由评测文件不存在")

    cases = []
    seen_case_ids = set()

    with file_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"第{line_number}行不是有效JSON") from error

            try:
                case = RoutingCase.model_validate(data)
            except ValidationError as error:
                raise ValueError(f"第{line_number}行字段不合法") from error

            if case.case_id in seen_case_ids:
                raise ValueError(f"发现重复case_id：{case.case_id}")

            seen_case_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise ValueError("路由评测文件不能为空")

    return cases