from enum import Enum
import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, ValidationError, model_validator


class SecurityAttackType(str, Enum):
    DIRECT_PROMPT_INJECTION = "direct_prompt_injection"
    INDIRECT_PROMPT_INJECTION = "indirect_prompt_injection"
    SYSTEM_PROMPT_EXTRACTION = "system_prompt_extraction"
    SENSITIVE_DATA_EXTRACTION = "sensitive_data_extraction"
    UNAUTHORIZED_TOOL_CALL = "unauthorized_tool_call"


class SecurityTarget(str, Enum):
    UNIFIED = "unified"
    ROUTER = "router"
    TOOL = "tool"
    RAG = "rag"
    CHAT = "chat"


class ExpectedSecurityBehavior(str, Enum):
    REFUSE = "refuse"
    IGNORE_ATTACK_AND_ANSWER = "ignore_attack_and_answer"
    NO_SENSITIVE_DISCLOSURE = "no_sensitive_disclosure"
    NO_UNAUTHORIZED_TOOL_CALL = "no_unauthorized_tool_call"


class SecurityCase(BaseModel):
    case_id: str = Field(min_length=1)
    attack_type: SecurityAttackType
    target: SecurityTarget
    query: str = Field(min_length=1)
    expected_behavior: ExpectedSecurityBehavior
    forbidden_phrases: list[str] = Field(default_factory=list)
    injected_content: str | None = None
    required_phrases: list[str] = Field(default_factory=list)
    allowed_tool_names: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_case_semantics(self) -> Self:
        if (self.attack_type is SecurityAttackType.INDIRECT_PROMPT_INJECTION
            and (self.injected_content is None or not self.injected_content.strip())):
            raise ValueError("间接提示词注入必须提供injected_content")

        if (self.expected_behavior is ExpectedSecurityBehavior.IGNORE_ATTACK_AND_ANSWER
            and not self.required_phrases):
            raise ValueError("继续回答用例必须提供required_phrases")

        return self


def load_security_cases(path: str | Path) -> list[SecurityCase]:
    file_path = Path(path)

    if not file_path.exists() or not file_path.is_file():
        raise ValueError("安全评测文件不存在")

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
                case = SecurityCase.model_validate(data)
            except ValidationError as error:
                raise ValueError(f"第{line_number}行字段不合法") from error

            if case.case_id in seen_case_ids:
                raise ValueError(f"发现重复case_id：{case.case_id}")

            seen_case_ids.add(case.case_id)
            cases.append(case)

    if not cases:
        raise ValueError("安全评测文件不能为空")
    return cases