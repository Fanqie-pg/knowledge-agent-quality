"""Immutable identities supplied by a trusted local host, never by model arguments."""
from dataclasses import dataclass, field
from enum import Enum

from app.business_errors import AuthorizationError
from app.business_audit import business_event as emit


class Role(str, Enum):
    EMPLOYEE = "employee"
    HR = "hr"


@dataclass(frozen=True)
class UserContext:
    principal_id: str
    employee_id: str
    role: Role
    employee_scope: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if (not isinstance(self.principal_id, str) or not self.principal_id.strip()
                or not isinstance(self.employee_id, str) or not self.employee_id.strip()
                or not isinstance(self.role, Role) or not isinstance(self.employee_scope, frozenset)
                or any(not isinstance(value, str) or not value for value in self.employee_scope)):
            raise ValueError("无效可信用户上下文")


PROFILES = {
    "employee-e001": UserContext("employee-e001", "E001", Role.EMPLOYEE),
    "employee-e002": UserContext("employee-e002", "E002", Role.EMPLOYEE),
    "hr": UserContext("hr", "E004", Role.HR, frozenset({"E001", "E002", "E003"})),
    # Explicit fixture for legacy unknown-employee checks, not an implicit admin.
    "hr-test": UserContext("hr-test", "E004", Role.HR, frozenset({"E001", "E002", "E003", "E999"})),
}


def load_user_context(profile: str | None) -> UserContext | None:
    if not profile:
        return None
    if profile not in PROFILES:
        raise ValueError("未知本地身份配置")
    return PROFILES[profile]


def require_identity(context: UserContext | None, operation: str) -> UserContext:
    if context is None:
        emit("authorization", "denied", operation=operation, code="AUTHENTICATION_REQUIRED")
        raise AuthorizationError("AUTHENTICATION_REQUIRED", "该操作需要可信登录身份")
    if not isinstance(context, UserContext):
        raise AuthorizationError("AUTHENTICATION_REQUIRED", "该操作需要可信登录身份")
    return context


def authorize_employee(context: UserContext | None, employee_id: str) -> UserContext:
    context = require_identity(context, "leave.read")
    permitted = employee_id == context.employee_id or (context.role is Role.HR and employee_id in context.employee_scope)
    emit("authorization", "allowed" if permitted else "denied", operation="leave.read", principal_id=context.principal_id,
         resource=employee_id, code="ALLOWED" if permitted else "ACCESS_DENIED")
    if not permitted:
        raise AuthorizationError("ACCESS_DENIED", "无权访问该员工数据")
    return context
