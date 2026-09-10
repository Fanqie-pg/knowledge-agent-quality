"""One trusted permission source; never derive grants from model arguments."""
from dataclasses import dataclass
import hashlib
import json

from app.business_errors import AuthorizationError
from app.identity import Role, UserContext
from app.business_audit import business_event


PERMISSION_POLICY_VERSION = "1.0-identity-derived"
UNSET_CONTEXT = object()
TOOL_OPERATIONS = {"calculate": "calculate", "multiply_numbers": "calculate", "get_leave_balance": "leave.read",
                   "create_ticket": "ticket.create", "get_ticket": "ticket.read"}
ROLE_OPERATIONS = {None: frozenset({"calculate"}),
                   Role.EMPLOYEE: frozenset(TOOL_OPERATIONS.values()),
                   Role.HR: frozenset(TOOL_OPERATIONS.values())}


def validate_context(context):
    if context is not None and not isinstance(context, UserContext):
        raise ValueError("工具权限需要可信UserContext")
    return context


def identity_binding(context):
    validate_context(context)
    value = None if context is None else {"principal": context.principal_id, "employee": context.employee_id,
                                          "role": context.role.value, "scope": sorted(context.employee_scope)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ensure_operation(context, operation):
    validate_context(context)
    if operation not in ROLE_OPERATIONS.get(context.role if context else None, frozenset()):
        code = "AUTHENTICATION_REQUIRED" if context is None else "TOOL_ACCESS_DENIED"
        business_event("authorization", "denied", operation=operation, code=code)
        raise AuthorizationError(code, "该操作需要可信登录身份" if context is None else "当前角色无权执行该操作")


@dataclass(frozen=True)
class PermissionSnapshot:
    principal_id: str | None
    employee_id: str | None
    role: str | None
    registered_tool_names: tuple[str, ...]
    allowed_tool_names: tuple[str, ...]
    leave_employee_ids: tuple[str, ...]
    legacy_request_limit: tuple[str, ...] | None = None

    def as_dict(self):
        return {"permission_policy_version": PERMISSION_POLICY_VERSION, "permission_source": "trusted_identity",
                "actor": self.principal_id, "identity": {"employee_id": self.employee_id, "role": self.role} if self.principal_id else None,
                "registered_tool_names": list(self.registered_tool_names), "effective_tool_names": list(self.allowed_tool_names),
                "leave_employee_ids": list(self.leave_employee_ids), "ticket_scope": "own_only" if self.principal_id else "none",
                "request_allowlist": None if self.legacy_request_limit is None else list(self.legacy_request_limit)}


def resolve_permissions(context, registered_tool_names, *, legacy_request_limit=None, public_tools=()):
    validate_context(context)
    registered = tuple(sorted(set(registered_tool_names)))
    public = frozenset(public_tools)
    if public.intersection(TOOL_OPERATIONS):
        raise ValueError("不能覆盖内置工具的权限映射")
    if legacy_request_limit is not None and (not isinstance(legacy_request_limit, (list, tuple))
            or any(not isinstance(name, str) for name in legacy_request_limit)):
        raise ValueError("历史工具限制必须为字符串列表")
    operations = ROLE_OPERATIONS.get(context.role if context else None, frozenset())
    allowed = [name for name in registered if TOOL_OPERATIONS.get(name) in operations or name in public]
    if legacy_request_limit is not None:
        allowed = [name for name in allowed if name in legacy_request_limit]
    scope = set()
    if context:
        scope.add(context.employee_id)
        if context.role is Role.HR:
            scope.update(context.employee_scope)
    return PermissionSnapshot(context.principal_id if context else None, context.employee_id if context else None,
                              context.role.value if context else None, registered, tuple(allowed), tuple(sorted(scope)),
                              None if legacy_request_limit is None else tuple(legacy_request_limit))
