"""The common authorization and validation boundary for local and MCP tools."""
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
import math
import sqlite3
import time
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from app.business_errors import AuthorizationError, BusinessError
from app.identity import UserContext, authorize_employee, require_identity
from app.tool_permissions import ensure_operation
from app.business_audit import business_event as emit

_request_key = ContextVar("business_request_key", default=None)


@contextmanager
def request_scope(key):
    token = _request_key.set(key)
    try:
        yield
    finally:
        _request_key.reset(token)


def current_request_key():
    return _request_key.get()


class StrictArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Operation(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


Number = StrictInt | StrictFloat


class CalculateArguments(StrictArguments):
    operation: Operation
    a: Number
    b: Number


class MultiplyArguments(StrictArguments):
    a: Number
    b: Number


class LeaveArguments(StrictArguments):
    employee_id: Annotated[str, Field(strict=True, min_length=1, max_length=64)]


class CreateTicketArguments(StrictArguments):
    title: Annotated[str, Field(strict=True, min_length=5, max_length=100)]
    description: Annotated[str, Field(strict=True, min_length=1, max_length=2000)]
    priority: Annotated[str, Field(strict=True, pattern=r"^P[123]$")]


class GetTicketArguments(StrictArguments):
    ticket_id: Annotated[str, Field(strict=True, min_length=1, max_length=80)]


def calculate(operation, a, b):
    arguments = CalculateArguments(operation=operation, a=a, b=b)
    try:
        if not math.isfinite(arguments.a) or not math.isfinite(arguments.b):
            raise BusinessError("INVALID_ARGUMENTS", "计算参数必须为有限数值")
        if arguments.operation is Operation.DIVIDE and arguments.b == 0:
            raise BusinessError("DIVISION_BY_ZERO", "除数不能为零")
        functions = {Operation.ADD: lambda: arguments.a + arguments.b, Operation.SUBTRACT: lambda: arguments.a - arguments.b,
                     Operation.MULTIPLY: lambda: arguments.a * arguments.b, Operation.DIVIDE: lambda: arguments.a / arguments.b}
        result = float(functions[arguments.operation]())
        if not math.isfinite(result):
            raise OverflowError
        return result
    except OverflowError as exc:
        raise BusinessError("NUMERIC_OVERFLOW", "计算结果超出有限数值范围") from exc


class BusinessService:
    def __init__(self, repository, user_context: UserContext | None, *, clock=time.time):
        self.repository, self.user_context, self.clock = repository, user_context, clock

    def _database(self, function, *args):
        try:
            return function(*args)
        except sqlite3.Error as exc:
            emit("business", "database_error", error_type=type(exc).__name__)
            raise BusinessError("DATABASE_UNAVAILABLE", "业务数据服务暂时不可用") from exc

    def get_leave_balance(self, employee_id):
        employee_id = LeaveArguments(employee_id=employee_id).employee_id.upper()
        ensure_operation(self.user_context, "leave.read")
        context = authorize_employee(self.user_context, employee_id)
        result = self._database(self.repository.get_leave_balance, employee_id)
        emit("business", "protected_read", operation="leave.read", principal_id=context.principal_id, resource=employee_id)
        return result

    def create_ticket(self, title, description, priority, *, request_key=None):
        ensure_operation(self.user_context, "ticket.create")
        context = require_identity(self.user_context, "ticket.create")
        payload = CreateTicketArguments(title=title, description=description, priority=priority).model_dump()
        key = request_key if request_key is not None else current_request_key()
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise BusinessError("REQUEST_CONTEXT_REQUIRED", "创建工单需要宿主提供有效的请求凭据")
        emit("authorization", "allowed", operation="ticket.create", principal_id=context.principal_id, resource=context.employee_id)
        result = self._database(self.repository.create_ticket, context.employee_id, payload, key, self.clock())
        emit("business", "committed_write" if not result["idempotent_replay"] else "idempotent_replay",
             operation="ticket.create", ticket_id=result["ticket_id"], request_key=key, principal_id=context.principal_id)
        return result

    def get_ticket(self, ticket_id):
        ensure_operation(self.user_context, "ticket.read")
        context = require_identity(self.user_context, "ticket.read")
        ticket_id = GetTicketArguments(ticket_id=ticket_id).ticket_id
        owner = self._database(self.repository.ticket_owner, ticket_id)
        if owner != context.employee_id:
            reason = "TICKET_NOT_FOUND" if owner is None else "TICKET_SCOPE_DENIED"
            emit("authorization", "denied", operation="ticket.read", principal_id=context.principal_id, code=reason)
            raise AuthorizationError("RESOURCE_NOT_ACCESSIBLE", "工单不存在或无权访问", internal_reason=reason)
        emit("authorization", "allowed", operation="ticket.read", principal_id=context.principal_id, resource=ticket_id)
        result = self._database(self.repository.get_own_ticket, ticket_id, context.employee_id)
        if result is None:
            raise AuthorizationError("RESOURCE_NOT_ACCESSIBLE", "工单不存在或无权访问")
        emit("business", "protected_read", operation="ticket.read", principal_id=context.principal_id, resource=ticket_id)
        return result
