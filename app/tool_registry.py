"""Model schemas contain business arguments, never trusted identity or request keys."""
from langchain_core.tools import StructuredTool
from app.tool_permissions import ensure_operation, identity_binding

from app.business_repository import SQLiteBusinessRepository
from app.business_service import (BusinessService, CalculateArguments, CreateTicketArguments, GetTicketArguments,
                                  LeaveArguments, MultiplyArguments, calculate as perform_calculation)


def build_tools(*, user_context=None, repository=None, profile="current", request_key=None) -> list[StructuredTool]:
    if profile not in {"current", "legacy"}:
        raise ValueError("未知工具配置")
    service = BusinessService(repository or SQLiteBusinessRepository(), user_context)

    def get_leave_balance(employee_id: str) -> int:
        """根据员工编号查询剩余年假天数"""
        return service.get_leave_balance(employee_id)

    def calculate(operation: str, a: float, b: float) -> float:
        """执行add加法、subtract减法、multiply乘法或divide除法。"""
        ensure_operation(user_context, "calculate")
        return perform_calculation(operation, a, b)

    def multiply_numbers(a: float, b: float) -> float:
        """用于计算两个数的乘积"""
        ensure_operation(user_context, "calculate")
        return perform_calculation("multiply", a, b)

    def create_ticket(title: str, description: str, priority: str) -> dict:
        """以当前用户身份创建工单；priority只允许P1、P2、P3。"""
        return service.create_ticket(title, description, priority, request_key=request_key)

    def get_ticket(ticket_id: str) -> dict:
        """查询当前用户本人创建的工单详情。"""
        return service.get_ticket(ticket_id)

    functions = [(get_leave_balance, LeaveArguments, True, False)]
    if profile == "legacy":
        functions.append((multiply_numbers, MultiplyArguments, False, False))
    else:
        functions += [(calculate, CalculateArguments, False, False), (create_ticket, CreateTicketArguments, True, True),
                      (get_ticket, GetTicketArguments, True, False)]
    return [StructuredTool.from_function(func=function, args_schema=schema,
                                        metadata={"source": "local", "requires_identity": sensitive, "side_effects": writes,
                                                  "identity_binding": identity_binding(user_context)})
            for function, schema, sensitive, writes in functions]
