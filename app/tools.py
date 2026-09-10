"""Direct entry points enforce the same validation and authorization as tools."""
from app.business_repository import SQLiteBusinessRepository
from app.business_service import BusinessService, calculate


def multiply_numbers(a: float, b: float) -> float:
    """用于计算两个数的乘积"""
    return calculate("multiply", a, b)


def get_leave_balance(employee_id: str, *, user_context=None, repository=None) -> int:
    """根据员工编号查询剩余年假天数；调用者必须提供可信上下文。"""
    return BusinessService(repository or SQLiteBusinessRepository(), user_context).get_leave_balance(employee_id)
