LEAVE_BALANCES = {
    "E001": 5,
    "E002": 0,
    "E003": 12,
    "E004": 3,
}


def multiply_numbers(a: float, b: float) -> float:
    """用于计算两个数的乘积"""
    return float(a * b)

def get_leave_balance(employee_id: str) -> int:
    """根据员工编号查询剩余年假天数"""
    if not employee_id or not employee_id.strip():
        raise ValueError("employee_id不能为空")
    normalized_employee_id = employee_id.strip().upper()
    if normalized_employee_id not in LEAVE_BALANCES:
        raise KeyError("员工不存在")
    return LEAVE_BALANCES[normalized_employee_id]
