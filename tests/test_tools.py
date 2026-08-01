import pytest

from app.tools import multiply_numbers, get_leave_balance

@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 3, 3.0),
        (12, 8, 96.0),
        (3.5, 7, 24.5),
        (0.0, 101, 0.0),
        (-2.5, 3.6, -9.0),
    ],
)
def test_multiply_numbers(a, b, expected):
    result = multiply_numbers(a, b)
    assert result == pytest.approx(expected)
    assert isinstance(result, float)

@pytest.mark.parametrize(
        "employee_id, expected",
        [
            ("E001", 5),
            (" e002 ", 0),
            ("e003", 12),
        ],
)
def test_get_leave_balance_returns_days(employee_id: str, expected: int) -> None:
    days = get_leave_balance(employee_id)
    assert days == expected
    assert isinstance(days, int)

@pytest.mark.parametrize(
        "employee_id",
        [
            None,
            "",
            " ",
            "\t",
            "\n",
            "\r",
        ],
)
def test_get_leave_balance_rejects_empty_employee_id(employee_id: str | None) -> None:
    with pytest.raises(ValueError, match="employee_id不能为空"):
        get_leave_balance(employee_id)

@pytest.mark.parametrize(
        "employee_id",
        [
            "E000",
            "E999",
        ]
)
def test_get_leave_balance_rejects_unknown_employee(employee_id: str)-> None:
    with pytest.raises(KeyError, match="员工不存在"):
        get_leave_balance(employee_id)