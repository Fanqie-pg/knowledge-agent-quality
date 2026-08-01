import pytest

from app.tools import multiply_numbers


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
