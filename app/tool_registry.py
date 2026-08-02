from langchain_core.tools import StructuredTool

from app.tools import get_leave_balance, multiply_numbers


def build_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(func=get_leave_balance),
        StructuredTool.from_function(func=multiply_numbers)
    ]