from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.tools import get_leave_balance as query_leave_balance
from app.tools import multiply_numbers as calculate_product

mcp = MCPServer(
    name="knowledge-agent-tools",
    instructions="提供企业知识助手所需的业务查询和计算工具。",
)


@mcp.tool(name="multiply_numbers")
def multiply_numbers_tool(a: float, b: float) -> float:
    """计算两个数的乘积。"""
    return calculate_product(a, b)


@mcp.tool(name="get_leave_balance")
def get_leave_balance_tool(employee_id: str) -> int:
    """根据员工编号查询剩余年假天数。"""
    try:
        return query_leave_balance(employee_id)
    except (KeyError, ValueError) as error:
        raise ToolError(str(error.args[0])) from error


if __name__ == "__main__":
    mcp.run("stdio")