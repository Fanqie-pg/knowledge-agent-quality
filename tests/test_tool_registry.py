from app.tool_registry import build_tools


def test_build_tools_registers_expected_tools():
    tools = build_tools()
    tools_names = {tool.name for tool in tools}
    assert tools_names == {"multiply_numbers", "get_leave_balance"}


def test_multiply_tool_has_expected_schema():
    tools_by_name = {tool.name: tool for tool in build_tools()}
    multiply_tool = tools_by_name["multiply_numbers"]
    schema = multiply_tool.args_schema.model_json_schema()

    assert multiply_tool.description == "用于计算两个数的乘积"
    assert set(schema["properties"]) == {"a", "b"}
    assert set(schema["required"]) == {"a", "b"}
    assert schema["properties"]["a"]["type"] == "number"
    assert schema["properties"]["b"]["type"] == "number"


def test_leave_balance_tool_has_expected_schema():
    tools_by_name = {tool.name: tool for tool in build_tools()}
    leave_balance_tool = tools_by_name["get_leave_balance"]
    schema = leave_balance_tool.args_schema.model_json_schema()

    assert leave_balance_tool.description == "根据员工编号查询剩余年假天数"
    assert set(schema["properties"].keys()) == {"employee_id"}
    assert set(schema["required"]) == {"employee_id"}
    assert schema["properties"]["employee_id"]["type"] == "string"


def test_registered_tools_invoke_underlying_functions():
    tools_by_name = {tool.name: tool for tool in build_tools()}
    multiply_result = tools_by_name["multiply_numbers"].invoke(
        {"a": 12, "b": 8}
    )
    leave_result = tools_by_name["get_leave_balance"].invoke(
        {"employee_id": "E001"}
    )
    assert multiply_result == 96.0
    assert leave_result == 5