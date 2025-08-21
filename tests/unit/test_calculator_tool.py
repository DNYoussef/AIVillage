from packages.agent_forge.phases.tool_persona_baking import ToolIntegrationSystem


def test_calculator_tool_valid_expression():
    system = ToolIntegrationSystem(None)
    result = system._calculator_tool("2 + 3 * 4")
    assert result["success"] is True
    assert result["result"] == "14"


def test_calculator_tool_malicious_expression():
    system = ToolIntegrationSystem(None)
    expr = "2 + __import__('os').system('echo hacked')"
    result = system._calculator_tool(expr)
    assert result["success"] is True
    assert result["result"] == f"Mathematical expression: {expr}"
