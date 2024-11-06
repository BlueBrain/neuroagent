import unittest
from neuroagent.scripts.validate_tool_calls import validate_tool
import pytest
class TestValidateTool(unittest.TestCase):

    def test_no_tools_called(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=[],
            optional_tools=[],
            forbidden_tools=[]
        )
        self.assertFalse(result)
        self.assertEqual(message, "Not all required tools were called")

    def test_all_required_tools_called_in_order(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=["tool1", "tool2"],
            optional_tools=[],
            forbidden_tools=[]
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_required_tools_called_out_of_order(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=["tool2", "tool1"],
            optional_tools=[],
            forbidden_tools=[]
        )
        self.assertFalse(result)

    def test_forbidden_tool_called(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool3"],
            optional_tools=[],
            forbidden_tools=["tool3"]
        )
        self.assertFalse(result)
        self.assertEqual(message, "Forbidden tool called: tool3")

    def test_optional_tools_called(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool2"],
            optional_tools=["tool2"],
            forbidden_tools=[]
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_unexpected_tool_called(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool3"],
            optional_tools=[],
            forbidden_tools=[]
        )
        self.assertFalse(result)
        self.assertEqual(message, "Unexpected tool called: tool3")

    def test_all_required_tools_called_with_optional_and_forbidden(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=["tool1", "tool2", "tool3"],
            optional_tools=["tool3"],
            forbidden_tools=["tool4"]
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")


if __name__ == '__main__':
    unittest.main()