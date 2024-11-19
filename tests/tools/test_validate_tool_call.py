import asyncio
import json
import unittest
from unittest.mock import AsyncMock, mock_open, patch

import aiohttp
import pytest

from src.neuroagent.scripts.avalidate_tool_calls import (
    fetch_tool_call,
    validate_tool,
    validate_tool_calls_async,
)


class TestValidateTool(unittest.TestCase):
    def test_no_tools_called(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=[],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertFalse(result)
        self.assertEqual(message, "Not all required tools were called")

    def test_all_required_tools_called_in_order(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=["tool1", "tool2"],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_required_tools_called_out_of_order(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=["tool2", "tool1"],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertFalse(result)

    def test_forbidden_tool_called(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool3"],
            optional_tools=[],
            forbidden_tools=["tool3"],
        )
        self.assertFalse(result)
        self.assertEqual(message, "Forbidden tool(s) called: {'tool3'}")

    def test_optional_tools_called(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool2"],
            optional_tools=["tool2"],
            forbidden_tools=[],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_unexpected_tool_called(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool3"],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertFalse(result)
        self.assertEqual(message, "Unexpected tool called: tool3")

    def test_all_required_tools_called_with_optional_and_forbidden(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2"],
            actual_tool_calls=["tool1", "tool2", "tool3"],
            optional_tools=["tool3"],
            forbidden_tools=["tool4"],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_only_optional_tools_called(self):
        result, message = validate_tool(
            required_tools=[],
            actual_tool_calls=["tool2"],
            optional_tools=["tool2"],
            forbidden_tools=[],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_only_forbidden_tools_called(self):
        result, message = validate_tool(
            required_tools=[],
            actual_tool_calls=["tool3"],
            optional_tools=[],
            forbidden_tools=["tool3"],
        )
        self.assertFalse(result)
        self.assertEqual(message, "Forbidden tool(s) called: {'tool3'}")

    def test_mixed_tools_called(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool2", "tool3"],
            optional_tools=["tool2"],
            forbidden_tools=["tool3"],
        )
        self.assertFalse(result)
        self.assertEqual(message, "Forbidden tool(s) called: {'tool3'}")

    def test_repeated_required_tools(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool1"],
            actual_tool_calls=["tool1", "tool1"],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_repeated_forbidden_tools(self):
        result, message = validate_tool(
            required_tools=["tool1"],
            actual_tool_calls=["tool1", "tool3", "tool3"],
            optional_tools=[],
            forbidden_tools=["tool3"],
        )
        self.assertFalse(result)
        self.assertEqual(message, "Forbidden tool(s) called: {'tool3'}")

    def test_overrepeated_tools(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2", "tool3"],
            actual_tool_calls=["tool1", "tool2", "tool2", "tool2", "tool3"],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_overrepeated_tools2(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2", "tool3"],
            actual_tool_calls=["tool1", "tool2", "tool3", "tool3", "tool3"],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")

    def test_overrepeated_tools3(self):
        result, message = validate_tool(
            required_tools=["tool1", "tool2", "tool3"],
            actual_tool_calls=[
                "tool1",
                "tool1",
                "tool1",
                "tool2",
                "tool2",
                "tool3",
                "tool3",
            ],
            optional_tools=[],
            forbidden_tools=[],
        )
        self.assertTrue(result)
        self.assertEqual(message, "All required tools called correctly")


@pytest.mark.asyncio
async def test_fetch_tool_call_success():
    test_case = {
        "prompt": "Test prompt",
        "expected_tools": [{"tool_name": "tool1"}, {"tool_name": "tool2"}],
        "optional_tools": ["tool3"],
        "forbidden_tools": ["tool4"],
    }

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "steps": [{"tool_name": "tool1"}, {"tool_name": "tool2"}]
    }

    # Mock the context manager behavior
    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response

        async with aiohttp.ClientSession() as session:
            base_url = "http://localhost:8000"  # Define the base URL for testing
            semaphore = asyncio.Semaphore(1)  # Create a semaphore for testing
            result = await fetch_tool_call(
                session, test_case, base_url, semaphore
            )  # Pass semaphore

            assert result["Prompt"] == "Test prompt"
            assert result["Actual"] == ["tool1", "tool2"]
            assert result["Expected"] == ["tool1", "tool2"]
            assert result["Optional"] == ["tool3"]
            assert result["Forbidden"] == ["tool4"]
            assert result["Match"] == "Yes"


@pytest.mark.asyncio
async def test_fetch_tool_call_failure():
    test_case = {
        "prompt": "Test prompt",
        "expected_tools": [{"tool_name": "tool1"}, {"tool_name": "tool2"}],
        "optional_tools": ["tool3"],
        "forbidden_tools": ["tool4"],
    }

    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.text.return_value = "Internal Server Error"

    with patch("aiohttp.ClientSession.post") as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response

        async with aiohttp.ClientSession() as session:
            base_url = "http://localhost:8000"  # Define the base URL for testing
            semaphore = asyncio.Semaphore(1)  # Create a semaphore for testing
            result = await fetch_tool_call(
                session, test_case, base_url, semaphore
            )  # Pass semaphore

            assert result["Prompt"] == "Test prompt"
            assert "API call failed" in result["Actual"]
            assert result["Expected"] == [
                {"tool_name": "tool1"},
                {"tool_name": "tool2"},
            ]
            assert result["Optional"] == ["tool3"]
            assert result["Forbidden"] == ["tool4"]
            assert result["Match"] == "No"


@pytest.mark.asyncio
async def test_validate_tool_calls_async():
    mock_data = json.dumps(
        [
            {
                "prompt": "Test prompt",
                "expected_tools": [{"tool_name": "tool1"}],
                "optional_tools": [],
                "forbidden_tools": [],
            }
        ]
    )

    with patch("builtins.open", mock_open(read_data=mock_data)):
        with patch(
            "src.neuroagent.scripts.avalidate_tool_calls.fetch_tool_call",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = {"Match": "Yes"}
            base_url = "http://localhost:8000"  # Define the base URL for testing
            data_file = "mock_data.json"  # Mock data file path
            await validate_tool_calls_async(base_url, data_file, "test_output.csv")


if __name__ == "__main__":
    unittest.main()
