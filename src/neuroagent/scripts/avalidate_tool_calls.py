"""Run validation on tool calls."""

import argparse
import asyncio
import json
from typing import Any, Dict, List, Tuple

import aiohttp
import pandas as pd

# import pytest

# Base URL for the local API
base_url = "http://localhost:8000"


async def fetch_tool_call(
    session: aiohttp.ClientSession, test_case: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fetch the tool call results for a given test case.

    This function sends an asynchronous POST request to the API with the provided
    test case data and validates the tool calls against expected outcomes.

    Args:
    ----
    session (aiohttp.ClientSession): The aiohttp session used to make the HTTP request.
    test_case (dict): A dictionary containing the test case data, including the prompt,
                      expected tools, optional tools, and forbidden tools.

    Returns
    -------
    dict: A dictionary containing the prompt, actual tool calls, expected tool calls,
          and whether the actual calls match the expected ones.
    """
    prompt = test_case["prompt"]
    expected_tool_calls = test_case["expected_tools"]
    optional_tools = test_case["optional_tools"]
    forbidden_tools = test_case["forbidden_tools"]

    print(f"Testing prompt: {prompt}")  # Verbose output

    # Send a request to the API
    async with session.post(
        f"{base_url}/qa/run",
        headers={"Content-Type": "application/json"},
        json={
            "query": prompt,
            "messages": [{"role": "user", "content": prompt}],
        },
    ) as response:
        if response.status == 200:
            steps = await response.json()
            called_tool_names = [
                step.get("tool_name", None) for step in steps.get("steps", [])
            ]
            expected_tool_names = [
                tool_call.get("tool_name", None) for tool_call in expected_tool_calls
            ]
            match, _ = validate_tool(
                expected_tool_names,
                called_tool_names,
                optional_tools=optional_tools,
                forbidden_tools=forbidden_tools,
            )
            return {
                "Prompt": prompt,
                "Actual": called_tool_names,
                "Expected": expected_tool_names,
                "Match": "Yes" if match else "No",
            }
        else:
            error_info = {
                "status_code": response.status,
                "response_content": await response.text(),
            }
            print(f"API call failed for prompt: {prompt} with error: {error_info}")
            return {
                "Prompt": prompt,
                "Actual": f"API call failed: {error_info}",
                "Expected": expected_tool_calls,
                "Match": "No",
            }


async def validate_tool_calls_async(
    output_file: str = "tool_call_evaluation.csv",
) -> None:
    """
    Run asynchronous tool call tests and save the results to a CSV file.

    This function reads test case data from a JSON file, performs asynchronous
    API calls to validate tool calls, and writes the results to a specified
    output CSV file.

    Args:
    ----
    output_file (str): The name of the output CSV file where the results will
                       be saved. Defaults to 'tool_call_evaluation.csv'.

    Returns
    -------
    None: This function does not return any value. It writes the results to a
          CSV file.
    """
    with open("tests/data/tool_calls.json") as f:
        tool_calls_data = json.load(f)

    results_list = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_tool_call(session, test_case) for test_case in tool_calls_data]
        results_list = await asyncio.gather(*tasks)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file)


def validate_tool(
    required_tools: List[str],
    actual_tool_calls: List[str],
    optional_tools: List[str],
    forbidden_tools: List[str],
) -> Tuple[bool, str]:
    """
    Validate the sequence of tool calls against required, optional, and forbidden tools.

    Args:
    ----
        required_tools (List): A list of tools that must be called in the specified order.
        actual_tool_calls (List): A list of tools that were actually called.
        optional_tools (List): A list of tools that can be called but are not required.
        forbidden_tools (List): A list of tools that must not be called.

    Returns
    -------
        tuple: A tuple containing a boolean and a string message. The boolean is True if the
               validation is successful, otherwise False. The string message provides details
               about the validation result.
    """
    # Check for forbidden tools
    for tool in actual_tool_calls:
        if tool in forbidden_tools:
            return False, f"Forbidden tool called: {tool}"

    # Validate required tools order
    order = 0
    for tool in actual_tool_calls:
        if order < len(required_tools) and tool == required_tools[order]:
            order += 1
        elif tool in optional_tools or (
            order > 0 and tool == required_tools[order - 1]
        ):
            continue
        elif tool not in required_tools[:order]:
            return False, f"Unexpected tool called: {tool}"

    # Check if all required tools were called
    if order != len(required_tools):
        return False, "Not all required tools were called"

    return True, "All required tools called correctly"


def main() -> None:
    """
    Execute the tool call validation process.

    This function sets up the argument parser to handle command-line arguments,
    specifically for specifying the output CSV file name. It then calls the
    test_tool_calls function with the provided output file name to perform
    the validation of tool calls and save the results.

    The function is designed to be the entry point when the script is run
    directly from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run tool call tests and save results."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tool_call_evaluation.csv",
        help="Output CSV file for results",
    )
    args = parser.parse_args()

    asyncio.run(validate_tool_calls_async(args.output))


if __name__ == "__main__":
    """
    Validate tool calls against expected outcomes and logs the results.

    The script reads a set of prompts and their expected tool calls, executes the tool calls,
    and compares the actual tool calls made with the expected ones. It logs whether the actual
    tool calls match the expected ones and saves the results to a CSV file.

    Usage:
        python validate_tool_calls.py --output <output_csv_file>

    Arguments:
        --output: The name of the output CSV file where the results will be saved.
                  Defaults to 'tool_call_evaluation.csv'.

    The script is intended to be run as a standalone module.
    """

    main()
