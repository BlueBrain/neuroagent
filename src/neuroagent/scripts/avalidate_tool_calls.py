"""Run validation on tool calls."""

import argparse
import asyncio
import json
import logging
from typing import Any

import aiohttp
import pandas as pd

logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format
)


async def fetch_tool_call(
    session: aiohttp.ClientSession,
    query: dict[str, Any],
    base_url: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """
    Fetch the tool call results for a given test case.

    This function sends an asynchronous POST request to the API with the provided
    test case data and validates the tool calls against expected outcomes.

    Args:
    ----
    session (aiohttp.ClientSession): The aiohttp session used to make the HTTP request.
    query (dict): A dictionary containing the test case data, including the prompt,
                      expected tools, optional tools, and forbidden tools.
    base_url (str): The base URL of the API.

    Returns
    -------
    dict: A dictionary containing the prompt, actual tool calls, expected tool calls,
          and whether the actual calls match the expected ones.
    """
    async with semaphore:
        prompt = query["prompt"]
        expected_tool_calls = query["expected_tools"]
        optional_tools = query["optional_tools"]
        forbidden_tools = query["forbidden_tools"]

        logging.info(f"Testing prompt: {prompt}")

        # Send a request to the API
        async with session.post(
            f"{base_url}/qa/run",
            headers={"Content-Type": "application/json"},
            json={
                "query": prompt,
            },
        ) as response:
            if response.status == 200:
                steps = await response.json()
                called_tool_names = [
                    step.get("tool_name", None) for step in steps.get("steps", [])
                ]
                expected_tool_names = [
                    tool_call.get("tool_name", None)
                    for tool_call in expected_tool_calls
                ]
                match, reason = validate_tool(
                    expected_tool_names,
                    called_tool_names,
                    optional_tools=optional_tools,
                    forbidden_tools=forbidden_tools,
                )
                return {
                    "Prompt": prompt,
                    "Actual": called_tool_names,
                    "Expected": expected_tool_names,
                    "Optional": optional_tools,
                    "Forbidden": forbidden_tools,
                    "Match": "Yes" if match else "No",
                    "Reason": reason if not match else "N/A",
                }
            else:
                # Attempt to parse the error message from the response content
                try:
                    error_content = await response.json()
                    error_message = error_content.get("content", "Unknown error")
                except Exception as e:
                    error_message = f"Failed to parse error message: {str(e)}"

                error_info = {
                    "status_code": response.status,
                    "response_content": error_message,
                }
                logging.error(
                    f"API call failed for prompt: {prompt} with error: {error_info}"
                )
                return {
                    "Prompt": prompt,
                    "Actual": f"API call failed: {error_info}",
                    "Expected": expected_tool_calls,
                    "Optional": optional_tools,
                    "Forbidden": forbidden_tools,
                    "Match": "No",
                    "Reason": f"API call failed: {error_info}",
                }


async def validate_tool_calls_async(
    base_url: str,
    data_file: str,
    output_file: str = "tool_call_evaluation.csv",
    max_concurrent_requests: int = 10,
) -> None:
    """
    Run asynchronous tool call tests and save the results to a CSV file.

    Args:
    ----
    base_url (str): The base URL of the API.
    data_file (str): The path to the JSON file containing test case data.
    output_file (str): The name of the output CSV file where the results will
                       be saved. Defaults to 'tool_call_evaluation.csv'.
    max_concurrent_requests (int): Maximum number of concurrent API requests.
                                   Defaults to 10.

    Returns
    -------
    None: This function does not return any value. It writes the results to a
          CSV file.
    """
    with open(data_file) as f:
        tool_calls_data = json.load(f)

    results_list = []
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_tool_call(session, query, base_url, semaphore)
            for query in tool_calls_data
        ]
        results_list = await asyncio.gather(*tasks)

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(output_file, index=False)


def validate_tool(
    required_tools: list[str],
    actual_tool_calls: list[str],
    optional_tools: list[str],
    forbidden_tools: list[str],
) -> tuple[bool, str]:
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
    if inter := set(actual_tool_calls) & set(forbidden_tools):
        return False, f"Forbidden tool(s) called: {inter}"

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
    specifically for specifying the base URL, port, data file path, and output
    CSV file name. It then calls the validate_tool_calls_async function with
    the provided arguments to perform the validation of tool calls and save
    the results.

    The function is designed to be the entry point when the script is run
    directly from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run tool call tests and save results."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost",
        help="Base URL for the API",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the API",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="tests/data/tool_calls.json",
        help="Path to the JSON file containing test case data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tool_call_evaluation.csv",
        help="Output CSV file for results",
    )
    args = parser.parse_args()

    # Construct the full base URL
    full_url = f"{args.base_url}:{args.port}"
    asyncio.run(validate_tool_calls_async(full_url, args.data, args.output))


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
