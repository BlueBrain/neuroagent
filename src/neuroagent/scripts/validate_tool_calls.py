"""Run validation on tool calls."""

import argparse
import json
from typing import List, Tuple

import pandas as pd

# import pytest
import requests
from tqdm import tqdm

# Base URL for the local API
base_url = "http://localhost:8000"


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
        elif tool in optional_tools or (order > 0 and tool == required_tools[order - 1]):
            continue
        elif tool not in required_tools[:order]:
            return False, f"Unexpected tool called: {tool}"

    # Check if all required tools were called
    if order != len(required_tools):
        return False, "Not all required tools were called"

    return True, "All required tools called correctly"


def test_tool_calls(output_file: str = "tool_call_evaluation.csv") -> None:
    """Test the tool calls by sending requests to the API and comparing the actualtool calls with the expected tool calls.

    Args:
    ----
        output_file (str): The name of the file to which the evaluation results
                           will be written. Defaults to "tool_call_evaluation.csv".

    This function reads test cases from a JSON file, sends each test case as a
    request to the API, and checks if the sequence of tool calls in the response
    matches the expected sequence. The results are stored in a list and can be
    written to a CSV file for further analysis.
    """
    with open("tests/data/tool_calls.json") as f:
        tool_calls_data = json.load(f)

    # List to store results
    results_list = []

    # Iterate over each test case with a progress bar
    for test_case in tqdm(tool_calls_data, desc="Processing test cases"):
        prompt = test_case["prompt"]
        expected_tool_calls = test_case["expected_tools"]
        optional_tools = test_case["optional_tools"]
        forbidden_tools = test_case["forbidden_tools"]

        print(f"Testing prompt: {prompt}")  # Verbose output

        # Send a request to the API
        response = requests.post(
            f"{base_url}/qa/run",  # Replace with the actual endpoint
            headers={
                "Content-Type": "application/json"
            },  # Ensure the correct header is set
            json={
                "query": prompt,  # Add the 'query' field with the prompt as its value
                "messages": [{"role": "user", "content": prompt}],
            },
        )

        # Check if the response is successful
        if response.status_code == 200:
            # Parse the response
            steps = response.json().get("steps", [])
            called_tool_names = [step.get("tool_name", None) for step in steps]
            expected_tool_names = [
                tool_call.get("tool_name", None) for tool_call in expected_tool_calls
            ]
            match, _ = validate_tool(
                expected_tool_names,
                called_tool_names,
                optional_tools=optional_tools,
                forbidden_tools=forbidden_tools,
            )

            # Append the result to the list
            results_list.append(
                {
                    "Prompt": prompt,
                    "Actual": called_tool_names,
                    "Expected": expected_tool_names,
                    "Match": "Yes" if match else "No",
                }
            )
        else:
            # Log the response status code and content for debugging
            error_info = {
                "status_code": response.status_code,
                "response_content": response.text,
            }

            print(
                f"API call failed for prompt: {prompt} with error: {error_info}"
            )  # Verbose output

            # Handle the case where the API call fails
            results_list.append(
                {
                    "Prompt": prompt,
                    "Actual": f"API call failed: {error_info}",
                    "Expected": expected_tool_calls,
                    "Match": "No",
                }
            )

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_df.to_csv(output_file)


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

    test_tool_calls(args.output)


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
