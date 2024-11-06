import argparse
import json

import pandas as pd
# import pytest
import requests
from tqdm import tqdm
from typing import List


# Base URL for the local API
base_url = "http://localhost:8000"


def is_subsequence(expected, actual):
    """
    Check if the expected sequence is a subsequence of the actual sequence.

    Args:
        expected (list): The expected sequence of items.
        actual (list): The actual sequence of items.

    Returns:
        bool: True if the expected sequence is a subsequence of the actual sequence, False otherwise.
    """
    it = iter(actual)
    # Check if all items in expected are in it and in the correct order
    return all(item in it for item in expected) and all(
        item in expected for item in actual
    )

def validate_tool(required_tools:List, 
                  actual_tool_calls:List, 
                  optional_tools:List, 
                  forbidden_tools:List):
    # Check for forbidden tools
    for tool in actual_tool_calls:
        if tool in forbidden_tools:
            return False, f"Forbidden tool called: {tool}"

    # Validate required tools order
    order = 0
    for tool in actual_tool_calls:
        if tool in required_tools[order:]:
            if tool == required_tools[order]:
                order += 1
            continue
        elif tool in required_tools[:order]:
            continue
        elif tool in optional_tools:
            continue
        else:
            return False, f"Unexpected tool called: {tool}"
    
    # Check if all required tools were called
    if order != len(required_tools):
        return False, "Not all required tools were called"
    
    return True, "All required tools called correctly"

# @pytest.mark.skip(reason="Skipping this test by default unless provoked")
def test_tool_calls(output_file="tool_call_evaluation.csv"):
    """
    Test the tool calls by sending requests to the API and comparing the actual
    tool calls with the expected tool calls.

    Args:
        output_file (str): The name of the file to which the evaluation results
                           will be written. Defaults to "tool_call_evaluation.csv".

    This function reads test cases from a JSON file, sends each test case as a
    request to the API, and checks if the sequence of tool calls in the response
    matches the expected sequence. The results are stored in a list and can be
    written to a CSV file for further analysis.
    """
    # Load the expected tool calls from the JSON file
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
            match, _ = validate_tool(expected_tool_names, 
                                  called_tool_names, 
                                  optional_tools=optional_tools,
                                  forbidden_tools=forbidden_tools)

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


def main():
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
    main()
