import requests
import json
import pandas as pd
from tqdm import tqdm

# Base URL for the local API
base_url = "http://localhost:8000"

def test_tool_calls():
    # Load the expected tool calls from the JSON file
    with open('tests/data/tool_calls.json') as f:
        tool_calls_data = json.load(f)

    # List to store results
    results_list = []

    # Iterate over each test case with a progress bar
    for test_case in tqdm(tool_calls_data, desc="Processing test cases"):
        prompt = test_case["prompt"]
        expected_tool_calls = test_case["expected_tool_calls"]
        
        print(f"Testing prompt: {prompt}")  # Verbose output

        # Send a request to the API
        response = requests.post(
            f"{base_url}/qa/run",  # Replace with the actual endpoint
            headers={"Content-Type": "application/json"},  # Ensure the correct header is set
            json={
                "query": prompt,  # Add the 'query' field with the prompt as its value
                "messages": [{"role": "user", "content": prompt}]
            }
        )

        # Check if the response is successful
        if response.status_code == 200:
            # Parse the response
            steps = response.json().get("steps", [])
            called_tool_names = [step.get("tool_name", None) for step in steps]
            expected_tool_names = [tool_call.get("tool_name", None) for tool_call in expected_tool_calls]
            match = called_tool_names == expected_tool_names

            # Append the result to the list
            results_list.append({
                "Prompt": prompt,
                "Actual": called_tool_names,
                "Expected": expected_tool_names,
                "Match": "Yes" if match else "No"
            })
        else:
            # Log the response status code and content for debugging
            error_info = {
                "status_code": response.status_code,
                "response_content": response.text
            }
            
            print(f"API call failed for prompt: {prompt} with error: {error_info}")  # Verbose output

            # Handle the case where the API call fails
            results_list.append({
                "Prompt": prompt,
                "Actual": f"API call failed: {error_info}",
                "Expected": expected_tool_calls,
                "Match": "No"
            })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results_df.to_csv('tool_call_evaluation.csv')