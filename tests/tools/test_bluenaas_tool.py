"""Tests BlueNaaS tool."""

from typing import Literal

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from neuroagent.tools import BlueNaaSTool
from neuroagent.tools.bluenaas_tool import (
    BlueNaaSInvalidatedOutput,
    BlueNaaSValidatedOutput,
    RecordingLocation,
)


def hil_usecases(
    tool_name,
    use_case: Literal["first_encounter", "approve/modify", "refuse", "change_topic"],
):
    generic_human = HumanMessage(
        content="run a simulation on a me model",
        additional_kwargs={},
        response_metadata={},
        id="02cf3adf-df10-45f3-af0d-05e7155a520c",
    )
    generic_ai_toolcall = AIMessage(
        content="",
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call_TxjrqoPaRNnFmiZwmb9iMkgM",
                    "function": {
                        "arguments": '{"me_model_id":"https://great_memodel.com"}',
                        "name": tool_name,
                    },
                    "type": "function",
                }
            ],
            "refusal": None,
        },
        response_metadata={
            "token_usage": {
                "completion_tokens": 166,
                "prompt_tokens": 4484,
                "total_tokens": 4650,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 3456},
            },
            "model_name": "gpt-4o-mini-2024-07-18",
            "system_fingerprint": "fp_e2bde53e6e",
            "finish_reason": "tool_calls",
            "logprobs": None,
        },
        id="run-5c0c50f3-668f-41df-b69d-8497bb7dc92e-0",
        tool_calls=[
            {
                "name": "bluenaas-tool",
                "args": {"me_model_id": "https://great_memodel"},
                "id": "call_TxjrqoPaRNnFmiZwmb9iMkgM",
                "type": "tool_call",
            }
        ],
        usage_metadata={
            "input_tokens": 4484,
            "output_tokens": 166,
            "total_tokens": 4650,
        },
    )
    messages = [generic_human, generic_ai_toolcall]
    if use_case == "first_encounter":
        return messages

    generic_ai_content = AIMessage(
        content="The simulation parameters for the selected ME model...",
        additional_kwargs={"refusal": None},
        response_metadata={
            "token_usage": {
                "completion_tokens": 177,
                "prompt_tokens": 4819,
                "total_tokens": 4996,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 4736},
            },
            "model_name": "gpt-4o-mini-2024-07-18",
            "system_fingerprint": "fp_e2bde53e6e",
            "finish_reason": "stop",
            "logprobs": None,
        },
        id="run-6e963cac-2392-483a-8a4c-0ac385130aba-0",
        usage_metadata={
            "input_tokens": 4819,
            "output_tokens": 177,
            "total_tokens": 4996,
        },
    )
    messages.extend(
        [
            ToolMessage(
                content="A simulation will be ran with the following inputs <json>{'currentInjection': {'injectTo': 'soma[0]', 'stimulus': {'stimulusType': 'current_clamp', 'stimulusProtocol': 'ap_waveform', 'amplitudes': [0.1]}}, 'recordFrom': [{'section': 'soma[0]', 'offset': 0.5}], 'conditions': {'celsius': 34, 'vinit': -73, 'hypamp': 0, 'max_time': 100, 'time_step': 0.05, 'seed': 100}, 'type': 'single-neuron-simulation', 'simulationDuration': 100}</json>. \n Please confirm that you are satisfied by the simulation parameters, or correct them accordingly.",
                name="bluenaas-tool",
                id="95a70cb3-8afc-4e51-8868-29ea9cf5c8bd",
                tool_call_id="call_IScHmOF8TsJEjqmZ9yL5ehRY",
                artifact={"is_validated": True},
            ),
            generic_ai_content,
        ]
    )
    if use_case == "approve/modify":
        # To modify, the tool call invocation needs to have a different signature
        messages.extend([generic_human, generic_ai_toolcall])
        return messages
    if use_case == "refuse":
        messages.extend(
            [generic_human, generic_ai_content, generic_human, generic_ai_toolcall]
        )
        return messages
    if use_case == "change_topic":
        messages.extend(
            [
                generic_human,
                AIMessage(
                    content="",
                    additional_kwargs={
                        "tool_calls": [
                            {
                                "id": "call_TxjrqoPaRNnFmiZwmb9iMkgM",
                                "function": {
                                    "arguments": '{"me_model_id":"https://great_memodel.com"}',
                                    "name": "other-tool",
                                },
                                "type": "function",
                            }
                        ],
                        "refusal": None,
                    },
                ),
                ToolMessage(
                    content="great content",
                    name="other-tool",
                    id="95a70cb3-8afc-4e51-8868-29ea9cf5c8bd",
                    tool_call_id="call_IScHmOF8TsJEjqmZ9yL5ehRY",
                    artifact={"is_validated": True},
                ),
                generic_ai_content,
                generic_human,
                generic_ai_toolcall,
            ]
        )
        return messages


@pytest.mark.asyncio
async def test_arun(httpx_mock):
    me_model_id = "great_id"
    url = "http://fake_url"

    httpx_mock.add_response(
        url=url + f"?model_id={me_model_id}",
        json={"t": [0.05, 0.1, 0.15, 0.2], "v": [-1.14, -0.67, -1.78]},
    )

    tool = BlueNaaSTool(
        metadata={
            "url": url,
            "httpx_client": httpx.AsyncClient(),
            "token": "fake_token",
        }
    )

    messages = hil_usecases("bluenaas-tool", "first_encounter")
    # First call to bluenaas. Need to validate
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages,
    )
    assert isinstance(response[0], BlueNaaSInvalidatedOutput)
    assert response[0].inputs == tool.create_json_api()
    assert response[1] == {"is_validated": True}

    messages_approve = hil_usecases("bluenaas-tool", "approve/modify")
    # Case where we call bluenaas after validating. Run simu
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages_approve,
    )
    assert isinstance(response[0], BlueNaaSValidatedOutput)
    assert response[0] == BlueNaaSValidatedOutput(status="success")
    assert response[1] == {"is_validated": False}

    messages_change_topic = hil_usecases("bluenaas-tool", "change_topic")

    # Don't validate but completely change topic
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages_change_topic,
    )
    assert isinstance(response[0], BlueNaaSInvalidatedOutput)
    assert response[1] == {"is_validated": True}

    messages_refusal = hil_usecases("bluenaas-tool", "refuse")

    # Don't validate but ask again to run
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages_refusal,
    )

    assert isinstance(response[0], BlueNaaSInvalidatedOutput)
    assert response[1] == {"is_validated": True}

    # Modify the input. Should ask for validation again
    response = await tool._arun(
        me_model_id=me_model_id, messages=messages_approve, conditions__celsius=40
    )

    assert isinstance(response[0], BlueNaaSInvalidatedOutput)
    assert response[1] == {"is_validated": True}


@pytest.mark.parametrize(
    "use_case,output",
    [
        ("first_encounter", False),
        ("approve/modify", True),
        ("refuse", False),
        ("change_topic", False),
    ],
)
def test_is_validated(use_case, output):
    messages = hil_usecases("bluenaas-tool", use_case)

    # Same json as in the messages
    json_api = BlueNaaSTool.create_json_api()
    assert BlueNaaSTool.is_validated(messages, json_api) == output

    # Different json as in the messages
    json_api = BlueNaaSTool.create_json_api(conditions__celsius=25)
    assert not BlueNaaSTool.is_validated(messages, json_api)


def test_create_json_api():
    url = "http://fake_url"

    tool = BlueNaaSTool(
        metadata={
            "url": url,
            "httpx_client": httpx.AsyncClient(),
            "token": "fake_token",
        }
    )

    json_api = tool.create_json_api(
        conditions__vinit=-3,
        current_injection__stimulus__stimulus_type="conductance",
        record_from=[
            RecordingLocation(),
            RecordingLocation(section="axon[78]", offset=0.1),
        ],
        current_injection__stimulus__amplitudes=[0.1, 0.5],
    )
    assert json_api == {
        "currentInjection": {
            "injectTo": "soma[0]",
            "stimulus": {
                "stimulusType": "conductance",
                "stimulusProtocol": "ap_waveform",
                "amplitudes": [0.1, 0.5],
            },
        },
        "recordFrom": [
            {"section": "soma[0]", "offset": 0.5},
            {"section": "axon[78]", "offset": 0.1},
        ],
        "conditions": {
            "celsius": 34,
            "vinit": -3,
            "hypamp": 0,
            "max_time": 100,
            "time_step": 0.05,
            "seed": 100,
        },
        "type": "single-neuron-simulation",
        "simulationDuration": 100,
    }
