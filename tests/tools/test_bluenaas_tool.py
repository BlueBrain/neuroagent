"""Tests BlueNaaS tool."""

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from neuroagent.tools import BlueNaaSTool
from neuroagent.tools.bluenaas_tool import (
    BlueNaaSInvalidatedOutput,
    BlueNaaSValidatedOutput,
    RecordingLocation,
)


@pytest.mark.asyncio
async def test_arun(httpx_mock):
    me_model_id = "great_id"
    url = "http://fake_url"

    httpx_mock.add_response(
        url=url + f"?model_id={me_model_id}",
        json={"t": [0.05, 0.1, 0.15, 0.2], "v": [-1.14, -0.67, -1.78]},
    )
    # Case where we call bluenaas the first time
    messages = [
        HumanMessage(
            content="run a simulation on a me model",
            additional_kwargs={},
            response_metadata={},
            id="02cf3adf-df10-45f3-af0d-05e7155a520c",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_TxjrqoPaRNnFmiZwmb9iMkgM",
                        "function": {
                            "arguments": '{"me_model_id":"https://openbluebrain.com/api/nexus/v1/resources/3284c079-ec97-4bfe-b039-d2c2fa5d8e19/b91f3847-8c5a-45d6-b038-53780059c774/_/https:%2F%2Fopenbluebrain.com%2Fdata%2F3284c079-ec97-4bfe-b039-d2c2fa5d8e19%2Fb91f3847-8c5a-45d6-b038-53780059c774%2F169f755e-745c-481a-bed5-14ed5591a2ee"}',
                            "name": "bluenaas-tool",
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
                    "args": {
                        "me_model_id": "https://openbluebrain.com/api/nexus/v1/resources/3284c079-ec97-4bfe-b039-d2c2fa5d8e19/b91f3847-8c5a-45d6-b038-53780059c774/_/https:%2F%2Fopenbluebrain.com%2Fdata%2F3284c079-ec97-4bfe-b039-d2c2fa5d8e19%2Fb91f3847-8c5a-45d6-b038-53780059c774%2F169f755e-745c-481a-bed5-14ed5591a2ee"
                    },
                    "id": "call_TxjrqoPaRNnFmiZwmb9iMkgM",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 4484,
                "output_tokens": 166,
                "total_tokens": 4650,
            },
        ),
    ]
    tool = BlueNaaSTool(
        metadata={
            "url": url,
            "httpx_client": httpx.AsyncClient(),
            "token": "fake_token",
        }
    )
    # First call to bluenaas. Need to validate
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages,
    )
    assert isinstance(response[0], BlueNaaSInvalidatedOutput)
    assert response[0].inputs == tool.create_json_api()
    assert response[1] == {"is_validated": True}

    # Add the output of the first call + the AI message in the state
    messages.extend(
        [
            ToolMessage(
                content="A simulation will be ran with the following inputs...",
                name="bluenaas-tool",
                id="95a70cb3-8afc-4e51-8868-29ea9cf5c8bd",
                tool_call_id="call_IScHmOF8TsJEjqmZ9yL5ehRY",
                artifact={"is_validated": True},
            ),
            AIMessage(
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
            ),
        ]
    )

    messages_2 = [
        *messages,
        HumanMessage(
            content="I agree",
            additional_kwargs={},
            response_metadata={},
            id="20bf996e-8f9f-4ad8-a607-913bf230280d",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_6kYSBUM43aq1zCsf3CHUjSo2",
                        "function": {
                            "arguments": '{"me_model_id":"https://great_memodel.com}',
                            "name": "bluenaas-tool",
                        },
                        "type": "function",
                    }
                ],
                "refusal": None,
            },
            response_metadata={
                "token_usage": {
                    "completion_tokens": 287,
                    "prompt_tokens": 5005,
                    "total_tokens": 5292,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 4864},
                },
                "model_name": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_e2bde53e6e",
                "finish_reason": "tool_calls",
                "logprobs": None,
            },
            id="run-bafc80a4-66fc-4509-a491-b2ea0d0ab505-0",
            tool_calls=[
                {
                    "name": "bluenaas-tool",
                    "args": {
                        "me_model_id": "https://great_memodel.com",
                    },
                    "id": "call_6kYSBUM43aq1zCsf3CHUjSo2",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 5005,
                "output_tokens": 287,
                "total_tokens": 5292,
            },
        ),
    ]
    # Case where we call bluenaas after validating. Run simu
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages_2,
        conditions__celsius=7,
        current_injection__inject_to="axon[1]",
    )
    assert isinstance(response[0], BlueNaaSValidatedOutput)
    assert response[0] == BlueNaaSValidatedOutput(status="success")
    assert response[1] == {"is_validated": False}

    messages_3 = [
        *messages,
        HumanMessage(
            content="Find the id of the thalamus",
            additional_kwargs={},
            response_metadata={},
            id="6f159a5a-e111-4134-bf59-79465b5eed77",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_I6TcpkjCDatXYO57lbAR3t2K",
                        "function": {
                            "arguments": '{"brain_region":"thalamus"}',
                            "name": "resolve-brain-region-tool",
                        },
                        "type": "function",
                    }
                ],
                "refusal": None,
            },
            response_metadata={
                "token_usage": {
                    "completion_tokens": 20,
                    "prompt_tokens": 5016,
                    "total_tokens": 5036,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 4864},
                },
                "model_name": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_e2bde53e6e",
                "finish_reason": "tool_calls",
                "logprobs": None,
            },
            id="run-6450cf4a-d09d-4402-aafa-44412ac8d665-0",
            tool_calls=[
                {
                    "name": "resolve-brain-region-tool",
                    "args": {"brain_region": "thalamus"},
                    "id": "call_I6TcpkjCDatXYO57lbAR3t2K",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 5016,
                "output_tokens": 20,
                "total_tokens": 5036,
            },
        ),
        ToolMessage(
            content='[{"brain_region_name":"Thalamus","brain_region_id":"http://api.brain-map.org/api/v2/data/Structure/549"}]',
            name="resolve-brain-region-tool",
            id="84685110-545d-4622-a270-dbc1799ad800",
            tool_call_id="call_I6TcpkjCDatXYO57lbAR3t2K",
        ),
        AIMessage(
            content="The ID of the thalamus is [http://api.brain-map.org/api/v2/data/Structure/549](http://api.brain-map.org/api/v2/data/Structure/549).",
            additional_kwargs={"refusal": None},
            response_metadata={
                "token_usage": {
                    "completion_tokens": 42,
                    "prompt_tokens": 5078,
                    "total_tokens": 5120,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 4864},
                },
                "model_name": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_e2bde53e6e",
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-be04d50b-9e22-42cc-baac-5f77b874b870-0",
            usage_metadata={
                "input_tokens": 5078,
                "output_tokens": 42,
                "total_tokens": 5120,
            },
        ),
        HumanMessage(
            content="Run the previous simulatiom",
            additional_kwargs={},
            response_metadata={},
            id="b75998a9-c413-4393-bb76-4534f4966122",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_2yAeojomRs9ie2B7frkG0wZR",
                        "function": {
                            "arguments": '{"me_model_id":"https://great_memodel.com}',
                            "name": "bluenaas-tool",
                        },
                        "type": "function",
                    }
                ],
                "refusal": None,
            },
            response_metadata={
                "token_usage": {
                    "completion_tokens": 287,
                    "prompt_tokens": 5133,
                    "total_tokens": 5420,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 4992},
                },
                "model_name": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_e2bde53e6e",
                "finish_reason": "tool_calls",
                "logprobs": None,
            },
            id="run-132d2c6b-47d2-4708-84ab-69f78840b37c-0",
            tool_calls=[
                {
                    "name": "bluenaas-tool",
                    "args": {
                        "me_model_id": "https://great_memodel/com",
                    },
                    "id": "call_2yAeojomRs9ie2B7frkG0wZR",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 5133,
                "output_tokens": 287,
                "total_tokens": 5420,
            },
        ),
    ]

    # Don't validate but completely change topic
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages_3,
    )
    assert isinstance(response[0], BlueNaaSInvalidatedOutput)
    assert response[1] == {"is_validated": True}

    messages_4 = [
        *messages,
        HumanMessage(
            content="Do not run this simulation",
            additional_kwargs={},
            response_metadata={},
            id="ce092494-0e90-45a6-b1ba-79ec20cf2b26",
        ),
        AIMessage(
            content="Understood! The simulation will not be run. If you have any other questions or need further assistance, feel free to ask!",
            additional_kwargs={"refusal": None},
            response_metadata={
                "token_usage": {
                    "completion_tokens": 27,
                    "prompt_tokens": 5026,
                    "total_tokens": 5053,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 4864},
                },
                "model_name": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_e2bde53e6e",
                "finish_reason": "stop",
                "logprobs": None,
            },
            id="run-a2b1ba37-42c4-47b9-a9ad-6ac405572b34-0",
            usage_metadata={
                "input_tokens": 5026,
                "output_tokens": 27,
                "total_tokens": 5053,
            },
        ),
        HumanMessage(
            content="I changed my mind run it",
            additional_kwargs={},
            response_metadata={},
            id="f7ad4abd-e25c-434a-9eea-aff8b19b8505",
        ),
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_IJqeVTyHWom11FHBJO5M16s9",
                        "function": {
                            "arguments": '{"me_model_id":"https://great_memodel.com"}',
                            "name": "bluenaas-tool",
                        },
                        "type": "function",
                    }
                ],
                "refusal": None,
            },
            response_metadata={
                "token_usage": {
                    "completion_tokens": 166,
                    "prompt_tokens": 5066,
                    "total_tokens": 5232,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                    "prompt_tokens_details": {"cached_tokens": 4992},
                },
                "model_name": "gpt-4o-mini-2024-07-18",
                "system_fingerprint": "fp_e2bde53e6e",
                "finish_reason": "tool_calls",
                "logprobs": None,
            },
            id="run-cef8e948-ec95-4754-9028-1654bcccd040-0",
            tool_calls=[
                {
                    "name": "bluenaas-tool",
                    "args": {"me_model_id": "https://great_memodel.com"},
                    "id": "call_IJqeVTyHWom11FHBJO5M16s9",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 5066,
                "output_tokens": 166,
                "total_tokens": 5232,
            },
        ),
    ]

    # Don't validate but ask again to run
    response = await tool._arun(
        me_model_id=me_model_id,
        messages=messages_4,
    )

    assert isinstance(response[0], BlueNaaSInvalidatedOutput)
    assert response[1] == {"is_validated": True}


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
