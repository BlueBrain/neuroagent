"""Tests BlueNaaS tool."""

import httpx
import pytest
from langchain_core.tools import ToolException

from neuroagent.tools import BlueNaaSTool
from neuroagent.tools.bluenaas_tool import BlueNaaSOutput, RecordingLocation


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
    response = await tool._arun(
        me_model_id=me_model_id,
        conditions__celsius=7,
        current_injection__inject_to="axon[1]",
    )
    assert isinstance(response, BlueNaaSOutput)


@pytest.mark.asyncio
async def test_arun_errors(httpx_mock, brain_region_json_path, tmp_path):
    url = "http://fake_url"
    httpx_mock.add_exception(httpx.ReadTimeout("Unable to read within timeout"))

    tool = BlueNaaSTool(
        metadata={
            "url": url,
            "httpx_client": httpx.AsyncClient(),
            "token": "fake_token",
        }
    )
    with pytest.raises(ToolException) as tool_exception:
        _ = await tool._arun(
            me_model_id="great_id",
        )

    assert tool_exception.value.args[0] == "Unable to read within timeout"


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
