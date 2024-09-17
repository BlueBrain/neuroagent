"""Tests Electrophys tool."""

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.tools import ToolException
from neuroagent.tools import ElectrophysFeatureTool
from neuroagent.tools.electrophys_tool import (
    CALCULATED_FEATURES,
    AmplitudeInput,
    FeaturesOutput,
)


class TestElectrophysTool:
    @pytest.mark.asyncio
    async def test_arun(self, httpx_mock):
        url = "http://fake_url"
        json_path = (
            Path(__file__).resolve().parent.parent / "data" / "trace_id_metadata.json"
        )
        with open(json_path) as f:
            electrophys_response = json.load(f)

        httpx_mock.add_response(
            url=url,
            json=electrophys_response,
        )

        trace_path = Path(__file__).resolve().parent.parent / "data" / "99111002.nwb"
        with open(trace_path, "rb") as f:
            trace_content = f.read()

        httpx_mock.add_response(
            url="https://bbp.epfl.ch/nexus/v1/files/demo/morpho-demo/https%3A%2F%2Fbbp.epfl.ch%2Fdata%2Fdata%2Fdemo%2Fmorpho-demo%2F01dffb7b-1122-4e1a-9acf-837e683da4ba",
            content=trace_content,
        )

        tool = ElectrophysFeatureTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
            }
        )

        trace_id = "https://bbp.epfl.ch/data/demo/morpho-demo/1761e604-03fc-452b-9bf2-2214782bb751"

        response = await tool._arun(
            trace_id=trace_id,
            stimuli_types=[
                "step",
            ],
            calculated_feature=[
                "mean_frequency",
            ],
        )
        assert isinstance(response, FeaturesOutput)
        assert isinstance(response.feature_dict, dict)
        assert len(response.feature_dict.keys()) == 1
        assert (
            len(response.feature_dict["step_0"].keys())
            == 2  # mean_frequency + 1 for stimulus current added manually
        )

        # With specified amplitude
        response = await tool._arun(
            trace_id=trace_id,
            stimuli_types=[
                "step",
            ],
            calculated_feature=[
                "mean_frequency",
            ],
            amplitude=AmplitudeInput(min_value=-0.5, max_value=1),
        )
        assert isinstance(response, FeaturesOutput)
        assert isinstance(response.feature_dict, dict)
        assert len(response.feature_dict.keys()) == 1
        assert (
            len(response.feature_dict["step_0.25"].keys())
            == 2  # mean_frequency + 1 for stimulus current added manually
        )

        # Without stimuli types and calculated features
        response = await tool._arun(
            trace_id=trace_id,
            stimuli_types=[
                "step",
            ],
            calculated_feature=[],
        )
        assert isinstance(response, FeaturesOutput)
        assert isinstance(response.feature_dict, dict)
        assert len(response.feature_dict.keys()) == 1
        assert (
            len(response.feature_dict["step_0"].keys())
            == len(list(CALCULATED_FEATURES.__args__[0].__args__))
            + 1  # 1 for stimulus current added manually
        )

    @pytest.mark.asyncio
    async def test_arun_errors(self):
        # Do not receive trace content back
        url = "http://fake_url"

        tool = ElectrophysFeatureTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
            }
        )

        with pytest.raises(ToolException) as tool_exception:
            _ = await tool._arun(
                trace_id="wrong-trace-id",
                stimuli_types=[
                    "idrest",
                ],
                calculated_feature=[
                    "mean_frequency",
                ],
            )

        assert (
            tool_exception.value.args[0]
            == "The provided ID (wrong-trace-id) is not valid."
        )
