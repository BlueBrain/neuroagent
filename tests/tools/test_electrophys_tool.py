"""Tests Electrophys tool."""

import json
from pathlib import Path

import httpx
import pytest

from neuroagent.tools import ElectrophysFeatureTool
from neuroagent.tools.electrophys_tool import (
    CALCULATED_FEATURES,
    AmplitudeInput,
    ElectrophysInput,
    ElectrophysMetadata,
)


class TestElectrophysTool:
    @pytest.mark.httpx_mock(can_send_already_matched_responses=True)
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

        trace_id = "https://bbp.epfl.ch/data/demo/morpho-demo/1761e604-03fc-452b-9bf2-2214782bb751"

        tool = ElectrophysFeatureTool(
            input_schema=ElectrophysInput(
                trace_id=trace_id,
                stimuli_types=[
                    "step",
                ],
                calculated_feature=[
                    "mean_frequency",
                ],
                amplitude=None,
            ),
            metadata=ElectrophysMetadata(
                knowledge_graph_url=url,
                search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
            ),
        )
        response = await tool.arun()
        assert isinstance(response, dict)
        assert len(response["feature_dict"].keys()) == 1
        assert (
            len(response["feature_dict"]["step_0"].keys())
            == 2  # mean_frequency + 1 for stimulus current added manually
        )

    @pytest.mark.httpx_mock(can_send_already_matched_responses=True)
    @pytest.mark.asyncio
    async def test_arun_with_amplitude(self, httpx_mock):
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

        trace_id = "https://bbp.epfl.ch/data/demo/morpho-demo/1761e604-03fc-452b-9bf2-2214782bb751"

        tool = ElectrophysFeatureTool(
            input_schema=ElectrophysInput(
                trace_id=trace_id,
                stimuli_types=[
                    "step",
                ],
                calculated_feature=[
                    "mean_frequency",
                ],
                amplitude=AmplitudeInput(min_value=-0.5, max_value=1),
            ),
            metadata=ElectrophysMetadata(
                knowledge_graph_url=url,
                search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
            ),
        )
        response = await tool.arun()
        assert isinstance(response, dict)
        assert len(response["feature_dict"].keys()) == 1
        assert (
            len(response["feature_dict"]["step_0.25"].keys())
            == 2  # mean_frequency + 1 for stimulus current added manually
        )

    @pytest.mark.httpx_mock(can_send_already_matched_responses=True)
    @pytest.mark.asyncio
    async def test_arun_without_stimuli_types(self, httpx_mock):
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

        trace_id = "https://bbp.epfl.ch/data/demo/morpho-demo/1761e604-03fc-452b-9bf2-2214782bb751"

        tool = ElectrophysFeatureTool(
            input_schema=ElectrophysInput(
                trace_id=trace_id,
                stimuli_types=[
                    "step",
                ],
                calculated_feature=[],
                amplitude=None,
            ),
            metadata=ElectrophysMetadata(
                knowledge_graph_url=url,
                search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
            ),
        )

        # Without stimuli types and calculated features
        response = await tool.arun()
        assert isinstance(response, dict)
        assert len(response["feature_dict"].keys()) == 1
        assert (
            len(response["feature_dict"]["step_0"].keys())
            == len(list(CALCULATED_FEATURES.__args__[0].__args__))
            + 1  # 1 for stimulus current added manually
        )

    @pytest.mark.asyncio
    async def test_arun_errors(self):
        # Do not receive trace content back
        url = "http://fake_url"

        tool = ElectrophysFeatureTool(
            input_schema=ElectrophysInput(
                trace_id="wrong-trace-id",
                stimuli_types=[
                    "idrest",
                ],
                calculated_feature=[
                    "mean_frequency",
                ],
                amplitude=None,
            ),
            metadata=ElectrophysMetadata(
                knowledge_graph_url=url,
                search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
            ),
        )

        with pytest.raises(ValueError) as tool_exception:
            await tool.arun()

        assert (
            tool_exception.value.args[0]
            == "The provided ID (wrong-trace-id) is not valid."
        )
