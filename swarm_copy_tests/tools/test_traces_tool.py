"""Tests Traces tool."""

import json
from pathlib import Path

import httpx
import pytest

from swarm_copy.tools import GetTracesTool
from swarm_copy.tools.traces_tool import TracesOutput, GetTracesMetadata, GetTracesInput


class TestTracesTool:
    @pytest.mark.httpx_mock(can_send_already_matched_responses=True)
    @pytest.mark.asyncio
    async def test_arun(self, httpx_mock, brain_region_json_path):
        url = "http://fake_url"
        json_path = Path(__file__).resolve().parent.parent / "data" / "get_traces.json"
        with open(json_path) as f:
            get_traces_response = json.load(f)

        httpx_mock.add_response(
            url=url,
            json=get_traces_response,
        )

        tool = GetTracesTool(
            metadata=GetTracesMetadata(
                knowledge_graph_url=url,
                trace_search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
                brainregion_path=brain_region_json_path,
            ),
            input_schema=GetTracesInput(
                brain_region_id="brain_region_id_link/549"
            )
        )

        response = await tool.arun()
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], TracesOutput)
        assert isinstance(response[0], TracesOutput)

    @pytest.mark.httpx_mock(can_send_already_matched_responses=True)
    @pytest.mark.asyncio
    async def test_arun_with_etype(self, httpx_mock, brain_region_json_path):
        url = "http://fake_url"
        json_path = Path(__file__).resolve().parent.parent / "data" / "get_traces.json"
        with open(json_path) as f:
            get_traces_response = json.load(f)

        httpx_mock.add_response(
            url=url,
            json=get_traces_response,
        )

        tool = GetTracesTool(
            metadata=GetTracesMetadata(
                knowledge_graph_url=url,
                trace_search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
                brainregion_path=brain_region_json_path,
            ),
            input_schema=GetTracesInput(
                brain_region_id="brain_region_id_link/549", etype_id="bAC_id/123"
            )
        )
        response = await tool.arun()
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], TracesOutput)
