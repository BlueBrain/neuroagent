"""Tests Traces tool."""

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.tools import ToolException
from neuroagent.tools import GetTracesTool
from neuroagent.tools.traces_tool import TracesOutput


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
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
                "brainregion_path": brain_region_json_path,
            }
        )

        response = await tool._arun(brain_region_id="brain_region_id_link/549")
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], TracesOutput)

        # With specific etype
        response = await tool._arun(
            brain_region_id="brain_region_id_link/549", etype="bAC"
        )
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], TracesOutput)

    def test_create_query(self):
        brain_region_ids = {"brain_region_id1"}
        etype = "bAC"

        tool = GetTracesTool(metadata={"search_size": 2})
        entire_query = tool.create_query(brain_region_ids=brain_region_ids, etype=etype)
        expected_query = {
            "size": 2,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "term": {
                                            "brainRegion.@id.keyword": (
                                                "brain_region_id1"
                                            )
                                        }
                                    },
                                ]
                            }
                        },
                        {
                            "term": {
                                "eType.@id.keyword": (
                                    "http://uri.interlex.org/base/ilx_0738199"
                                )
                            }
                        },
                        {
                            "term": {
                                "@type.keyword": "https://bbp.epfl.ch/ontologies/core/bmo/ExperimentalTrace"
                            }
                        },
                        {"term": {"curated": True}},
                        {"term": {"deprecated": False}},
                    ]
                }
            },
        }
        assert entire_query == expected_query

    @pytest.mark.asyncio
    async def test_arun_errors(self, httpx_mock, brain_region_json_path):
        url = "http://fake_url"

        # Mocking an issue
        httpx_mock.add_response(
            url=url,
            json={},
        )

        tool = GetTracesTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
                "brainregion_path": brain_region_json_path,
            }
        )

        with pytest.raises(ToolException) as tool_exception:
            _ = await tool._arun(brain_region_id="brain_region_id_link/549")

        assert tool_exception.value.args[0] == "'hits'"
