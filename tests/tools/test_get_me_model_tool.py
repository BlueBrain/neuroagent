"""Tests Get Morpho tool."""

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.tools import ToolException

from neuroagent.tools.get_me_model_tool import GetMEModelTool, MEModelOutput


class TestGetMEModelTool:
    @pytest.mark.asyncio
    async def test_arun(self, httpx_mock, brain_region_json_path, tmp_path):
        url = "http://fake_url"
        json_path = (
            Path(__file__).resolve().parent.parent / "data" / "kg_me_model_output.json"
        )
        with open(json_path) as f:
            knowledge_graph_response = json.load(f)

        httpx_mock.add_response(
            url=url,
            json=knowledge_graph_response,
        )
        tool = GetMEModelTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
                "brainregion_path": brain_region_json_path,
                "celltypes_path": tmp_path,
            }
        )
        response = await tool._arun(
            brain_region_id="brain_region_id_link/549",
            mtype_id="mtype_id_link/549",
            etype_id="etype_id_link/549",
        )
        assert isinstance(response, list)
        assert len(response) == 1
        assert isinstance(response[0], MEModelOutput)

    @pytest.mark.asyncio
    async def test_arun_errors(self, httpx_mock, brain_region_json_path, tmp_path):
        url = "http://fake_url"
        httpx_mock.add_response(
            url=url,
            json={},
        )

        tool = GetMEModelTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
                "brainregion_path": brain_region_json_path,
                "celltypes_path": tmp_path,
            }
        )
        with pytest.raises(ToolException) as tool_exception:
            _ = await tool._arun(
                brain_region_id="brain_region_id_link/bad",
                mtype_id="mtype_id_link/superbad",
                etype_id="etype_id_link/superbad",
            )

        assert tool_exception.value.args[0] == "'hits'"


def test_create_query():
    url = "http://fake_url"

    tool = GetMEModelTool(
        metadata={
            "url": url,
            "search_size": 2,
            "httpx_client": httpx.AsyncClient(),
            "token": "fake_token",
        }
    )

    # This should be a set, but passing a list here ensures that the test doesn;t rely on order.
    brain_regions_ids = ["brain-region-id/68", "brain-region-id/131"]
    mtype_id = "mtype-id/1234"
    etype_id = "etype-id/1234"

    entire_query = tool.create_query(
        brain_regions_ids=brain_regions_ids, mtype_ids={mtype_id}, etype_id=etype_id
    )
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
                                        "brainRegion.@id.keyword": "brain-region-id/68"
                                    }
                                },
                                {
                                    "term": {
                                        "brainRegion.@id.keyword": "brain-region-id/131"
                                    }
                                },
                            ]
                        }
                    },
                    {"term": {"@type.keyword": ("https://neuroshapes.org/MEModel")}},
                    {"term": {"deprecated": False}},
                    {
                        "bool": {
                            "should": [{"term": {"mType.@id.keyword": "mtype-id/1234"}}]
                        }
                    },
                    {"term": {"eType.@id.keyword": "etype-id/1234"}},
                ]
            }
        },
    }
    assert isinstance(entire_query, dict)
    assert entire_query == expected_query

    # Case 2 with no mtype
    entire_query1 = tool.create_query(brain_regions_ids=brain_regions_ids)
    expected_query1 = {
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
                                        "brainRegion.@id.keyword": "brain-region-id/68"
                                    }
                                },
                                {
                                    "term": {
                                        "brainRegion.@id.keyword": "brain-region-id/131"
                                    }
                                },
                            ]
                        }
                    },
                    {"term": {"@type.keyword": ("https://neuroshapes.org/MEModel")}},
                    {"term": {"deprecated": False}},
                ]
            }
        },
    }
    assert entire_query1 == expected_query1
