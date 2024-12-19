"""Tests Get Morpho tool."""

import json
from pathlib import Path

import httpx
import pytest

from neuroagent.tools import GetMorphoTool
from neuroagent.tools.get_morpho_tool import GetMorphoInput, GetMorphoMetadata


class TestGetMorphoTool:
    @pytest.mark.asyncio
    async def test_arun(self, httpx_mock, brain_region_json_path, tmp_path):
        url = "http://fake_url"
        json_path = (
            Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"
        )
        with open(json_path) as f:
            knowledge_graph_response = json.load(f)

        httpx_mock.add_response(
            url=url,
            json=knowledge_graph_response,
        )
        tool = GetMorphoTool(
            input_schema=GetMorphoInput(
                brain_region_id="brain_region_id_link/549",
                mtype_id="brain_region_id_link/549",
            ),
            metadata=GetMorphoMetadata(
                knowledge_graph_url=url,
                morpho_search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
                brainregion_path=brain_region_json_path,
                celltypes_path=tmp_path,
            ),
        )
        response = await tool.arun()
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], dict)

    @pytest.mark.asyncio
    async def test_arun_errors(self, httpx_mock, brain_region_json_path, tmp_path):
        url = "http://fake_url"
        httpx_mock.add_response(
            url=url,
            json={},
        )

        tool = GetMorphoTool(
            input_schema=GetMorphoInput(
                brain_region_id="brain_region_id_link/bad",
                mtype_id="brain_region_id_link/superbad",
            ),
            metadata=GetMorphoMetadata(
                knowledge_graph_url=url,
                morpho_search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
                brainregion_path=brain_region_json_path,
                celltypes_path=tmp_path,
            ),
        )
        with pytest.raises(KeyError) as tool_exception:
            await tool.arun()

        assert tool_exception.value.args[0] == "hits"


def test_create_query(brain_region_json_path, tmp_path):
    url = "http://fake_url"

    tool = GetMorphoTool(
        input_schema=GetMorphoInput(
            brain_region_id="not_needed",
            mtype_id="not_needed",
        ),
        metadata=GetMorphoMetadata(
            knowledge_graph_url=url,
            morpho_search_size=2,
            httpx_client=httpx.AsyncClient(),
            token="fake_token",
            brainregion_path=brain_region_json_path,
            celltypes_path=tmp_path,
        ),
    )

    # This should be a set, but passing a list here ensures that the test doesn;t rely on order.
    brain_regions_ids = ["brain-region-id/68", "brain-region-id/131"]
    mtype_id = "mtype-id/1234"

    entire_query = tool.create_query(
        brain_regions_ids=brain_regions_ids, mtype_ids={mtype_id}
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
                    {"bool": {"should": [{"term": {"mType.@id.keyword": mtype_id}}]}},
                    {
                        "term": {
                            "@type.keyword": (
                                "https://neuroshapes.org/ReconstructedNeuronMorphology"
                            )
                        }
                    },
                    {"term": {"deprecated": False}},
                    {"term": {"curated": True}},
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
                    {
                        "term": {
                            "@type.keyword": (
                                "https://neuroshapes.org/ReconstructedNeuronMorphology"
                            )
                        }
                    },
                    {"term": {"deprecated": False}},
                    {"term": {"curated": True}},
                ]
            }
        },
    }
    assert entire_query1 == expected_query1
