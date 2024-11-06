"""Tests Get Morpho tool."""

import json
from pathlib import Path

import httpx
import pytest

from swarm_copy.tools import GetMorphoTool
from swarm_copy.tools.get_morpho_tool import KnowledgeGraphOutput, GetMorphoInput, GetMorphoMetadata


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
                mtype_id="brain_region_id_link/549"
            ),
            metadata=GetMorphoMetadata(
                knowledge_graph_url=url,
                morpho_search_size=2,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
                brainregion_path=brain_region_json_path,
                celltypes_path=tmp_path
            )
        )
        response = await tool.arun()
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], KnowledgeGraphOutput)

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
                celltypes_path=tmp_path
            )
        )
        with pytest.raises(KeyError) as tool_exception:
            await tool.arun()

        assert tool_exception.value.args[0] == "hits"
