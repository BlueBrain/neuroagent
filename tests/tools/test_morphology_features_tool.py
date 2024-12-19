"""Tests Morphology features tool."""

import json
from pathlib import Path

import httpx
import pytest

from neuroagent.tools import MorphologyFeatureTool
from neuroagent.tools.morphology_features_tool import (
    MorphologyFeatureInput,
    MorphologyFeatureMetadata,
)


class TestMorphologyFeatureTool:
    @pytest.mark.asyncio
    async def test_arun(self, httpx_mock):
        url = "http://fake_url"
        morphology_id = "https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies/046fb11c-8de8-42e8-9303-9d5a65ac04b9"
        json_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "morphology_id_metadata_response.json"
        )
        with open(json_path) as f:
            morphology_metadata_response = json.load(f)

        # Mock get morphology ids
        httpx_mock.add_response(
            url=url,
            json=morphology_metadata_response,
        )

        morphology_path = Path(__file__).resolve().parent.parent / "data" / "simple.swc"
        with open(morphology_path) as f:
            morphology_content = f.read()

        # Mock get object id request
        httpx_mock.add_response(
            url="https://bbp.epfl.ch/nexus/v1/files/bbp/mouselight/https%3A%2F%2Fbbp.epfl.ch%2Fneurosciencegraph%2Fdata%2Fad8fec6f-d59c-4998-beb4-274fa115add7",
            content=morphology_content,
        )

        tool = MorphologyFeatureTool(
            metadata=MorphologyFeatureMetadata(
                knowledge_graph_url=url,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
            ),
            input_schema=MorphologyFeatureInput(morphology_id=morphology_id),
        )

        response = await tool.arun()
        assert isinstance(response[0], dict)
        assert len(response[0]["feature_dict"]) == 23

    @pytest.mark.asyncio
    async def test_arun_errors_404(self, httpx_mock):
        url = "http://fake_url"
        morphology_id = "https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies/046fb11c-8de8-42e8-9303-9d5a65ac04b9"

        tool = MorphologyFeatureTool(
            metadata=MorphologyFeatureMetadata(
                knowledge_graph_url=url,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
            ),
            input_schema=MorphologyFeatureInput(morphology_id=morphology_id),
        )

        # test different failures
        # Failure 1
        httpx_mock.add_response(
            url=url,
            status_code=404,
        )
        with pytest.raises(ValueError) as tool_exception:
            await tool.arun()

        assert (
            tool_exception.value.args[0] == "We did not find the object"
            " https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies/046fb11c-8de8-42e8-9303-9d5a65ac04b9"
            " you are asking"
        )

    @pytest.mark.asyncio
    async def test_arun_wrong_id(self, httpx_mock):
        url = "http://fake_url"
        morphology_id = "https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies/046fb11c-8de8-42e8-9303-9d5a65ac04b9"

        tool = MorphologyFeatureTool(
            metadata=MorphologyFeatureMetadata(
                knowledge_graph_url=url,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
            ),
            input_schema=MorphologyFeatureInput(morphology_id=morphology_id),
        )

        # Failure 2
        fake_json = {"hits": {"hits": [{"_source": {"@id": "wrong_id"}}]}}
        httpx_mock.add_response(
            url=url,
            json=fake_json,
        )
        with pytest.raises(ValueError) as tool_exception:
            await tool.arun()

        assert (
            tool_exception.value.args[0] == "We did not find the object"
            " https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies/046fb11c-8de8-42e8-9303-9d5a65ac04b9"
            " you are asking"
        )
