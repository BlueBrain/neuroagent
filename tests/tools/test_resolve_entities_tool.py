"""Test the revole_brain_region_tool."""

import pytest
from httpx import AsyncClient

from neuroagent.tools import ResolveEntitiesTool
from neuroagent.tools.resolve_entities_tool import (
    BRResolveOutput,
    EtypeResolveOutput,
    MTypeResolveOutput,
    ResolveBRInput,
    ResolveBRMetadata,
)


@pytest.mark.asyncio
async def test_arun(httpx_mock, get_resolve_query_output):
    # Mock exact match to fail
    httpx_mock.add_response(
        url="http://fake_sparql_url.com/78",
        json={
            "head": {"vars": ["subject", "predicate", "object", "context"]},
            "results": {"bindings": []},
        },
    )

    # Hit fuzzy match
    httpx_mock.add_response(
        url="http://fake_sparql_url.com/78",
        json=get_resolve_query_output[2],
    )

    # Hit ES match
    httpx_mock.add_response(
        url="http://fake_class_url.com/78",
        json=get_resolve_query_output[4],
    )

    # Mock exact match to fail (mtype)
    httpx_mock.add_response(
        url="http://fake_sparql_url.com/78",
        json={
            "head": {"vars": ["subject", "predicate", "object", "context"]},
            "results": {"bindings": []},
        },
    )

    # Hit fuzzy match (mtype)
    httpx_mock.add_response(
        url="http://fake_sparql_url.com/78",
        json=get_resolve_query_output[3],
    )
    # Hit ES match (mtype).
    httpx_mock.add_response(
        url="http://fake_class_url.com/78", json=get_resolve_query_output[5]
    )

    tool = ResolveEntitiesTool(
        metadata=ResolveBRMetadata(
            token="greattokenpleasedontexpire",
            httpx_client=AsyncClient(timeout=None),
            kg_sparql_url="http://fake_sparql_url.com/78",
            kg_class_view_url="http://fake_class_url.com/78",
        ),
        input_schema=ResolveBRInput(
            brain_region="Field", mtype="Interneu", etype="bAC"
        ),
    )

    response = await tool.arun()
    assert response == [
        BRResolveOutput(
            brain_region_name="Field CA1",
            brain_region_id="http://api.brain-map.org/api/v2/data/Structure/382",
        ).model_dump(),
        BRResolveOutput(
            brain_region_name="Field CA2",
            brain_region_id="http://api.brain-map.org/api/v2/data/Structure/423",
        ).model_dump(),
        BRResolveOutput(
            brain_region_name="Field CA3",
            brain_region_id="http://api.brain-map.org/api/v2/data/Structure/463",
        ).model_dump(),
        MTypeResolveOutput(
            mtype_name="Interneuron", mtype_id="https://neuroshapes.org/Interneuron"
        ).model_dump(),
        MTypeResolveOutput(
            mtype_name="Hippocampus CA3 Oriens Interneuron",
            mtype_id="http://uri.interlex.org/base/ilx_0105044",
        ).model_dump(),
        MTypeResolveOutput(
            mtype_name="Spinal Cord Ventral Horn Interneuron IA",
            mtype_id="http://uri.interlex.org/base/ilx_0110929",
        ).model_dump(),
        EtypeResolveOutput(
            etype_name="bAC", etype_id="http://uri.interlex.org/base/ilx_0738199"
        ).model_dump(),
    ]
