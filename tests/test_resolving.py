import pytest
from httpx import AsyncClient
from neuroagent.resolving import (
    es_resolve,
    escape_punctuation,
    resolve_query,
    sparql_exact_resolve,
    sparql_fuzzy_resolve,
)


@pytest.mark.asyncio
async def test_sparql_exact_resolve(httpx_mock, get_resolve_query_output):
    brain_region = "Thalamus"
    url = "http://fakeurl.com"
    mocked_response = get_resolve_query_output[0]
    httpx_mock.add_response(
        url=url,
        json=mocked_response,
    )
    response = await sparql_exact_resolve(
        query=brain_region,
        resource_type="nsg:BrainRegion",
        sparql_view_url=url,
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
    )
    assert response == [
        {
            "label": "Thalamus",
            "id": "http://api.brain-map.org/api/v2/data/Structure/549",
        }
    ]

    httpx_mock.reset()

    mtype = "Interneuron"
    mocked_response = get_resolve_query_output[1]
    httpx_mock.add_response(
        url=url,
        json=mocked_response,
    )
    response = await sparql_exact_resolve(
        query=mtype,
        resource_type="bmo:BrainCellType",
        sparql_view_url=url,
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
    )
    assert response == [
        {"label": "Interneuron", "id": "https://neuroshapes.org/Interneuron"}
    ]


@pytest.mark.asyncio
async def test_sparql_fuzzy_resolve(httpx_mock, get_resolve_query_output):
    brain_region = "Field"
    url = "http://fakeurl.com"
    mocked_response = get_resolve_query_output[2]
    httpx_mock.add_response(
        url=url,
        json=mocked_response,
    )
    response = await sparql_fuzzy_resolve(
        query=brain_region,
        resource_type="nsg:BrainRegion",
        sparql_view_url=url,
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
        search_size=3,
    )
    assert response == [
        {
            "label": "Field CA1",
            "id": "http://api.brain-map.org/api/v2/data/Structure/382",
        },
        {
            "label": "Field CA2",
            "id": "http://api.brain-map.org/api/v2/data/Structure/423",
        },
        {
            "label": "Field CA3",
            "id": "http://api.brain-map.org/api/v2/data/Structure/463",
        },
    ]
    httpx_mock.reset()

    mtype = "Interneu"
    mocked_response = get_resolve_query_output[3]
    httpx_mock.add_response(
        url=url,
        json=mocked_response,
    )
    response = await sparql_fuzzy_resolve(
        query=mtype,
        resource_type="bmo:BrainCellType",
        sparql_view_url=url,
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
        search_size=3,
    )
    assert response == [
        {"label": "Interneuron", "id": "https://neuroshapes.org/Interneuron"},
        {
            "label": "Hippocampus CA3 Oriens Interneuron",
            "id": "http://uri.interlex.org/base/ilx_0105044",
        },
        {
            "label": "Spinal Cord Ventral Horn Interneuron IA",
            "id": "http://uri.interlex.org/base/ilx_0110929",
        },
    ]


@pytest.mark.asyncio
async def test_es_resolve(httpx_mock, get_resolve_query_output):
    brain_region = "Auditory Cortex"
    mocked_response = get_resolve_query_output[4]
    httpx_mock.add_response(
        url="http://goodurl.com",
        json=mocked_response,
    )
    response = await es_resolve(
        query=brain_region,
        resource_type="nsg:BrainRegion",
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
        search_size=3,
        es_view_url="http://goodurl.com",
    )
    assert response == [
        {
            "label": "Cerebral cortex",
            "id": "http://api.brain-map.org/api/v2/data/Structure/688",
        },
        {
            "label": "Cerebellar cortex",
            "id": "http://api.brain-map.org/api/v2/data/Structure/528",
        },
        {
            "label": "Frontal pole, cerebral cortex",
            "id": "http://api.brain-map.org/api/v2/data/Structure/184",
        },
    ]
    httpx_mock.reset()

    mtype = "Ventral neuron"
    mocked_response = get_resolve_query_output[5]
    httpx_mock.add_response(
        url="http://goodurl.com",
        json=mocked_response,
    )
    response = await es_resolve(
        query=mtype,
        resource_type="bmo:BrainCellType",
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
        search_size=3,
        es_view_url="http://goodurl.com",
    )
    assert response == [
        {
            "label": "Ventral Tegmental Area Dopamine Neuron",
            "id": "http://uri.interlex.org/base/ilx_0112352",
        },
        {
            "label": "Spinal Cord Ventral Horn Motor Neuron Gamma",
            "id": "http://uri.interlex.org/base/ilx_0110943",
        },
        {
            "label": "Hypoglossal Nucleus Motor Neuron",
            "id": "http://uri.interlex.org/base/ilx_0105169",
        },
    ]


@pytest.mark.asyncio
async def test_resolve_query(httpx_mock, get_resolve_query_output):
    url = "http://terribleurl.com"
    class_view_url = "http://somewhatokurl.com"
    # Mock exact match to fail
    httpx_mock.add_response(
        url=url,
        json={
            "head": {"vars": ["subject", "predicate", "object", "context"]},
            "results": {"bindings": []},
        },
    )

    # Hit fuzzy match
    httpx_mock.add_response(
        url=url,
        json=get_resolve_query_output[2],
    )

    # Hit ES match
    httpx_mock.add_response(
        url=class_view_url,
        json=get_resolve_query_output[4],
    )
    response = await resolve_query(
        query="Field",
        resource_type="nsg:BrainRegion",
        sparql_view_url=url,
        es_view_url=class_view_url,
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
        search_size=3,
    )
    assert response == [
        {
            "label": "Field CA1",
            "id": "http://api.brain-map.org/api/v2/data/Structure/382",
        },
        {
            "label": "Field CA2",
            "id": "http://api.brain-map.org/api/v2/data/Structure/423",
        },
        {
            "label": "Field CA3",
            "id": "http://api.brain-map.org/api/v2/data/Structure/463",
        },
    ]
    httpx_mock.reset()

    httpx_mock.add_response(url=url, json=get_resolve_query_output[0])

    # Hit fuzzy match
    httpx_mock.add_response(
        url=url,
        json={
            "head": {"vars": ["subject", "predicate", "object", "context"]},
            "results": {"bindings": []},
        },
    )

    # Hit ES match
    httpx_mock.add_response(url=class_view_url, json={"hits": {"hits": []}})

    response = await resolve_query(
        query="Thalamus",
        resource_type="nsg:BrainRegion",
        sparql_view_url=url,
        es_view_url=class_view_url,
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
        search_size=3,
    )
    assert response == [
        {
            "label": "Thalamus",
            "id": "http://api.brain-map.org/api/v2/data/Structure/549",
        }
    ]
    httpx_mock.reset()
    httpx_mock.add_response(
        url=url,
        json={
            "head": {"vars": ["subject", "predicate", "object", "context"]},
            "results": {"bindings": []},
        },
    )

    # Hit fuzzy match
    httpx_mock.add_response(
        url=url,
        json={
            "head": {"vars": ["subject", "predicate", "object", "context"]},
            "results": {"bindings": []},
        },
    )

    # Hit ES match
    httpx_mock.add_response(
        url=class_view_url,
        json=get_resolve_query_output[4],
    )
    response = await resolve_query(
        query="Auditory Cortex",
        resource_type="nsg:BrainRegion",
        sparql_view_url=url,
        es_view_url=class_view_url,
        token="greattokenpleasedontexpire",
        httpx_client=AsyncClient(),
        search_size=3,
    )
    assert response == [
        {
            "label": "Cerebral cortex",
            "id": "http://api.brain-map.org/api/v2/data/Structure/688",
        },
        {
            "label": "Cerebellar cortex",
            "id": "http://api.brain-map.org/api/v2/data/Structure/528",
        },
        {
            "label": "Frontal pole, cerebral cortex",
            "id": "http://api.brain-map.org/api/v2/data/Structure/184",
        },
    ]


@pytest.mark.parametrize(
    "before,after",
    [
        ("this is a text", "this is a text"),
        ("this is text with punctuation!", "this is text with punctuation\\\\!"),
    ],
)
def test_escape_punctuation(before, after):
    assert after == escape_punctuation(before)


def test_failing_escape_punctuation():
    text = 15  # this is not a string
    with pytest.raises(TypeError) as e:
        escape_punctuation(text)
    assert e.value.args[0] == "Only accepting strings."
