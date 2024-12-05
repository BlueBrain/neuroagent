"""Test dependencies."""

import json
import os
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import Mock

import pytest
from httpx import AsyncClient

from swarm_copy.app.dependencies import (
    Settings,
    get_cell_types_kg_hierarchy,
    get_connection_string,
    get_httpx_client,
    get_settings,
    get_update_kg_hierarchy,
    get_user_id,
)


def test_get_settings(patch_required_env):
    settings = get_settings()
    assert settings.tools.literature.url == "https://fake_url"
    assert settings.knowledge_graph.url == "https://fake_url/api/nexus/v1/search/query/"


@pytest.mark.asyncio
async def test_get_httpx_client():
    request = Mock()
    request.headers = {"x-request-id": "greatid"}
    httpx_client_iterator = get_httpx_client(request=request)
    assert isinstance(httpx_client_iterator, AsyncIterator)
    async for httpx_client in httpx_client_iterator:
        assert isinstance(httpx_client, AsyncClient)
        assert httpx_client.headers["x-request-id"] == "greatid"


@pytest.mark.asyncio
async def test_get_user(httpx_mock, monkeypatch, patch_required_env):
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__USERNAME", "fake_username")
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__PASSWORD", "fake_password")
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__ISSUER", "https://great_issuer.com")
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__VALIDATE_TOKEN", "true")

    fake_response = {
        "sub": "12345",
        "email_verified": False,
        "name": "Machine Learning Test User",
        "groups": [],
        "preferred_username": "sbo-ml",
        "given_name": "Machine Learning",
        "family_name": "Test User",
        "email": "email@epfl.ch",
    }
    httpx_mock.add_response(
        url="https://great_issuer.com/protocol/openid-connect/userinfo",
        json=fake_response,
    )

    settings = Settings()
    client = AsyncClient()
    token = "eyJgreattoken"
    user_id = await get_user_id(token=token, settings=settings, httpx_client=client)

    assert user_id == fake_response["sub"]


@pytest.mark.asyncio
async def test_get_update_kg_hierarchy(
    tmp_path, httpx_mock, monkeypatch, patch_required_env
):
    token = "fake_token"
    file_name = "fake_file"
    client = AsyncClient()

    file_url = "https://fake_file_url"

    monkeypatch.setenv(
        "NEUROAGENT_KNOWLEDGE_GRAPH__HIERARCHY_URL", "http://fake_hierarchy_url.com"
    )

    settings = Settings(
        knowledge_graph={"br_saving_path": tmp_path / "test_brain_region.json"}
    )

    json_response_url = {
        "head": {"vars": ["file_url"]},
        "results": {"bindings": [{"file_url": {"type": "uri", "value": file_url}}]},
    }
    with open(
        Path(__file__).parent.parent.parent
        / "tests"
        / "data"
        / "KG_brain_regions_hierarchy_test.json"
    ) as fh:
        json_response_file = json.load(fh)

    httpx_mock.add_response(
        url=settings.knowledge_graph.sparql_url, json=json_response_url
    )
    httpx_mock.add_response(url=file_url, json=json_response_file)

    await get_update_kg_hierarchy(
        token,
        client,
        settings,
        file_name,
    )

    assert os.path.exists(settings.knowledge_graph.br_saving_path)


@pytest.mark.asyncio
async def test_get_cell_types_kg_hierarchy(
    tmp_path, httpx_mock, monkeypatch, patch_required_env
):
    token = "fake_token"
    file_name = "fake_file"
    client = AsyncClient()

    file_url = "https://fake_file_url"
    monkeypatch.setenv(
        "NEUROAGENT_KNOWLEDGE_GRAPH__HIERARCHY_URL", "http://fake_hierarchy_url.com"
    )

    settings = Settings(
        knowledge_graph={"ct_saving_path": tmp_path / "test_cell_types_region.json"}
    )

    json_response_url = {
        "head": {"vars": ["file_url"]},
        "results": {"bindings": [{"file_url": {"type": "uri", "value": file_url}}]},
    }
    with open(
        Path(__file__).parent.parent.parent
        / "tests"
        / "data"
        / "kg_cell_types_hierarchy_test.json"
    ) as fh:
        json_response_file = json.load(fh)

    httpx_mock.add_response(
        url=settings.knowledge_graph.sparql_url, json=json_response_url
    )
    httpx_mock.add_response(url=file_url, json=json_response_file)

    await get_cell_types_kg_hierarchy(
        token,
        client,
        settings,
        file_name,
    )

    assert os.path.exists(settings.knowledge_graph.ct_saving_path)


def test_get_connection_string_full(monkeypatch, patch_required_env):
    monkeypatch.setenv("NEUROAGENT_DB__PREFIX", "http://")
    monkeypatch.setenv("NEUROAGENT_DB__USER", "John")
    monkeypatch.setenv("NEUROAGENT_DB__PASSWORD", "Doe")
    monkeypatch.setenv("NEUROAGENT_DB__HOST", "localhost")
    monkeypatch.setenv("NEUROAGENT_DB__PORT", "5000")
    monkeypatch.setenv("NEUROAGENT_DB__NAME", "test")

    settings = Settings()
    result = get_connection_string(settings)
    assert (
        result == "http://John:Doe@localhost:5000/test"
    ), "must return fully formed connection string"
