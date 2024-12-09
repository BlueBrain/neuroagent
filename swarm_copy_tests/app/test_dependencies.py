"""Test dependencies."""

import json
import os
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import Mock, patch

import pytest
from httpx import AsyncClient
from fastapi import Request, HTTPException

from swarm_copy.app.app_utils import setup_engine
from swarm_copy.app.database.sql_schemas import Base, Threads
from swarm_copy.app.dependencies import (
    Settings,
    get_cell_types_kg_hierarchy,
    get_connection_string,
    get_httpx_client,
    get_settings,
    get_update_kg_hierarchy,
    get_user_id, get_session, get_vlab_and_project, get_starting_agent, get_kg_token,
)
from swarm_copy.new_types import Agent


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



@pytest.mark.asyncio
@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
async def test_get_vlab_and_project(
    patch_required_env, httpx_mock, db_connection, monkeypatch
):
    # Setup DB with one thread to do the tests
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__VALIDATE_TOKEN", "true")
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "Super_user"
    token = "fake_token"
    httpx_client = AsyncClient()
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project",
        json="test_project_ID",
    )

    # create test thread table
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    try:
        # Test with info in headers.
        good_request_headers = Request(
            scope={
                "type": "http",
                "method": "Get",
                "url": "http://fake_url/thread_id",
                "headers": [
                    (b"x-virtual-lab-id", b"test_vlab"),
                    (b"x-project-id", b"test_project"),
                ],
            },
        )
        ids = await get_vlab_and_project(
            user_id=user_id,
            session=session,
            request=good_request_headers,
            settings=test_settings,
            token=token,
            httpx_client=httpx_client,
        )
        assert ids == {"vlab_id": "test_vlab", "project_id": "test_project"}
    finally:
        # don't forget to close the session, otherwise the tests hangs.
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_vlab_and_project_no_info_in_headers(
    patch_required_env, db_connection, monkeypatch
):
    # Setup DB with one thread to do the tests
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__VALIDATE_TOKEN", "true")
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "Super_user"
    token = "fake_token"
    httpx_client = AsyncClient()

    # create test thread table
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    try:
        # Test with no infos in headers.
        bad_request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "scheme": "http",
                "server": ("example.com", 80),
                "path_params": {"dummy_patram": "fake_thread_id"},
                "headers": [
                    (b"wong_header", b"wrong value"),
                ],
            }
        )
        with pytest.raises(HTTPException) as error:
            await get_vlab_and_project(
                user_id=user_id,
                session=session,
                request=bad_request,
                settings=test_settings,
                token=token,
                httpx_client=httpx_client,
            )
        assert (
            error.value.detail == "Thread not found."
        )
    finally:
        # don't forget to close the session, otherwise the tests hangs.
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
async def test_get_vlab_and_project_valid_thread_id(
    patch_required_env, httpx_mock, db_connection, monkeypatch
):
    # Setup DB with one thread to do the tests
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__VALIDATE_TOKEN", "true")
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "Super_user"
    token = "fake_token"
    httpx_client = AsyncClient()
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab_DB/projects/project_id_DB",
        json="test_project_ID",
    )


    # create test thread table
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    try:
        # Test with no infos in headers, but valid thread_ID.
        good_request_DB = Request(
            scope={
                "type": "http",
                "method": "GET",
                "scheme": "http",
                "server": ("example.com", 80),
                "path_params": {"thread_id": new_thread.thread_id},
                "headers": [
                    (b"wong_header", b"wrong value"),
                ],
            }
        )
        ids_from_DB = await get_vlab_and_project(
            user_id=user_id,
            session=session,
            request=good_request_DB,
            settings=test_settings,
            token=token,
            httpx_client=httpx_client,
        )
        assert ids_from_DB == {"vlab_id": "test_vlab_DB", "project_id": "project_id_DB"}

    finally:
        # don't forget to close the session, otherwise the tests hangs.
        await session.close()
        await engine.dispose()


def test_get_starting_agent(patch_required_env):
    settings = Settings()
    agent = get_starting_agent(None, settings)

    assert isinstance(agent, Agent)


@pytest.mark.parametrize(
    "input_token, expected_token",
    [
        ("existing_token", "existing_token"),
        (None, "new_token"),
    ],
)
def test_get_kg_token(patch_required_env, input_token, expected_token):
    settings = Settings()
    mock = Mock()
    mock.token.return_value = {"access_token": expected_token}
    with (
        patch("swarm_copy.app.dependencies.KeycloakOpenID", return_value=mock),
    ):
        result = get_kg_token(settings, input_token)
        assert result == expected_token
