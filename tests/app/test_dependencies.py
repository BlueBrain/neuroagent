"""Test dependencies."""

import json
import os
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import Mock

import pytest
from fastapi import Request
from fastapi.exceptions import HTTPException
from httpx import AsyncClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from neuroagent.agents import SimpleAgent, SimpleChatAgent
from neuroagent.app.dependencies import (
    Settings,
    get_agent,
    get_agent_memory,
    get_bluenaas_tool,
    get_brain_region_resolver_tool,
    get_cell_types_kg_hierarchy,
    get_chat_agent,
    get_electrophys_feature_tool,
    get_engine,
    get_httpx_client,
    get_kg_morpho_feature_tool,
    get_language_model,
    get_literature_tool,
    get_me_model_tool,
    get_morpho_tool,
    get_morphology_feature_tool,
    get_session,
    get_traces_tool,
    get_update_kg_hierarchy,
    get_user_id,
    thread_to_vp,
    validate_project,
)
from neuroagent.app.routers.database.schemas import Base, Threads
from neuroagent.tools import (
    ElectrophysFeatureTool,
    GetMEModelTool,
    GetMorphoTool,
    GetTracesTool,
    KGMorphoFeatureTool,
    LiteratureSearchTool,
    MorphologyFeatureTool,
)


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


def test_get_literature_tool(monkeypatch, patch_required_env):
    url = "https://fake_url"

    httpx_client = AsyncClient()
    settings = Settings()
    token = "fake_token"

    literature_tool = get_literature_tool(token, settings, httpx_client)
    assert isinstance(literature_tool, LiteratureSearchTool)
    assert literature_tool.metadata["url"] == url
    assert literature_tool.metadata["retriever_k"] == 500
    assert literature_tool.metadata["reranker_k"] == 8
    assert literature_tool.metadata["use_reranker"] is True

    monkeypatch.setenv("NEUROAGENT_TOOLS__LITERATURE__RETRIEVER_K", "30")
    monkeypatch.setenv("NEUROAGENT_TOOLS__LITERATURE__RERANKER_K", "1")
    monkeypatch.setenv("NEUROAGENT_TOOLS__LITERATURE__USE_RERANKER", "false")
    settings = Settings()

    literature_tool = get_literature_tool(token, settings, httpx_client)
    assert isinstance(literature_tool, LiteratureSearchTool)
    assert literature_tool.metadata["url"] == url
    assert literature_tool.metadata["retriever_k"] == 30
    assert literature_tool.metadata["reranker_k"] == 1
    assert literature_tool.metadata["use_reranker"] is False


@pytest.mark.parametrize(
    "tool_call,has_search_size,tool_env_name,expected_tool_class",
    (
        [get_morpho_tool, True, "MORPHO", GetMorphoTool],
        [get_kg_morpho_feature_tool, True, "KG_MORPHO_FEATURES", KGMorphoFeatureTool],
        [get_traces_tool, True, "TRACE", GetTracesTool],
        [get_electrophys_feature_tool, False, None, ElectrophysFeatureTool],
        [get_morphology_feature_tool, False, None, MorphologyFeatureTool],
        [get_me_model_tool, True, "ME_MODEL", GetMEModelTool],
    ),
)
def test_get_tool(
    tool_call,
    has_search_size,
    tool_env_name,
    expected_tool_class,
    monkeypatch,
    patch_required_env,
):
    url = "https://fake_url/api/nexus/v1/search/query/"
    token = "fake_token"

    httpx_client = AsyncClient()
    settings = Settings()

    tool = tool_call(settings=settings, token=token, httpx_client=httpx_client)
    assert isinstance(tool, expected_tool_class)
    assert tool.metadata["url"] == url
    assert tool.metadata["token"] == "fake_token"

    if has_search_size:
        monkeypatch.setenv(f"NEUROAGENT_TOOLS__{tool_env_name}__SEARCH_SIZE", "100")
        settings = Settings()

        tool = tool_call(settings=settings, token=token, httpx_client=httpx_client)
        assert isinstance(tool, expected_tool_class)
        assert tool.metadata["url"] == url
        assert tool.metadata["search_size"] == 100


@pytest.mark.asyncio
async def test_get_memory(patch_required_env, db_connection):
    conn_string = await anext(get_agent_memory(None))

    assert conn_string is None

    conn_string = await anext(get_agent_memory(db_connection))

    if db_connection.startswith("sqlite"):
        assert isinstance(conn_string, AsyncSqliteSaver)
    if db_connection.startswith("postgresql"):
        assert isinstance(conn_string, AsyncPostgresSaver)
    await conn_string.conn.close()  # Needs to be re-closed for some reasons.


def test_language_model(monkeypatch, patch_required_env):
    monkeypatch.setenv("NEUROAGENT_OPENAI__MODEL", "dummy")
    monkeypatch.setenv("NEUROAGENT_OPENAI__TEMPERATURE", "99")
    monkeypatch.setenv("NEUROAGENT_OPENAI__MAX_TOKENS", "99")

    settings = Settings()

    language_model = get_language_model(settings)

    assert isinstance(language_model, ChatOpenAI)
    assert language_model.model_name == "dummy"
    assert language_model.temperature == 99
    assert language_model.max_tokens == 99


def test_thread_to_vp(patch_required_env, db_connection):
    # Setup DB with one thread to do the tests
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    engine = get_engine(test_settings, db_connection)
    session = next(get_session(engine))
    user_id = "Super_user"
    # create test thread
    Base.metadata.create_all(bind=engine)
    new_thread = Threads(
        user_sub=user_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    session.commit()
    session.refresh(new_thread)

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
    ids = thread_to_vp(user_id, session, good_request_headers)
    assert ids == {"vlab_id": "test_vlab", "project_id": "test_project"}

    # Test with no infos in headers.
    bad_request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("example.com", 80),
            "path": "/test/test_thread_id",
            "headers": [
                (b"wong_header", b"wrong value"),
            ],
        }
    )
    with pytest.raises(HTTPException) as error:
        _ = thread_to_vp(user_id, session, bad_request)
    assert error.value.detail == "thread not found when trying to validate project ID."

    # Test with no infos in headers, but valid thread_ID.
    good_request_DB = Request(
        scope={
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("example.com", 80),
            "path": f"/test/{new_thread.thread_id}",
            "headers": [
                (b"wong_header", b"wrong value"),
            ],
        }
    )
    ids_from_DB = thread_to_vp(user_id, session, good_request_DB)
    assert ids_from_DB == {"vlab_id": "test_vlab_DB", "project_id": "project_id_DB"}

    # don't forget to close the session, otherwise the tests hangs.
    session.close()


@pytest.mark.asyncio
async def test_validate_project(patch_required_env, httpx_mock):
    test_settings = Settings()
    httpx_client = AsyncClient()
    token = "fake_token"
    test_vp = {"vlab_id": "test_vlab_DB", "project_id": "project_id_DB"}

    # test with bad config
    httpx_mock.add_response(
        url=f'{test_settings.virtual_lab.get_project_url}/{test_vp["vlab_id"]}/projects/{test_vp["project_id"]}',
        status_code=404,
    )
    with pytest.raises(HTTPException) as error:
        _ = await validate_project(httpx_client, test_vp, token, test_settings)
    assert error.value.status_code == 401

    # test with good config
    httpx_mock.add_response(
        url=f'{test_settings.virtual_lab.get_project_url}/{test_vp["vlab_id"]}/projects/{test_vp["project_id"]}',
        json="test_project_ID",
    )
    project_id = await validate_project(httpx_client, test_vp, token, test_settings)
    assert project_id == test_vp["project_id"]


@pytest.mark.asyncio
async def test_get_agent(monkeypatch, httpx_mock, patch_required_env):
    monkeypatch.setenv("NEUROAGENT_AGENT__MODEL", "simple")
    token = "fake_token"
    httpx_client = AsyncClient()
    settings = Settings()

    thread_to_vp = {"vlab_id": "test_vlab", "project_id": "test_project"}
    httpx_mock.add_response(
        url=f'{settings.virtual_lab.get_project_url}/{thread_to_vp["vlab_id"]}/projects/{thread_to_vp["project_id"]}',
        json="test_project_ID",
    )
    valid_project = await validate_project(
        httpx_client=httpx_client,
        thread_to_vp=thread_to_vp,
        token=token,
        settings=settings,
    )

    language_model = get_language_model(settings)
    bluenaas_tool = get_bluenaas_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    literature_tool = get_literature_tool(
        token=token, settings=settings, httpx_client=httpx_client
    )
    morpho_tool = get_morpho_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    morphology_feature_tool = get_morphology_feature_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    kg_morpho_feature_tool = get_kg_morpho_feature_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    electrophys_feature_tool = get_electrophys_feature_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    traces_tool = get_traces_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    br_resolver_tool = get_brain_region_resolver_tool(
        token=token,
        httpx_client=httpx_client,
        settings=settings,
    )
    me_model_tool = get_me_model_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )

    agent = get_agent(
        valid_project,
        llm=language_model,
        bluenaas_tool=bluenaas_tool,
        literature_tool=literature_tool,
        br_resolver_tool=br_resolver_tool,
        morpho_tool=morpho_tool,
        morphology_feature_tool=morphology_feature_tool,
        kg_morpho_feature_tool=kg_morpho_feature_tool,
        electrophys_feature_tool=electrophys_feature_tool,
        traces_tool=traces_tool,
        settings=settings,
        me_model_tool=me_model_tool,
    )

    assert isinstance(agent, SimpleAgent)


@pytest.mark.asyncio
async def test_get_chat_agent(
    monkeypatch, db_connection, httpx_mock, patch_required_env
):
    monkeypatch.setenv("NEUROAGENT_DB__PREFIX", "sqlite://")

    token = "fake_token"
    httpx_client = AsyncClient()
    settings = Settings()

    thread_to_vp = {"vlab_id": "test_vlab", "project_id": "test_project"}
    httpx_mock.add_response(
        url=f'{settings.virtual_lab.get_project_url}/{thread_to_vp["vlab_id"]}/projects/{thread_to_vp["project_id"]}',
        json="test_project_ID",
    )
    valid_project = await validate_project(
        httpx_client=httpx_client,
        thread_to_vp=thread_to_vp,
        token=token,
        settings=settings,
    )

    language_model = get_language_model(settings)
    bluenaas_tool = get_bluenaas_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    literature_tool = get_literature_tool(
        token=token, settings=settings, httpx_client=httpx_client
    )
    morpho_tool = get_morpho_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    morphology_feature_tool = get_morphology_feature_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    kg_morpho_feature_tool = get_kg_morpho_feature_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    electrophys_feature_tool = get_electrophys_feature_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    traces_tool = get_traces_tool(
        settings=settings, token=token, httpx_client=httpx_client
    )
    br_resolver_tool = get_brain_region_resolver_tool(
        token=token,
        httpx_client=httpx_client,
        settings=settings,
    )

    memory = await anext(get_agent_memory(db_connection))

    agent = get_chat_agent(
        valid_project,
        llm=language_model,
        bluenaas_tool=bluenaas_tool,
        literature_tool=literature_tool,
        br_resolver_tool=br_resolver_tool,
        morpho_tool=morpho_tool,
        morphology_feature_tool=morphology_feature_tool,
        kg_morpho_feature_tool=kg_morpho_feature_tool,
        electrophys_feature_tool=electrophys_feature_tool,
        traces_tool=traces_tool,
        memory=memory,
    )

    assert isinstance(agent, SimpleChatAgent)
    await memory.conn.close()  # Needs to be re-closed for some reasons.


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
