"""Test dependencies."""

import json
import os
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import Mock, patch

import pytest
from httpx import AsyncClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Secret
from sqlalchemy.exc import SQLAlchemyError

from neuroagent.agents import SimpleAgent, SimpleChatAgent
from neuroagent.app.dependencies import (
    Settings,
    get_agent,
    get_agent_memory,
    get_brain_region_resolver_tool,
    get_cell_types_kg_hierarchy,
    get_chat_agent,
    get_electrophys_feature_tool,
    get_httpx_client,
    get_kg_morpho_feature_tool,
    get_language_model,
    get_literature_tool,
    get_morpho_tool,
    get_morphology_feature_tool,
    get_traces_tool,
    get_update_kg_hierarchy,
    get_user_id, get_settings, get_connection_string, get_engine,
)
from neuroagent.tools import (
    ElectrophysFeatureTool,
    GetMorphoTool,
    GetTracesTool,
    KGMorphoFeatureTool,
    LiteratureSearchTool,
    MorphologyFeatureTool,
)


@patch.dict(os.environ, {
    "NEUROAGENT_TOOLS__LITERATURE__URL": "https://localhost1",
    "NEUROAGENT_KNOWLEDGE_GRAPH__BASE_URL": "https://localhost2",
    "NEUROAGENT_KEYCLOAK__USERNAME": "user2",
    "NEUROAGENT_KEYCLOAK__PASSWORD": "password2"
})
def test_get_settings():
    settings = get_settings()
    assert settings.tools.literature.url == "https://localhost1"
    assert settings.knowledge_graph.url == "https://localhost2/search/query/"


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


def test_get_agent(monkeypatch, patch_required_env):
    monkeypatch.setenv("NEUROAGENT_AGENT__MODEL", "simple")
    token = "fake_token"
    httpx_client = AsyncClient()
    settings = Settings()

    language_model = get_language_model(settings)
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

    agent = get_agent(
        llm=language_model,
        literature_tool=literature_tool,
        br_resolver_tool=br_resolver_tool,
        morpho_tool=morpho_tool,
        morphology_feature_tool=morphology_feature_tool,
        kg_morpho_feature_tool=kg_morpho_feature_tool,
        electrophys_feature_tool=electrophys_feature_tool,
        traces_tool=traces_tool,
        settings=settings,
    )

    assert isinstance(agent, SimpleAgent)


@pytest.mark.asyncio
async def test_get_chat_agent(monkeypatch, db_connection, patch_required_env):
    monkeypatch.setenv("NEUROAGENT_DB__PREFIX", "sqlite://")

    token = "fake_token"
    httpx_client = AsyncClient()
    settings = Settings()

    language_model = get_language_model(settings)
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
        llm=language_model,
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


def fake_get_settings():
    class MockedDb:
        def __init__(self, prefix, user, password, host, port, name):
            self.prefix = prefix
            self.user = user
            self.password = Secret(password)
            self.host = host
            self.port = port
            self.name = name

    class MockedSettings:
        def __init__(self, db):
            self.db = db

    return [
        MockedSettings(MockedDb("http://", "John", "Doe", "localhost", 5000, "test")),
        MockedSettings(MockedDb("", "", "", "", None, None)),
    ]


def test_get_connection_string_full():
    settings = fake_get_settings()[0]
    result = get_connection_string(settings)
    assert (
        result == "http://John:Doe@localhost:5000/test"
    ), "must return fully formed connection string"


def test_get_connection_string_no_prefix():
    settings = fake_get_settings()[1]
    result = get_connection_string(settings)
    assert result is None, "should return None when prefix is not set"


@patch('neuroagent.app.dependencies.create_engine')
def test_get_engine(create_engine_mock):
    create_engine_mock.return_value = Mock()

    settings = Mock()
    settings.db = Mock()
    settings.db.prefix = "prefix"
    settings.db.password = None
    connection_string = "https://localhost"
    retval = get_engine(
        settings=settings,
        connection_string=connection_string
    )
    assert retval is not None


@patch('neuroagent.app.dependencies.create_engine')
def test_get_engine_no_connection_string(create_engine_mock):
    create_engine_mock.return_value = Mock()

    settings = Mock()
    settings.db = Mock()
    settings.db.prefix = "prefix"
    settings.db.password = None
    retval = get_engine(
        settings=settings,
        connection_string=None
    )
    assert retval is None


@patch('neuroagent.app.dependencies.create_engine')
def test_get_engine_error(create_engine_mock):
    create_engine_mock.side_effect = SQLAlchemyError("An error occurred")

    settings = Mock()
    settings.db = Mock()
    settings.db.prefix = "prefix"
    settings.db.password = None
    connection_string = "https://localhost"
    with pytest.raises(SQLAlchemyError):
        get_engine(
            settings=settings,
            connection_string=connection_string
        )
