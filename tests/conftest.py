"""Test configuration."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import MetaData, create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from neuroagent.app.config import Settings
from neuroagent.app.dependencies import get_kg_token, get_settings
from neuroagent.app.main import app
from neuroagent.tools import GetMorphoTool


@pytest.fixture(name="app_client")
def client_fixture():
    """Get client and clear app dependency_overrides."""
    app_client = TestClient(app)
    test_settings = Settings(
        tools={
            "literature": {
                "url": "fake_literature_url",
            },
        },
        knowledge_graph={
            "base_url": "https://fake_url/api/nexus/v1",
        },
        openai={
            "token": "fake_token",
        },
        keycloak={
            "username": "fake_username",
            "password": "fake_password",
        },
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    # mock keycloak authentication
    app.dependency_overrides[get_kg_token] = lambda: "fake_token"
    yield app_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True, scope="session")
def dont_look_at_env_file():
    """Never look inside of the .env when running unit tests."""
    Settings.model_config["env_file"] = None


@pytest.fixture()
def patch_required_env(monkeypatch):
    monkeypatch.setenv("NEUROAGENT_TOOLS__LITERATURE__URL", "https://fake_url")
    monkeypatch.setenv(
        "NEUROAGENT_KNOWLEDGE_GRAPH__BASE_URL", "https://fake_url/api/nexus/v1"
    )
    monkeypatch.setenv("NEUROAGENT_OPENAI__TOKEN", "dummy")
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__VALIDATE_TOKEN", "False")
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__PASSWORD", "password")


@pytest.fixture(params=["sqlite", "postgresql"], name="db_connection")
def setup_sql_db(request, tmp_path):
    db_type = request.param

    # To start the postgresql database:
    # docker run -it --rm -p 5432:5432 -e POSTGRES_USER=test -e POSTGRES_PASSWORD=password postgres:latest
    path = (
        f"sqlite:///{tmp_path / 'test_db.db'}"
        if db_type == "sqlite"
        else "postgresql://test:password@localhost:5432"
    )
    if db_type == "postgresql":
        try:
            engine = create_engine(path).connect()
        except OperationalError:
            pytest.skip("Postgres database not connected")
    yield path
    if db_type == "postgresql":
        metadata = MetaData()
        engine = create_engine(path)
        session = Session(bind=engine)

        metadata.reflect(engine)
        metadata.drop_all(bind=engine)
        session.commit()


@pytest.fixture
def get_resolve_query_output():
    with open("tests/data/resolve_query.json") as f:
        outputs = json.loads(f.read())
    return outputs


@pytest.fixture
def brain_region_json_path():
    br_path = Path(__file__).parent / "data" / "brainregion_hierarchy.json"
    return br_path


@pytest.fixture
async def fake_llm_with_tools(brain_region_json_path):
    class FakeFuntionChatModel(GenericFakeChatModel):
        def bind_tools(self, functions: list):
            return self

    # If you need another fake response to use different tools,
    # you can do in your test
    # ```python
    # llm, _ = await anext(fake_llm_with_tools)
    # llm.responses = my_fake_responses
    # ```
    # and simply bind the corresponding tools
    fake_responses = [
        AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_zHhwfNLSvGGHXMoILdIYtDVI",
                        "function": {
                            "arguments": '{"brain_region_id":"http://api.brain-map.org/api/v2/data/Structure/549"}',
                            "name": "get-morpho-tool",
                        },
                        "type": "function",
                    }
                ]
            },
            response_metadata={"finish_reason": "tool_calls"},
            id="run-3828644d-197b-401b-8634-e6ecf01c2e7c-0",
            tool_calls=[
                {
                    "name": "get-morpho-tool",
                    "args": {
                        "brain_region_id": (
                            "http://api.brain-map.org/api/v2/data/Structure/549"
                        )
                    },
                    "id": "call_zHhwfNLSvGGHXMoILdIYtDVI",
                }
            ],
        ),
        AIMessage(
            content="Great answer",
            response_metadata={"finish_reason": "stop"},
            id="run-42768b30-044a-4263-8c5c-da61429aa9da-0",
        ),
    ]

    # If you use this tool in your test, DO NOT FORGET to mock the url response with the following snippet:
    #
    # ```python
    # json_path = Path(__file__).resolve().parent.parent / "data" / "knowledge_graph.json"
    # with open(json_path) as f:
    #     knowledge_graph_response = json.load(f)

    # httpx_mock.add_response(
    #     url="http://fake_url",
    #     json=knowledge_graph_response,
    # )
    # ```
    # The http call is not mocked here because one might want to change the responses
    # and the tools used.
    async_client = AsyncClient()
    tool = GetMorphoTool(
        metadata={
            "url": "http://fake_url",
            "search_size": 2,
            "httpx_client": async_client,
            "token": "fake_token",
            "brainregion_path": brain_region_json_path,
        }
    )

    yield FakeFuntionChatModel(messages=iter(fake_responses)), [tool], fake_responses
    await async_client.aclose()
