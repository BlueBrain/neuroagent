"""Test configuration."""

import json
from pathlib import Path
from typing import ClassVar

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from swarm_copy.app.config import Settings
from swarm_copy.app.dependencies import Agent, get_kg_token, get_settings
from swarm_copy.app.main import app
from swarm_copy.tools.base_tool import BaseTool
from swarm_copy_tests.mock_client import MockOpenAIClient, create_mock_response


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

@pytest.fixture
def mock_openai_client():
    """Fake openai client."""
    m = MockOpenAIClient()
    m.set_response(
        create_mock_response(
            {"role": "assistant", "content": "sample response content"}
        )
    )
    return m


@pytest.fixture(name="get_weather_tool")
def fake_tool():
    """Fake get weather tool."""

    class FakeToolInput(BaseModel):
        location: str

    class FakeToolMetadata(
        BaseModel
    ):  # Should be a BaseMetadata but we don't want httpx client here
        model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
        planet: str | None = None

    class FakeTool(BaseTool):
        name: ClassVar[str] = "get_weather"
        description: ClassVar[str] = "Great description"
        metadata: FakeToolMetadata
        input_schema: FakeToolInput

        async def arun(self):
            if self.metadata.planet:
                return f"It's sunny today in {self.input_schema.location} from planet {self.metadata.planet}."
            return "It's sunny today."

    return FakeTool


@pytest.fixture
def agent_handoff_tool():
    """Fake agent handoff tool."""

    class HandoffToolInput(BaseModel):
        pass

    class HandoffToolMetadata(
        BaseModel
    ):  # Should be a BaseMetadata but we don't want httpx client here
        to_agent: Agent
        model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    class HandoffTool(BaseTool):
        name: ClassVar[str] = "agent_handoff_tool"
        description: ClassVar[str] = "Handoff to another agent."
        metadata: HandoffToolMetadata
        input_schema: HandoffToolInput

        async def arun(self):
            return self.metadata.to_agent

    return HandoffTool

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


@pytest_asyncio.fixture(params=["sqlite", "postgresql"], name="db_connection")
async def setup_sql_db(request, tmp_path):
    db_type = request.param

    # To start the postgresql database:
    # docker run -it --rm -p 5432:5432 -e POSTGRES_USER=test -e POSTGRES_PASSWORD=password postgres:latest
    path = (
        f"sqlite+aiosqlite:///{tmp_path / 'test_db.db'}"
        if db_type == "sqlite"
        else "postgresql+asyncpg://test:password@localhost:5432"
    )
    if db_type == "postgresql":
        try:
            async with create_async_engine(path).connect() as conn:
                pass
        except Exception:
            pytest.skip("Postgres database not connected")
    yield path
    if db_type == "postgresql":
        metadata = MetaData()
        engine = create_async_engine(path)
        session = AsyncSession(bind=engine)
        async with engine.begin() as conn:
            await conn.run_sync(metadata.reflect)
            await conn.run_sync(metadata.drop_all)

        await session.commit()
        await engine.dispose()
        await session.aclose()


@pytest.fixture
def get_resolve_query_output():
    with open("tests/data/resolve_query.json") as f:
        outputs = json.loads(f.read())
    return outputs


@pytest.fixture
def brain_region_json_path():
    br_path = Path(__file__).parent / "data" / "brainregion_hierarchy.json"
    return br_path


@pytest.fixture(name="settings")
def settings():
    return Settings(
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
