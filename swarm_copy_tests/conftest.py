"""Test configuration."""

import json
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from swarm_copy.app.config import Settings
from swarm_copy.app.dependencies import get_kg_token, get_settings
from swarm_copy.app.main import app


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

