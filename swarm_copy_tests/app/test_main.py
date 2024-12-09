import logging
from unittest.mock import patch

from fastapi.testclient import TestClient

from neuroagent.app.dependencies import get_settings
from neuroagent.app.main import app


def test_settings_endpoint(app_client, dont_look_at_env_file, settings):
    response = app_client.get("/settings")

    replace_secretstr = settings.model_dump()
    replace_secretstr["keycloak"]["password"] = "**********"
    replace_secretstr["openai"]["token"] = "**********"
    assert response.json() == replace_secretstr


def test_readyz(app_client):
    response = app_client.get(
        "/",
    )

    body = response.json()
    assert isinstance(body, dict)
    assert body["status"] == "ok"


def test_lifespan(caplog, monkeypatch, tmp_path, patch_required_env, db_connection):
    get_settings.cache_clear()
    caplog.set_level(logging.INFO)

    monkeypatch.setenv("NEUROAGENT_LOGGING__LEVEL", "info")
    monkeypatch.setenv("NEUROAGENT_LOGGING__EXTERNAL_PACKAGES", "warning")
    monkeypatch.setenv("NEUROAGENT_KNOWLEDGE_GRAPH__DOWNLOAD_HIERARCHY", "true")
    monkeypatch.setenv("NEUROAGENT_DB__PREFIX", db_connection)

    save_path_brainregion = tmp_path / "fake.json"

    async def save_dummy(*args, **kwargs):
        with open(save_path_brainregion, "w") as f:
            f.write("test_text")

    with (
        patch("neuroagent.app.main.get_update_kg_hierarchy", new=save_dummy),
        patch("neuroagent.app.main.get_cell_types_kg_hierarchy", new=save_dummy),
        patch("neuroagent.app.main.get_kg_token", new=lambda *args, **kwargs: "dev"),
    ):
        # The with statement triggers the startup.
        with TestClient(app) as test_client:
            test_client.get("/healthz")
    # check if the brain region dummy file was created.
    assert save_path_brainregion.exists()

    assert caplog.record_tuples[0][::2] == (
        "neuroagent.app.dependencies",
        "Reading the environment and instantiating settings",
    )

    assert (
        logging.getLevelName(logging.getLogger("neuroagent").getEffectiveLevel())
        == "INFO"
    )
    assert (
        logging.getLevelName(logging.getLogger("httpx").getEffectiveLevel())
        == "WARNING"
    )
    assert (
        logging.getLevelName(logging.getLogger("fastapi").getEffectiveLevel())
        == "WARNING"
    )
    assert (
        logging.getLevelName(logging.getLogger("bluepyefe").getEffectiveLevel())
        == "CRITICAL"
    )
