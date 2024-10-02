"""Test config"""

import pytest
from pydantic import ValidationError

from neuroagent.app.config import Settings


def test_required(monkeypatch, patch_required_env):
    settings = Settings()

    assert settings.tools.literature.url == "https://fake_url"
    assert settings.knowledge_graph.base_url == "https://fake_url/api/nexus/v1"
    assert settings.openai.token.get_secret_value() == "dummy"
    assert settings.knowledge_graph.use_token
    assert settings.knowledge_graph.token.get_secret_value() == "token"

    # make sure not case sensitive
    monkeypatch.delenv("NEUROAGENT_TOOLS__LITERATURE__URL")
    monkeypatch.setenv("neuroagent_tools__literature__URL", "https://new_fake_url")

    settings = Settings()
    assert settings.tools.literature.url == "https://new_fake_url"


def test_no_settings():
    # We get an error when no custom variables provided
    with pytest.raises(ValidationError):
        Settings()


def test_setup_tools(monkeypatch, patch_required_env):
    monkeypatch.setenv("NEUROAGENT_TOOLS__TRACE__SEARCH_SIZE", "20")
    monkeypatch.setenv("NEUROAGENT_TOOLS__MORPHO__SEARCH_SIZE", "20")
    monkeypatch.setenv("NEUROAGENT_TOOLS__KG_MORPHO_FEATURES__SEARCH_SIZE", "20")

    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__USERNAME", "user")
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__PASSWORD", "pass")

    settings = Settings()

    assert settings.tools.morpho.search_size == 20
    assert settings.tools.trace.search_size == 20
    assert settings.tools.kg_morpho_features.search_size == 20
    assert settings.keycloak.username == "user"
    assert settings.keycloak.password.get_secret_value() == "pass"
    assert settings.knowledge_graph.use_token
    assert settings.knowledge_graph.token.get_secret_value() == "token"


def test_check_consistency(monkeypatch):
    # We get an error when no custom variables provided
    url = "https://fake_url"
    monkeypatch.setenv("NEUROAGENT_TOOLS__LITERATURE__URL", url)
    monkeypatch.setenv("NEUROAGENT_KNOWLEDGE_GRAPH__URL", url)

    with pytest.raises(ValueError):
        Settings()

    monkeypatch.setenv("NEUROAGENT_GENERATIVE__OPENAI__TOKEN", "dummy")
    monkeypatch.setenv("NEUROAGENT_KNOWLEDGE_GRAPH__USE_TOKEN", "true")

    with pytest.raises(ValueError):
        Settings()

    monkeypatch.setenv("NEUROAGENT_KNOWLEDGE_GRAPH__USE_TOKEN", "false")

    with pytest.raises(ValueError):
        Settings()

    monkeypatch.setenv("NEUROAGENT_KNOWLEDGE_GRAPH__TOKEN", "Hello")

    with pytest.raises(ValueError):
        Settings()
