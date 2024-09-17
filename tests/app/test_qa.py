from unittest.mock import AsyncMock, Mock

from neuroagent.agents import AgentOutput, AgentStep
from neuroagent.app.config import Settings
from neuroagent.app.dependencies import get_agent, get_chat_agent, get_settings
from neuroagent.app.main import app


def test_run_agent(app_client):
    agent_output = AgentOutput(
        response="This is my response",
        steps=[
            AgentStep(tool_name="tool1", arguments="covid-19"),
            AgentStep(
                tool_name="tool2",
                arguments={"query": "covid-19", "brain_region": "thalamus"},
            ),
        ],
    )
    agent_mock = AsyncMock()
    agent_mock.arun.return_value = agent_output
    app.dependency_overrides[get_agent] = lambda: agent_mock

    response = app_client.post(
        "/qa/run", json={"inputs": "This is my query", "parameters": {}}
    )
    assert response.status_code == 200
    assert response.json() == agent_output.model_dump()

    # Missing inputs
    response = app_client.post("/qa/run", json={})
    assert response.status_code == 422


def test_run_chat_agent(app_client, tmp_path, patch_required_env):
    agent_output = AgentOutput(
        response="This is my response",
        steps=[
            AgentStep(tool_name="tool1", arguments="covid-19"),
            AgentStep(
                tool_name="tool2",
                arguments={"query": "covid-19", "brain_region": "thalamus"},
            ),
        ],
    )
    p = tmp_path / "test_db.db"
    test_settings = Settings(
        db={"prefix": f"sqlite:///{p}"},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    agent_mock = AsyncMock()
    agent_mock.arun.return_value = agent_output
    app.dependency_overrides[get_chat_agent] = lambda: agent_mock
    with app_client as app_client:
        create_output = app_client.post("/threads/").json()
        response = app_client.post(
            f"/qa/chat/{create_output['thread_id']}",
            json={"inputs": "This is my query", "parameters": {}},
        )
    assert response.status_code == 200
    assert response.json() == agent_output.model_dump()

    # Missing thread_id inputs
    response = app_client.post(
        "/qa/chat", json={"inputs": "This is my query", "parameters": {}}
    )
    assert response.status_code == 404


async def streamed_response():
    response = [
        "Calling ",
        "tool ",
        ": ",
        "resolve_brain_region_tool ",
        "with ",
        "arguments ",
        ": ",
        "{",
        "brain_region",
        ": ",
        "thalamus",
        "}",
        "\n ",
        "This",
        " is",
        " an",
        " amazingly",
        " well",
        " streamed",
        " response",
        ".",
        " I",
        " can",
        "'t",
        " believe",
        " how",
        " good",
        " it",
        " is",
        "!",
    ]
    for word in response:
        yield word


def test_chat_streamed(app_client, tmp_path, patch_required_env):
    """Test the generative QA endpoint with a fake LLM."""
    agent_mock = Mock()
    agent_mock.astream.return_value = streamed_response()
    app.dependency_overrides[get_chat_agent] = lambda: agent_mock
    p = tmp_path / "test_db.db"
    test_settings = Settings(
        db={"prefix": f"sqlite:///{p}"},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    expected_tokens = (
        b"Calling tool : resolve_brain_region_tool with arguments : {brain_region:"
        b" thalamus}\n This is an amazingly well streamed response. I can't believe how"
        b" good it is!"
    )
    with app_client as app_client:
        create_output = app_client.post("/threads/").json()
        response = app_client.post(
            f"/qa/chat_streamed/{create_output['thread_id']}",
            json={"inputs": "This is my query", "parameters": {}},
        )
    assert response.status_code == 200
    assert response.content == expected_tokens
