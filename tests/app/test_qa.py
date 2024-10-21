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

    response = app_client.post("/qa/run", json={"query": "This is my query"})
    assert response.status_code == 200
    assert response.json() == agent_output.model_dump()

    # Missing query
    response = app_client.post("/qa/run", json={})
    assert response.status_code == 422


def test_run_chat_agent(app_client, httpx_mock, patch_required_env, db_connection):
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
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    agent_mock = AsyncMock()
    agent_mock.arun.return_value = agent_output
    app.dependency_overrides[get_chat_agent] = lambda: agent_mock
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    with app_client as app_client:
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        response = app_client.post(
            f"/qa/chat/{create_output['thread_id']}",
            json={"query": "This is my query"},
        )
    assert response.status_code == 200
    assert response.json() == agent_output.model_dump()

    # Missing thread_id query
    response = app_client.post("/qa/chat", json={"query": "This is my query"})
    assert response.status_code == 404


async def streamed_response():
    response = [
        "Calling ",
        "tool ",
        ": ",
        "resolve_entities_tool ",
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


def test_chat_streamed(app_client, httpx_mock, patch_required_env, db_connection):
    """Test the generative QA endpoint with a fake LLM."""
    agent_mock = Mock()
    agent_mock.astream.return_value = streamed_response()
    app.dependency_overrides[get_chat_agent] = lambda: agent_mock

    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    expected_tokens = (
        b"Calling tool : resolve_entities_tool with arguments : {brain_region:"
        b" thalamus}\n This is an amazingly well streamed response. I can't believe how"
        b" good it is!"
    )
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    with app_client as app_client:
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        response = app_client.post(
            f"/qa/chat_streamed/{create_output['thread_id']}",
            json={"query": "This is my query"},
        )
    assert response.status_code == 200
    assert response.content == expected_tokens
