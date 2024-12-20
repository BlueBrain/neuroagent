from unittest.mock import AsyncMock, Mock

import pytest

from neuroagent.app.config import Settings
from neuroagent.app.dependencies import (
    get_agents_routine,
    get_settings,
    get_starting_agent,
)
from neuroagent.app.main import app
from neuroagent.app.routers import qa
from neuroagent.new_types import Agent, Response


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_run_agent(app_client, patch_required_env, httpx_mock):
    agent_output = Response(
        messages=[
            {"role": "user", "content": "Hello"},
            {"content": "Hello! How can I assist you today?"},
        ],
        agent=Agent(
            name="Agent",
            model="gpt-4o-mini",
            instructions="You are a helpfull assistant",
        ),
    )
    agent_routine = AsyncMock()
    agent_routine.arun.return_value = agent_output
    mock_agent = Agent()

    test_settings = Settings()
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_agents_routine] = lambda: agent_routine
    app.dependency_overrides[get_starting_agent] = lambda: mock_agent

    with app_client as app_client:
        response = app_client.post("/qa/run", json={"query": "This is my query"})

        assert response.status_code == 200
        assert (
            response.json()["message"]
            == agent_output.model_dump()["messages"][-1]["content"]
        )

        # Missing query
        response = app_client.post("/qa/run", json={})
        assert response.status_code == 422


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_run_chat_agent(app_client, httpx_mock, patch_required_env, db_connection):
    agent_output = Response(
        messages=[
            {"role": "user", "content": "Hello"},
            {"content": "Hello! How can I assist you today?", "role": "assistant"},
        ],
        agent=Agent(
            name="Agent",
            model="gpt-4o-mini",
            instructions="You are a helpfull assistant",
        ),
    )
    agent_routine = AsyncMock()
    agent_routine.arun.return_value = agent_output

    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_agents_routine] = lambda: agent_routine
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
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )
        assert response.status_code == 200
        assert (
            response.json()["message"]
            == agent_output.model_dump()["messages"][-1]["content"]
        )

        # Missing thread_id query
        response = app_client.post(
            "/qa/chat",
            json={"query": "This is my query"},
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )
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


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_chat_streamed(app_client, httpx_mock, patch_required_env, db_connection):
    """Test the generative QA endpoint with a fake LLM."""
    qa.stream_agent_response = Mock()
    qa.stream_agent_response.return_value = streamed_response()

    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    agent_routine = Mock()
    app.dependency_overrides[get_agents_routine] = lambda: agent_routine

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
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )
    assert response.status_code == 200
    assert response.content == expected_tokens
