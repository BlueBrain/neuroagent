"""Test of the tool router."""

import json

import pytest

from neuroagent.agent_routine import Agent, AgentsRoutine
from neuroagent.app.config import Settings
from neuroagent.app.database.schemas import ToolCallSchema
from neuroagent.app.dependencies import (
    get_agents_routine,
    get_context_variables,
    get_settings,
    get_starting_agent,
)
from neuroagent.app.main import app
from tests.mock_client import create_mock_response


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.asyncio
async def test_get_tool_calls(
    patch_required_env,
    httpx_mock,
    app_client,
    db_connection,
    mock_openai_client,
    get_weather_tool,
):
    routine = AgentsRoutine(client=mock_openai_client)

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[
                    {"name": "get_weather", "args": {"location": "Geneva"}}
                ],
            ),
            create_mock_response(
                {"role": "assistant", "content": "sample response content"}
            ),
        ]
    )
    agent = Agent(tools=[get_weather_tool])

    app.dependency_overrides[get_agents_routine] = lambda: routine
    app.dependency_overrides[get_starting_agent] = lambda: agent
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )

    with app_client as app_client:
        wrong_response = app_client.get("/tools/test/1234")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_output["thread_id"]

        # Fill the thread
        app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
            params={"thread_id": thread_id},
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )

        tool_calls = app_client.get(f"/tools/{thread_id}/wrong_id")
        assert tool_calls.status_code == 404
        assert tool_calls.json() == {"detail": {"detail": "Message not found."}}

        # Get the messages of the thread
        messages = app_client.get(f"/threads/{thread_id}").json()
        message_id = messages[-1]["message_id"]
        tool_calls = app_client.get(f"/tools/{thread_id}/{message_id}").json()

    assert (
        tool_calls[0]
        == ToolCallSchema(
            tool_call_id="mock_tc_id",
            name="get_weather",
            arguments={"location": "Geneva"},
        ).model_dump()
    )


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.asyncio
async def test_get_tool_output(
    patch_required_env,
    app_client,
    httpx_mock,
    db_connection,
    mock_openai_client,
    agent_handoff_tool,
):
    routine = AgentsRoutine(client=mock_openai_client)

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "agent_handoff_tool", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "sample response content"}
            ),
        ]
    )
    agent_1 = Agent(name="Test agent 1", tools=[agent_handoff_tool])
    agent_2 = Agent(name="Test agent 2", tools=[])

    app.dependency_overrides[get_agents_routine] = lambda: routine
    app.dependency_overrides[get_starting_agent] = lambda: agent_1
    app.dependency_overrides[get_context_variables] = lambda: {"to_agent": agent_2}
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )

    with app_client as app_client:
        wrong_response = app_client.get("/tools/output/test/123")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_output["thread_id"]

        # Fill the thread
        app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
            params={"thread_id": thread_id},
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )

        tool_output = app_client.get(f"/tools/output/{thread_id}/123")
        assert tool_output.status_code == 200
        assert tool_output.json() == []

        # Get the messages of the thread
        messages = app_client.get(f"/threads/{thread_id}").json()
        message_id = messages[-1]["message_id"]
        tool_calls = app_client.get(f"/tools/{thread_id}/{message_id}").json()

        tool_call_id = tool_calls[0]["tool_call_id"]
        tool_output = app_client.get(f"/tools/output/{thread_id}/{tool_call_id}")

    assert tool_output.json() == [json.dumps({"assistant": agent_2.name})]


@pytest.mark.asyncio
async def test_get_required_validation(
    patch_required_env,
    app_client,
    httpx_mock,
    db_connection,
    mock_openai_client,
    agent_handoff_tool,
):
    routine = AgentsRoutine(client=mock_openai_client)

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "agent_handoff_tool", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "sample response content"}
            ),
        ]
    )
    agent_handoff_tool.hil = True
    agent_1 = Agent(name="Test agent 1", tools=[agent_handoff_tool])
    agent_2 = Agent(name="Test agent 2", tools=[])

    app.dependency_overrides[get_agents_routine] = lambda: routine
    app.dependency_overrides[get_starting_agent] = lambda: agent_1
    app.dependency_overrides[get_context_variables] = lambda: {"to_agent": agent_2}
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )

    with app_client as app_client:
        wrong_response = app_client.get("/tools/validation/test/")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_output["thread_id"]
        validation_list = app_client.get(f"/tools/validation/{thread_id}/")
        assert validation_list.json() == []
        # Fill the thread
        app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
            params={"thread_id": thread_id},
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )

        validation_list = app_client.get(f"/tools/validation/{thread_id}/")
        assert validation_list.status_code == 200
        assert validation_list.json() == [
            {
                "message": "Please validate the following inputs before proceeding.",
                "name": "agent_handoff_tool",
                "inputs": {},
                "tool_call_id": "mock_tc_id",
            }
        ]

        # Validate the tool call
        app_client.patch(
            f"/tools/validation/{thread_id}/{validation_list.json()[0]['tool_call_id']}",
            json={"is_validated": True},
        )
        # Validation list should now be empty
        validation_list = app_client.get(f"/tools/validation/{thread_id}/")
        assert validation_list.json() == []


async def test_validate_input(
    patch_required_env,
    app_client,
    httpx_mock,
    db_connection,
    mock_openai_client,
    agent_handoff_tool,
):
    routine = AgentsRoutine(client=mock_openai_client)

    mock_openai_client.set_sequential_responses(
        [
            create_mock_response(
                message={"role": "assistant", "content": ""},
                function_calls=[{"name": "agent_handoff_tool", "args": {}}],
            ),
            create_mock_response(
                {"role": "assistant", "content": "sample response content"}
            ),
        ]
    )
    agent_handoff_tool.hil = True
    agent_1 = Agent(name="Test agent 1", tools=[agent_handoff_tool])
    agent_2 = Agent(name="Test agent 2", tools=[])

    app.dependency_overrides[get_agents_routine] = lambda: routine
    app.dependency_overrides[get_starting_agent] = lambda: agent_1
    app.dependency_overrides[get_context_variables] = lambda: {"to_agent": agent_2}
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )

    with app_client as app_client:
        wrong_response = app_client.get("/tools/validation/test/123")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_output["thread_id"]

        # Fill the thread
        response = app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
            params={"thread_id": thread_id},
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )

        assert response.status_code == 200
        assert response.json() == [
            {
                "message": "Please validate the following inputs before proceeding.",
                "name": "agent_handoff_tool",
                "inputs": {},
                "tool_call_id": "mock_tc_id",
            }
        ]

        # Validate the tool call
        to_validate_list = response.json()
        validated = app_client.patch(
            f"/tools/validation/{thread_id}/{to_validate_list[0]['tool_call_id']}",
            json={"is_validated": True},
        )
        assert validated.status_code == 200
        assert validated.json() == {
            "tool_call_id": to_validate_list[0]["tool_call_id"],
            "name": to_validate_list[0]["name"],
            "arguments": to_validate_list[0]["tool_call_id"],
        }

        # Check that is has been validated and cannot be validated anymore
        validated = app_client.patch(
            f"/tools/validation/{thread_id}/{to_validate_list[0]['tool_call_id']}",
            json={"is_validated": True},
        )
        assert validated.status_code == 403
        assert validated.content == b"The tool call has already been validated."
