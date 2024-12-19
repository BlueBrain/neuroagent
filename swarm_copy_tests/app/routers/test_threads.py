import pytest

from swarm_copy.agent_routine import Agent, AgentsRoutine
from swarm_copy.app.config import Settings
from swarm_copy.app.dependencies import (
    get_agents_routine,
    get_settings,
    get_starting_agent,
)
from swarm_copy.app.main import app
from swarm_copy_tests.mock_client import create_mock_response


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_create_thread(patch_required_env, httpx_mock, app_client, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    with app_client as app_client:
        # Create a thread
        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
    assert create_output["thread_id"]
    assert create_output["title"] == "New chat"
    assert create_output["vlab_id"] == "test_vlab"
    assert create_output["project_id"] == "test_project"


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_get_threads(patch_required_env, httpx_mock, app_client, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    with app_client as app_client:
        threads = app_client.get("/threads/").json()
        assert not threads
        create_output_1 = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        create_output_2 = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        threads = app_client.get("/threads/").json()

    assert len(threads) == 2
    assert threads[0] == create_output_1
    assert threads[1] == create_output_2


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
@pytest.mark.asyncio
async def test_get_messages(
    patch_required_env,
    httpx_mock,
    app_client,
    db_connection,
    mock_openai_client,
    get_weather_tool,
):
    # Put data in the db
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
        # wrong thread ID
        wrong_response = app_client.get("/threads/test")
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
            headers={"x-virtual-lab-id": "test_vlab", "x-project-id": "test_project"},
        )

        create_output = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        empty_thread_id = create_output["thread_id"]
        empty_messages = app_client.get(f"/threads/{empty_thread_id}").json()
        assert empty_messages == []

        # Get the messages of the thread
        messages = app_client.get(f"/threads/{thread_id}").json()

    assert messages[0]["order"] == 0
    assert messages[0]["entity"] == "user"
    assert messages[0]["msg_content"] == "This is my query"
    assert messages[0]["message_id"]
    assert messages[0]["creation_date"]

    assert messages[1]["order"] == 3
    assert messages[1]["entity"] == "ai_message"
    assert messages[1]["msg_content"] == "sample response content"
    assert messages[1]["message_id"]
    assert messages[1]["creation_date"]


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_update_thread_title(patch_required_env, httpx_mock, app_client, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings

    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    with app_client as app_client:
        threads = app_client.get("/threads/").json()
        assert not threads

        # Check when wrong thread id
        wrong_response = app_client.patch(
            "/threads/wrong_id", json={"title": "great_title"}
        )
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        create_thread_response = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_thread_response["thread_id"]

        updated_title = "Updated Thread Title"
        update_response = app_client.patch(
            f"/threads/{thread_id}", json={"title": updated_title}
        ).json()

        assert update_response["title"] == updated_title


@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
def test_delete_thread(patch_required_env, httpx_mock, app_client, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings

    httpx_mock.add_response(
        url=f"{test_settings.virtual_lab.get_project_url}/test_vlab/projects/test_project"
    )
    with app_client as app_client:
        threads = app_client.get("/threads/").json()
        assert not threads

        # Check when wrong thread id
        wrong_response = app_client.delete("/threads/wrong_id")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        create_thread_response = app_client.post(
            "/threads/?virtual_lab_id=test_vlab&project_id=test_project"
        ).json()
        thread_id = create_thread_response["thread_id"]

        threads = app_client.get("/threads/").json()
        assert len(threads) == 1
        assert threads[0]["thread_id"] == thread_id

        delete_response = app_client.delete(f"/threads/{thread_id}").json()
        assert delete_response["Acknowledged"] == "true"

        threads = app_client.get("/threads/").json()
        assert not threads
