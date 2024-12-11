import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from swarm_copy.app.config import Settings
from swarm_copy.app.dependencies import get_settings
from swarm_copy.app.main import app


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


@pytest.fixture(autouse=True)
def stop_patches():
    yield
    patch.stopall()
