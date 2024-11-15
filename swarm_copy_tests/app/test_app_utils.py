"""Test app utils."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.exceptions import HTTPException
from httpx import AsyncClient

from swarm_copy.app.app_utils import setup_engine, validate_project
from swarm_copy.app.config import Settings


@pytest.mark.asyncio
async def test_validate_project(patch_required_env, httpx_mock, monkeypatch):
    monkeypatch.setenv("NEUROAGENT_KEYCLOAK__VALIDATE_TOKEN", "true")
    httpx_client = AsyncClient()
    token = "fake_token"
    test_vp = {"vlab_id": "test_vlab_DB", "project_id": "project_id_DB"}
    vlab_url = "https://openbluebrain.com/api/virtual-lab-manager/virtual-labs"

    # test with bad config
    httpx_mock.add_response(
        url=f'{vlab_url}/{test_vp["vlab_id"]}/projects/{test_vp["project_id"]}',
        status_code=404,
    )
    with pytest.raises(HTTPException) as error:
        await validate_project(
            httpx_client=httpx_client,
            vlab_id=test_vp["vlab_id"],
            project_id=test_vp["project_id"],
            token=token,
            vlab_project_url=vlab_url,
        )
    assert error.value.status_code == 401

    # test with good config
    httpx_mock.add_response(
        url=f'{vlab_url}/{test_vp["vlab_id"]}/projects/{test_vp["project_id"]}',
        json="test_project_ID",
    )
    await validate_project(
        httpx_client=httpx_client,
        vlab_id=test_vp["vlab_id"],
        project_id=test_vp["project_id"],
        token=token,
        vlab_project_url=vlab_url,
    )
    # we jsut want to assert that the httpx_mock was called.


@patch("neuroagent.app.app_utils.create_async_engine")
def test_setup_engine(create_engine_mock, monkeypatch, patch_required_env):
    create_engine_mock.return_value = AsyncMock()

    monkeypatch.setenv("NEUROAGENT_DB__PREFIX", "prefix")

    settings = Settings()

    connection_string = "postgresql+asyncpg://user:password@localhost/dbname"
    retval = setup_engine(settings=settings, connection_string=connection_string)
    assert retval is not None


@patch("neuroagent.app.app_utils.create_async_engine")
def test_setup_engine_no_connection_string(
    create_engine_mock, monkeypatch, patch_required_env
):
    create_engine_mock.return_value = AsyncMock()

    monkeypatch.setenv("NEUROAGENT_DB__PREFIX", "prefix")

    settings = Settings()

    retval = setup_engine(settings=settings, connection_string=None)
    assert retval is None
