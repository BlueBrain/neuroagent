"""Test app utils."""

import pytest
from fastapi.exceptions import HTTPException
from httpx import AsyncClient

from neuroagent.app.app_utils import validate_project


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
        await validate_project(httpx_client, test_vp, token, vlab_url)
    assert error.value.status_code == 401

    # test with good config
    httpx_mock.add_response(
        url=f'{vlab_url}/{test_vp["vlab_id"]}/projects/{test_vp["project_id"]}',
        json="test_project_ID",
    )
    await validate_project(httpx_client, test_vp, token, vlab_url)
    # we jsut want to assert that the httpx_mock was called.
