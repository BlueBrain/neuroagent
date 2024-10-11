"""App utilities functions."""

from fastapi import HTTPException
from httpx import AsyncClient
from starlette.status import HTTP_401_UNAUTHORIZED


async def validate_project(
    httpx_client: AsyncClient,
    vlab_and_project: dict[str, str],
    token: str,
    vlab_project_url: str,
) -> None:
    """Check user appartenance to vlab and project before running agent."""
    response = await httpx_client.get(
        f'{vlab_project_url}/{vlab_and_project["vlab_id"]}/projects/{vlab_and_project["project_id"]}',
        headers={"Authorization": f"Bearer {token}"},
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="User does not belong to the project.",
        )
