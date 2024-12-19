"""App utilities functions."""

import logging
from typing import Any

from fastapi import HTTPException
from httpx import AsyncClient
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from starlette.status import HTTP_401_UNAUTHORIZED

from neuroagent.app.config import Settings

logger = logging.getLogger(__name__)


def setup_engine(
    settings: Settings, connection_string: str | None = None
) -> AsyncEngine | None:
    """Get the SQL engine."""
    if connection_string:
        engine_kwargs: dict[str, Any] = {"url": connection_string}
        if "sqlite" in settings.db.prefix:  # type: ignore
            # https://fastapi.tiangolo.com/tutorial/sql-databases/#create-the-sqlalchemy-engine
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        engine = create_async_engine(**engine_kwargs)
    else:
        logger.warning("The SQL db_prefix needs to be set to use the SQL DB.")
        return None
    try:
        engine.connect()
        logger.info(
            "Successfully connected to the SQL database"
            f" {connection_string if not settings.db.password else connection_string.replace(settings.db.password.get_secret_value(), '*****')}."
        )
        return engine
    except SQLAlchemyError:
        logger.warning(
            "Failed connection to SQL database"
            f" {connection_string if not settings.db.password else connection_string.replace(settings.db.password.get_secret_value(), '*****')}."
        )
        return None


async def validate_project(
    httpx_client: AsyncClient,
    vlab_id: str,
    project_id: str,
    token: str,
    vlab_project_url: str,
) -> None:
    """Check user appartenance to vlab and project before running agent."""
    response = await httpx_client.get(
        f"{vlab_project_url}/{vlab_id}/projects/{project_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if response.status_code != 200:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="User does not belong to the project.",
        )
