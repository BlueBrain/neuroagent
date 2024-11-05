"""Main."""

import logging
from contextlib import asynccontextmanager
from logging.config import dictConfig
from typing import Annotated, Any, AsyncContextManager
from uuid import uuid4

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from httpx import AsyncClient
from pydantic import BaseModel

from swarm_copy.app.app_utils import setup_engine
from swarm_copy.app.config import Settings
from swarm_copy.app.database.sql_schemas import Base
from swarm_copy.app.dependencies import (
    get_agents_routine,
    get_cell_types_kg_hierarchy,
    get_context_variables,
    get_kg_token,
    get_settings,
    get_starting_agent,
    get_update_kg_hierarchy,
    get_connection_string,
    get_settings,
)
from swarm_copy.app.routers import qa

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "correlation_id": {
            "()": "asgi_correlation_id.CorrelationIdFilter",
            "uuid_length": 32,
            "default_value": "-",
        },
    },
    "formatters": {
        "request_id": {
            "class": "logging.Formatter",
            "format": (
                "[%(levelname)s] %(asctime)s (%(correlation_id)s) %(name)s %(message)s"
            ),
        },
    },
    "handlers": {
        "request_id": {
            "class": "logging.StreamHandler",
            "filters": ["correlation_id"],
            "formatter": "request_id",
        },
    },
    "loggers": {
        "": {
            "handlers": ["request_id"],
            "level": "INFO",
            "propagate": True,
        },
    },
}
dictConfig(LOGGING)

logger = logging.getLogger(__name__)


@asynccontextmanager  # type: ignore
async def lifespan(fastapi_app: FastAPI) -> AsyncContextManager[None]:  # type: ignore
    """Read environment (settings of the application)."""
    app_settings = fastapi_app.dependency_overrides.get(get_settings, get_settings)()

    # Get the sqlalchemy engine and store it in app state.
    engine = setup_engine(app_settings, get_connection_string(app_settings))
    fastapi_app.state.engine = engine

    # Create the tables for the agent memory.
    if engine:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    logging.getLogger().setLevel(app_settings.logging.external_packages.upper())
    logging.getLogger("neuroagent").setLevel(app_settings.logging.level.upper())
    logging.getLogger("bluepyefe").setLevel("CRITICAL")

    if app_settings.knowledge_graph.download_hierarchy:
        # update KG hierarchy file if requested
        await get_update_kg_hierarchy(
            token=get_kg_token(app_settings, token=None),
            httpx_client=AsyncClient(),
            settings=app_settings,
        )
        await get_cell_types_kg_hierarchy(
            token=get_kg_token(app_settings, token=None),
            httpx_client=AsyncClient(),
            settings=app_settings,
        )

    yield
    if engine:
        await engine.dispose()


app = FastAPI(
    title="Agents",
    summary=(
        "Use an AI agent to answer queries based on the knowledge graph, literature"
        " search and neuroM."
    ),
    version="0.0.0",
    swagger_ui_parameters={"tryItOutEnabled": True},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http:\/\/localhost:(\d+)\/?.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    CorrelationIdMiddleware,
    header_name="X-Request-ID",
    update_request_header=True,
    generator=lambda: uuid4().hex,
    transformer=lambda a: a,
)


app.include_router(qa.router)


@app.get("/healthz")
def healthz() -> str:
    """Check the health of the API."""
    return "200"


@app.get("/")
def readyz() -> dict[str, str]:
    """Check if the API is ready to accept traffic."""
    return {"status": "ok"}


@app.get("/settings")
def settings(settings: Annotated[Settings, Depends(get_settings)]) -> Any:
    """Show complete settings of the backend.

    Did not add return model since it pollutes the Swagger UI.
    """
    return settings
