"""FastAPI for the Agent."""

import logging
from contextlib import asynccontextmanager
from logging.config import dictConfig
from typing import Annotated, Any, AsyncContextManager
from uuid import uuid4

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware

from neuroagent import __version__
from neuroagent.app.config import Settings
from neuroagent.app.dependencies import (
    auth,
    get_agent_memory,
    get_cell_types_kg_hierarchy,
    get_connection_string,
    get_engine,
    get_kg_token,
    get_settings,
    get_update_kg_hierarchy,
)
from neuroagent.app.middleware import strip_path_prefix
from neuroagent.app.routers import qa
from neuroagent.app.routers.database import threads, tools
from neuroagent.app.routers.database.schemas import Base, Threads  # noqa: F401

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
    # hacky but works: https://github.com/tiangolo/fastapi/issues/425
    app_settings = fastapi_app.dependency_overrides.get(get_settings, get_settings)()
    if app_settings.keycloak.validate_token:
        auth.model.flows = app_settings.keycloak.flows  # type: ignore

    engine = fastapi_app.dependency_overrides.get(get_engine, get_engine)(
        app_settings, get_connection_string(app_settings)
    )
    # This creates the checkpoints and writes tables.
    await anext(
        fastapi_app.dependency_overrides.get(get_agent_memory, get_agent_memory)(
            get_connection_string(app_settings)
        )
    )
    if engine:
        Base.metadata.create_all(bind=engine)

    prefix = app_settings.misc.application_prefix
    fastapi_app.openapi_url = f"{prefix}/openapi.json"
    fastapi_app.servers = [{"url": prefix}]
    # Do not rely on the middleware order in the list "fastapi_app.user_middleware" since this is subject to changes.
    try:
        cors_middleware = filter(
            lambda x: x.__dict__["cls"] == CORSMiddleware, fastapi_app.user_middleware
        ).__next__()
        cors_middleware.kwargs["allow_origins"] = (
            app_settings.misc.cors_origins.replace(" ", "").split(",")
        )
    except StopIteration:
        pass

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


app = FastAPI(
    title="Agents",
    summary=(
        "Use an AI agent to answer queries based on the knowledge graph, literature"
        " search and neuroM."
    ),
    version=__version__,
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
app.add_middleware(BaseHTTPMiddleware, dispatch=strip_path_prefix)

app.include_router(qa.router)
app.include_router(threads.router)
app.include_router(tools.router)


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
