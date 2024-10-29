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
from pydantic import BaseModel

from swarm_copy.app.app_utils import setup_engine
from swarm_copy.app.database.sql_schemas import Base
from swarm_copy.app.dependencies import (
    get_agents_routine,
    get_connection_string,
    get_context_variables,
    get_settings,
    get_starting_agent,
)
from swarm_copy.new_types import Agent
from swarm_copy.run import AgentsRoutine
from swarm_copy.stream import stream_agent_response

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


class AgentRequest(BaseModel):
    """Class for agent request."""

    query: str


@app.post("/run/qa")
async def run_simple_agent(
    user_request: AgentRequest,
    agent_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
) -> str:
    """Run a single agent query."""
    response = await agent_routine.arun(
        agent, [{"role": "user", "content": user_request.query}], context_variables
    )
    return response.messages


@app.post("/run/streamed")
async def stream_simple_agent(
    user_request: AgentRequest,
    agents_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
) -> StreamingResponse:
    """Run a single agent query in a streamed fashion."""
    stream_generator = stream_agent_response(
        agents_routine,
        agent,
        [{"role": "user", "content": user_request.query}],
        context_variables,
    )
    return StreamingResponse(stream_generator, media_type="text/event-stream")
