"""App dependencies."""

import logging
from functools import cache
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends
from openai import AsyncOpenAI

from swarm_copy.app.config import Settings
from swarm_copy.new_types import Agent
from swarm_copy.run import AgentsRoutine
from swarm_copy.tools import PrintAccountDetailsTool

logger = logging.getLogger(__name__)


@cache
def get_settings() -> Settings:
    """Get the global settings."""
    logger.info("Reading the environment and instantiating settings")
    return Settings()


async def get_openai_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[AsyncOpenAI | None]:
    """Get the OpenAi Async client."""
    if not settings.openai.token:
        yield None
    else:
        try:
            client = AsyncOpenAI(api_key=settings.openai.token.get_secret_value())
            yield client
        finally:
            await client.close()


def get_starting_agent(settings: Annotated[Settings, Depends(get_settings)]) -> Agent:
    """Get the starting agent."""
    logger.info(f"Loading model {settings.openai.model}.")
    agent = Agent(
        name="Agent",
        instructions="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
        tools=[PrintAccountDetailsTool],
        model=settings.openai.model,
    )
    return agent


def get_context_variables(
    settings: Annotated[Settings, Depends(get_settings)],
    starting_agent: Annotated[Agent, Depends(get_starting_agent)],
) -> dict[str, Any]:
    """Get the global context variables to feed the tool's metadata."""
    return {"user_id": 1234, "starting_agent": starting_agent}


def get_agents_routine(
    openai: Annotated[AsyncOpenAI | None, Depends(get_openai_client)],
) -> AgentsRoutine:
    """Get the AgentRoutine client."""
    return AgentsRoutine(openai)
