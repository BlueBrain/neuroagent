import logging
from functools import cache
from typing import Annotated, Any

from fastapi import Depends
from openai import AsyncOpenAI

from swarm_copy.app.config import Settings
from swarm_copy.new_types import Agent
from swarm_copy.run import Swarm
from swarm_copy.tools import PrintAccountDetailsTool

logger = logging.getLogger(__name__)


@cache
def get_settings():
    logger.info("Reading the environment and instantiating settings")
    return Settings()


def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI()


def get_starting_agent(settings: Annotated[Settings, Depends(get_settings)]) -> Agent:
    agent = Agent(
        name="Agent",
        instructions="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
        functions=[PrintAccountDetailsTool],
    )
    return agent


def get_context_variables(
    settings: Annotated[Settings, Depends(get_settings)],
    starting_agent: Annotated[Agent, Depends(get_starting_agent)],
) -> dict[str, Any]:
    return {"user_id": 1234, "starting_agent": starting_agent}


def get_swarm(openai: Annotated[AsyncOpenAI, Depends(get_openai_client)]) -> Swarm:
    return Swarm(openai)
