"""App dependencies."""

import logging
from functools import cache
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, HTTPException, Request
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from swarm_copy.app.config import Settings
from swarm_copy.app.database.sql_schemas import Threads, Users
from swarm_copy.new_types import Agent
from swarm_copy.run import AgentsRoutine
from swarm_copy.tools import PrintAccountDetailsTool

logger = logging.getLogger(__name__)


@cache
def get_settings() -> Settings:
    """Get the global settings."""
    logger.info("Reading the environment and instantiating settings")
    return Settings()


def get_openai_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncOpenAI | None:
    """Get the OpenAi Async client."""
    if settings.openai.token:
        return AsyncOpenAI(api_key=settings.openai.token.get_secret_value())
    else:
        return None


def get_connection_string(
    settings: Annotated[Settings, Depends(get_settings)],
) -> str | None:
    """Get the db interacting class for chat agent."""
    if settings.db.prefix:
        connection_string = settings.db.prefix
        if settings.db.user and settings.db.password:
            # Add authentication.
            connection_string += (
                f"{settings.db.user}:{settings.db.password.get_secret_value()}@"
            )
        if settings.db.host:
            # Either in file DB or connect to remote host.
            connection_string += settings.db.host
        if settings.db.port:
            # Add the port if remote host.
            connection_string += f":{settings.db.port}"
        if settings.db.name:
            # Add database name if specified.
            connection_string += f"/{settings.db.name}"
        return connection_string
    else:
        return None


def get_engine(request: Request) -> AsyncEngine | None:
    """Get the SQL engine."""
    return request.app.state.engine


async def get_session(
    engine: Annotated[AsyncEngine | None, Depends(get_engine)],
) -> AsyncIterator[AsyncSession]:
    """Yield a session per request."""
    if not engine:
        raise HTTPException(
            status_code=500,
            detail={
                "detail": "Couldn't connect to the SQL DB.",
            },
        )
    async with AsyncSession(engine) as session:
        yield session


def get_starting_agent(
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Agent:
    """Get the starting agent."""
    logger.info(f"Loading model {settings.openai.model}.")
    agent = Agent(
        name="Agent",
        instructions="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
        tools=[PrintAccountDetailsTool],
        model=settings.openai.model,
        database_session=session,
    )
    return agent


# TEMP function, will get replaced by the CRUDs.
async def get_user_thread_id(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> str:
    """Temp function to get the thread id."""
    # for now hard coded temp user_sub and thread_id.
    user_sub = "dev"
    thread_id = "dev_thread"

    # check if user is in DB.
    query_user = await session.execute(select(Users).where(Users.user_sub == user_sub))
    user = query_user.scalar_one_or_none()
    if not user:
        new_user = Users(user_sub=user_sub)
        session.add(new_user)
        await session.commit()
        await session.refresh(new_user)
        user = new_user

    # check if thread is in DB.
    thread = await session.get(Threads, thread_id)
    if not thread or thread.user_id != user.id:
        new_thread = Threads(user_id=user.id, thread_id=thread_id)
        session.add(new_thread)
        await session.commit()
        await session.refresh(new_thread)
        thread = new_thread

    return thread.thread_id


def get_context_variables(
    settings: Annotated[Settings, Depends(get_settings)],
    starting_agent: Annotated[Agent, Depends(get_starting_agent)],
    thread_id: Annotated[str, Depends(get_user_thread_id)],
) -> dict[str, Any]:
    """Get the global context variables to feed the tool's metadata."""
    return {"user_id": 1234, "starting_agent": starting_agent, "thread_id": thread_id}


def get_agents_routine(
    openai: Annotated[AsyncOpenAI, Depends(get_openai_client)],
) -> AgentsRoutine:
    """Get the AgentRoutine client."""
    return AgentsRoutine(openai)
