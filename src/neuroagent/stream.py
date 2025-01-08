"""Wrapper around streaming methods to reinitiate connections due to the way fastAPI StreamingResponse works."""

from typing import Any, AsyncIterator

from fastapi import Request
from httpx import AsyncClient
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from neuroagent.agent_routine import AgentsRoutine
from neuroagent.app.database.sql_schemas import Messages, Threads, utc_now
from neuroagent.new_types import Agent, Response


async def stream_agent_response(
    agents_routine: AgentsRoutine,
    agent: Agent,
    messages: list[Messages],
    context_variables: dict[str, Any],
    thread: Threads,
    request: Request,
) -> AsyncIterator[str]:
    """Redefine fastAPI connections to enable streaming."""
    # Restore the OpenAI client
    if isinstance(agents_routine.client, AsyncOpenAI):
        connected_agents_routine = AgentsRoutine(
            client=AsyncOpenAI(api_key=agents_routine.client.api_key)
        )
    else:
        connected_agents_routine = AgentsRoutine(client=None)

    # Restore the httpx client
    httpx_client = AsyncClient(
        timeout=None,
        verify=False,
        headers={
            "x-request-id": context_variables["httpx_client"].headers["x-request-id"]
        },
    )
    context_variables["httpx_client"] = httpx_client
    # Restore the session
    engine = request.app.state.engine
    session = AsyncSession(engine)
    # Need to rebind every single message
    for message in messages:
        session.add(message)

    iterator = connected_agents_routine.astream(agent, messages, context_variables)
    async for chunk in iterator:
        # To stream to the user
        if not isinstance(chunk, Response):
            yield chunk
        # Final chunk that contains the whole response
        elif chunk.hil_messages:
            yield str(
                [hil_message.model_dump_json() for hil_message in chunk.hil_messages]
            )

    # Save the new messages in DB
    thread.update_date = utc_now()
    await session.commit()
    await session.close()
