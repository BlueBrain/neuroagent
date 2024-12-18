"""Wrapper around streaming methods to reinitiate connections due to the way fastAPI StreamingResponse works."""

from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.agent_routine import AgentsRoutine
from swarm_copy.app.database.db_utils import save_history
from swarm_copy.new_types import Agent, Response


async def stream_agent_response(
    agents_routine: AgentsRoutine,
    agent: Agent,
    messages: list[dict[str, str]],
    context_variables: dict[str, Any],
    user_id: str,
    thread_id: str,
    session: AsyncSession,
) -> AsyncIterator[str]:
    """Redefine fastAPI connections to enable streaming."""
    if isinstance(agents_routine.client, AsyncOpenAI):
        connected_agents_routine = AgentsRoutine(
            client=AsyncOpenAI(api_key=agents_routine.client.api_key)
        )
    else:
        connected_agents_routine = AgentsRoutine(client=None)

    iterator = connected_agents_routine.astream(agent, messages, context_variables)
    async for chunk in iterator:
        # To stream to the user
        if not isinstance(chunk, Response):
            yield chunk
        # Final chunk that contains the whole response
        else:
            to_db = chunk

    await save_history(
        user_id=user_id,
        history=to_db.messages,
        offset=len(messages) - 1,
        thread_id=thread_id,
        session=session,
    )
