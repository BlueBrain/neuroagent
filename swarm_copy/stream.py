"""Wrapper around streaming methods to reinitiate connections due to the way fastAPI StreamingResponse works."""

import json
import time
from typing import Any, AsyncIterator

from httpx import AsyncClient
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.db_utils import save_history
from swarm_copy.new_types import Agent
from swarm_copy.run import AgentsRoutine


async def stream_agent_response(
    agents_routine: AgentsRoutine,
    agent: Agent,
    messages: list[dict[str, str]],
    context_variables: dict[str, Any],
    user_id: str,
    thread_id: str,
    session: AsyncSession,
    not_ai_tool: bool,
) -> AsyncIterator[str]:
    """Redefine fastAPI connections to enable streaming."""
    if isinstance(agents_routine.client, AsyncOpenAI):
        connected_agents_routine = AgentsRoutine(
            client=AsyncOpenAI(api_key=agents_routine.client.api_key)
        )
    else:
        connected_agents_routine = AgentsRoutine(client=None)
    context_variables["httpx_client"] = AsyncClient(timeout=None, verify=False)

    iterator = connected_agents_routine.astream(agent, messages, context_variables)
    async for chunk in iterator:
        # To stream to the user
        if not isinstance(chunk, tuple):
            yield chunk
        # Final chunk that contains the whole response
        else:
            if chunk[1] and chunk[1] != "PLOT":
                time.sleep(0.1)
                tool_valid_list = json.dumps([tool.model_dump() for tool in chunk[1]])
                yield "\n<requires_human_approval>\n" + tool_valid_list
            to_db = chunk[0]

    offset = len(messages) - 1 if not_ai_tool else len(messages)
    await save_history(
        user_id=user_id,
        history=to_db.messages if not_ai_tool else to_db.messages[1:],
        offset=offset,
        thread_id=thread_id,
        session=session,
    )
