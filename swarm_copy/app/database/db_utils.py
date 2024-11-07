"""Utilities for the agent's database."""

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.sql_schemas import Entity, Messages, utc_now
from swarm_copy.app.dependencies import get_thread


async def put_messages_in_db(
    history: list[dict[str, Any]],
    user_id: str,
    thread_id: str,
    offset: int,
    session: AsyncSession,
) -> None:
    """Add the new messages in the database."""
    for i, messages in enumerate(history):
        if messages["role"] != "assistant":
            entity = str(messages["role"]).upper()
        elif messages["content"]:
            entity = Entity.AI_MESSAGE.value.upper()
        else:
            entity = Entity.AI_TOOL.value.upper()

        new_msg = Messages(
            order=i + offset,
            thread_id=thread_id,
            entity=entity,
            content=json.dumps(messages),
        )
        session.add(new_msg)

    # we need to update the thread update time
    thread = await get_thread(user_id=user_id, thread_id=thread_id, session=session)
    thread.update_date = utc_now()
    await session.commit()


async def get_messages_from_db(
    user_id: str, thread_id: str, session: AsyncSession
) -> list[dict[str, Any]]:
    """Retreive the message history from the DB."""
    thread = await get_thread(user_id=user_id, thread_id=thread_id, session=session)
    db_messages = thread.messages

    messages = []
    if db_messages:
        for msg in db_messages:
            if msg.content:
                messages.append(json.loads(msg.content))

    return messages
