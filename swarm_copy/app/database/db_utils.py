"""Utilities for the agent's database."""

import datetime
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.sql_schemas import Messages, Threads


async def put_messages_in_db(
    history: list[dict[str, Any]], thread_id: str, offset: int, session: AsyncSession
) -> None:
    """Add the new messages in the database."""
    for i, messages in enumerate(history):
        new_msg = Messages(
            order=i + offset,
            thread_id=thread_id,
            content=json.dumps(messages),
        )
        session.add(new_msg)
    # we need to update the thread update time
    thread = await session.get(Threads, thread_id)
    thread.update_date = datetime.datetime.now()
    await session.commit()


async def get_messages_from_db(
    thread_id: str, session: AsyncSession
) -> list[dict[str, Any] | None]:
    """Retreive the message history from the DB."""
    message_query = await session.execute(
        select(Messages).where(Messages.thread_id == thread_id).order_by(Messages.order)
    )
    db_messages = message_query.scalars().all()

    messages = []
    if db_messages:
        for msg in db_messages:
            messages.append(json.loads(msg.content))

    return messages
