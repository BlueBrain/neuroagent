"""Utilities for the agent's database."""

import json
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.sql_schemas import Messages, Threads, utc_now


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
    if thread:
        thread.update_date = utc_now()
    else:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Thread not found.",
            },
        )
    await session.commit()


async def get_messages_from_db(
    user_id: str, thread_id: str, session: AsyncSession
) -> list[dict[str, Any]]:
    """Retreive the message history from the DB."""
    message_query = await session.execute(
        select(Messages)
        .join(Threads)
        .where(Threads.user_id == user_id, Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    db_messages = message_query.scalars().all()

    messages = []
    if db_messages:
        for msg in db_messages:
            if msg.content:
                messages.append(json.loads(msg.content))

    return messages
