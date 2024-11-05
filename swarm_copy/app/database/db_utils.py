"""Utilities for the agent's database."""

import json
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.sql_schemas import Messages, Threads, utc_now


async def get_thread(user_id: str, thread_id: str, session: AsyncSession) -> Threads:
    """Check if the current thread / user matches."""
    thread_result = await session.execute(
        select(Threads).where(
            Threads.user_id == user_id, Threads.thread_id == thread_id
        )
    )
    thread = thread_result.scalars().one_or_none()
    if not thread:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Thread not found.",
            },
        )
    return thread


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
            entity = "AI_MESSAGE"
        else:
            entity = "AI_TOOL"

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
    await get_thread(user_id=user_id, thread_id=thread_id, session=session)
    messages_result = await session.execute(
        select(Messages)
        .where(Messages.thread.has(user_id=user_id), Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    db_messages = messages_result.scalars().all()

    messages = []
    if db_messages:
        for msg in db_messages:
            if msg.content:
                messages.append(json.loads(msg.content))

    return messages
