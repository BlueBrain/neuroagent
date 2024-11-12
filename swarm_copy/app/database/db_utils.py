"""Utilities for the agent's database."""

import json
from typing import Annotated, Any

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.sql_schemas import Entity, Messages, Threads, utc_now
from swarm_copy.app.dependencies import get_session, get_user_id


async def get_thread(
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Threads:
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
        if messages["role"] == "user":
            entity = Entity.USER
        elif messages["role"] == "tool":
            entity = Entity.TOOL
        elif messages["role"] == "assistant" and messages["content"]:
            entity = Entity.AI_MESSAGE
        elif messages["role"] == "assistant" and not messages["content"]:
            entity = Entity.AI_TOOL
        else:
            raise HTTPException(status_code=500, detail="Unknown message entity.")

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
    thread: Annotated[Threads, Depends(get_thread)],
) -> list[dict[str, Any]]:
    """Retreive the message history from the DB."""
    db_messages: list[Messages] = await thread.awaitable_attrs.messages

    messages = []
    if db_messages:
        for msg in db_messages:
            if msg.content:
                messages.append(json.loads(msg.content))

    return messages
