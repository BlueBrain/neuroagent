"""Utilities for the agent's database."""

import json
from typing import Annotated, Any

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuroagent.app.database.sql_schemas import (
    Entity,
    Messages,
    Threads,
    ToolCalls,
    utc_now,
)
from neuroagent.app.dependencies import get_session, get_user_id
from neuroagent.utils import get_entity


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


async def save_history(
    history: list[dict[str, Any]],
    offset: int,
    thread: Threads,
    session: AsyncSession,
) -> None:
    """Add the new messages in the database."""
    for i, message in enumerate(history):
        tool_calls = []
        entity = get_entity(message)
        if entity == Entity.AI_TOOL:
            # If AI_TOOL, create separate ToolCall entries in DB and remove it from content
            tool_calls = [
                ToolCalls(
                    tool_call_id=tool_call["id"],
                    name=tool_call["function"]["name"],
                    arguments=json.dumps(tool_call["function"]["arguments"]),
                )
                for tool_call in message["tool_calls"]
            ]
            message.pop("tool_calls")
        else:
            raise HTTPException(status_code=500, detail="Unknown message entity.")

        new_msg = Messages(
            order=i + offset,
            thread_id=thread.thread_id,
            entity=entity,
            content=json.dumps(message),
            tool_calls=tool_calls,
        )
        session.add(new_msg)

    # we need to update the thread update time
    thread.update_date = utc_now()
    await session.commit()


# db_messages: list[Messages] = await thread.awaitable_attrs.messages
