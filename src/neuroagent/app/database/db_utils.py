"""Utilities for the agent's database."""

from typing import Annotated

from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuroagent.app.database.sql_schemas import (
    Threads,
)
from neuroagent.app.dependencies import get_session, get_user_id


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
