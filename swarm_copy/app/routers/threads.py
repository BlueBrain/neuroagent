"""Threads CRUDs."""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.schemas import MessagesRead, ThreadsRead
from swarm_copy.app.database.sql_schemas import Messages, Threads
from swarm_copy.app.dependencies import get_session, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["Threads' CRUD"])


@router.post("/")
async def create_thread(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    title: str = "title",
) -> ThreadsRead:
    """Create thread."""
    new_thread = Threads(user_id=user_id, title=title)
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    return ThreadsRead(**new_thread.__dict__)


@router.get("/")
async def get_threads(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> list[ThreadsRead]:
    """Get threads for a user."""
    query = select(Threads).where(Threads.user_id == user_id)
    results = await session.execute(query)
    threads = results.scalars().all()
    return [ThreadsRead(**thread.__dict__) for thread in threads]


@router.get("/{thread_id}")
async def get_messages(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
) -> list[MessagesRead]:
    """Get thread."""
    message_query = await session.execute(
        select(Messages)
        .where(Messages.thread.has(user_id=user_id), Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    db_messages = message_query.scalars().all()

    messages = []
    for msg in db_messages:
        message_dict = json.loads(msg.content)
        if message_dict["role"] == "user":
            msg.entity = "Human"
            msg.content = message_dict["content"]
            messages.append(MessagesRead(**msg.__dict__))
        if message_dict["role"] == "assistant" and message_dict["content"]:
            msg.entity = "AI"
            msg.content = message_dict["content"]
            messages.append(MessagesRead(**msg.__dict__))

    return messages


@router.patch("/{thread_id}")
async def update_thread_title(
    session: Annotated[AsyncSession, Depends(get_session)],
    new_title: str,
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
) -> ThreadsRead:
    """Update thread."""
    query = select(Threads).where(
        Threads.user_id == user_id, Threads.thread_id == thread_id
    )
    thread_to_update = await session.execute(query)
    thread_to_update = thread_to_update.scalar_one()
    thread_to_update.title = new_title
    await session.commit()
    await session.refresh(thread_to_update)
    return ThreadsRead(**thread_to_update.__dict__)


@router.delete("/{thread_id}")
async def delete_thread(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
) -> dict[str, str]:
    """Delete the specified thread."""
    query = select(Threads).where(
        Threads.user_id == user_id, Threads.thread_id == thread_id
    )
    thread_to_delete = await session.execute(query)
    thread_to_delete = thread_to_delete.scalar_one()
    await session.delete(thread_to_delete)
    await session.commit()
    return {"Acknowledged": "true"}
