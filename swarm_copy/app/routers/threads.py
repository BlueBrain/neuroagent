"""Threads CRUDs."""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.schemas import MessagesRead, ThreadsRead, ThreadUpdate
from swarm_copy.app.database.sql_schemas import Threads
from swarm_copy.app.dependencies import get_session, get_thread, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["Threads' CRUD"])


@router.post("/")
async def create_thread(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    title: str = "New chat",
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
    thread_result = await session.execute(
        select(Threads).where(Threads.user_id == user_id)
    )
    threads = thread_result.scalars().all()
    return [ThreadsRead(**thread.__dict__) for thread in threads]


@router.get("/{thread_id}")
async def get_messages(
    thread: Annotated[Threads, Depends(get_thread)],
) -> list[MessagesRead]:
    """Get all mesaages of the thread."""
    db_messages = thread.messages

    messages = []
    for msg in db_messages:
        if msg.entity.value in ("user", "ai_message"):
            messages.append(
                MessagesRead(
                    msg_entity=msg.entity.value,
                    msg_content=json.loads(msg.content)["content"],
                    **msg.__dict__,
                )
            )

    return messages


@router.patch("/{thread_id}")
async def update_thread_title(
    session: Annotated[AsyncSession, Depends(get_session)],
    update_thread: ThreadUpdate,
    thread: Annotated[Threads, Depends(get_thread)],
) -> ThreadsRead:
    """Update thread."""
    thread_data = update_thread.model_dump(exclude_unset=True)
    for key, value in thread_data.items():
        setattr(thread, key, value)
    session.add(thread)
    await session.commit()
    await session.refresh(thread)
    return ThreadsRead(**thread.__dict__)


@router.delete("/{thread_id}")
async def delete_thread(
    session: Annotated[AsyncSession, Depends(get_session)],
    thread: Annotated[Threads, Depends(get_thread)],
) -> dict[str, str]:
    """Delete the specified thread."""
    await session.delete(thread)
    await session.commit()
    return {"Acknowledged": "true"}
