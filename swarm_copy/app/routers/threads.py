"""Threads CRUDs."""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.db_utils import get_thread
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
    thread_result = await session.execute(
        select(Threads).where(Threads.user_id == user_id)
    )
    threads = thread_result.scalars().all()
    return [ThreadsRead(**thread.__dict__) for thread in threads]


@router.get("/{thread_id}")
async def get_messages(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
) -> list[MessagesRead]:
    """Get all mesaages of the thread."""
    await get_thread(user_id=user_id, thread_id=thread_id, session=session)

    messages_result = await session.execute(
        select(Messages)
        .where(Messages.thread.has(user_id=user_id), Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    db_messages = messages_result.scalars().all()

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
    new_title: str,
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
) -> ThreadsRead:
    """Update thread."""
    thread_to_update = await get_thread(
        user_id=user_id, thread_id=thread_id, session=session
    )
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
    thread_to_delete = await get_thread(
        user_id=user_id, thread_id=thread_id, session=session
    )
    await session.delete(thread_to_delete)
    await session.commit()
    return {"Acknowledged": "true"}
