"""Conversation related CRUD operations."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from sqlalchemy import MetaData, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from neuroagent.app.dependencies import (
    get_agent_memory,
    get_engine,
    get_session,
    get_user_id,
)
from neuroagent.app.routers.database.schemas import (
    GetThreadsOutput,
    Threads,
    ThreadsRead,
    ThreadsUpdate,
)
from neuroagent.app.routers.database.sql import get_object

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threads", tags=["Threads' CRUD"])


@router.post("/")
def create_thread(
    session: Annotated[Session, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    title: str = "title",
) -> ThreadsRead:
    """Create thread.
    \f

    Parameters
    ----------
    session
        SQL session to communicate with the db.
    user_id
        ID of the current user.
    title
        Title of the thread to create.

    Returns
    -------
    thread_dict: {'thread_id': 'thread_name'}
        Conversation created.
    """  # noqa: D301, D400, D205
    new_thread = Threads(user_sub=user_id, title=title)
    session.add(new_thread)
    session.commit()
    session.refresh(new_thread)
    return ThreadsRead(**new_thread.__dict__)


@router.get("/")
def get_threads(
    session: Annotated[Session, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> list[ThreadsRead]:
    """Get threads for a user.
    \f

    Parameters
    ----------
    session
        SQL session to communicate with the db.
    user_id
        ID of the current user.

    Returns
    -------
    list[ThreadsRead]
        List the threads from the given user id.
    """  # noqa: D205, D301, D400
    query = select(Threads).where(Threads.user_sub == user_id)
    threads = session.execute(query).all()
    return [ThreadsRead(**thread[0].__dict__) for thread in threads]


@router.get("/{thread_id}")
async def get_thread(
    _: Annotated[Threads, Depends(get_object)],
    memory: Annotated[AsyncSqliteSaver | None, Depends(get_agent_memory)],
    thread_id: str,
) -> list[GetThreadsOutput]:
    """Get thread.
    \f

    Parameters
    ----------
    _
        Thread object returned by SQLAlchemy
    memory
        Langgraph's checkpointer's instance.
    thread_id
        ID of the thread.

    Returns
    -------
    messages
        Conversation with messages.

    Raises
    ------
    HTTPException
        If the thread is not from the current user.
    """  # noqa: D301, D205, D400
    if memory is None:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Couldn't connect to the SQL DB.",
            },
        )
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})
    messages = await memory.aget(config)
    if not messages:
        return []

    output: list[GetThreadsOutput] = []
    # Reconstruct the conversation. Also output message_id for other endpoints
    for message in messages["channel_values"]["messages"]:
        if isinstance(message, HumanMessage):
            output.append(
                GetThreadsOutput(
                    message_id=message.id,
                    entity="Human",
                    message=message.content,
                )
            )
        if isinstance(message, AIMessage) and message.content:
            output.append(
                GetThreadsOutput(
                    message_id=message.id,
                    entity="AI",
                    message=message.content,
                )
            )
    return output


@router.patch("/{thread_id}")
def update_thread_title(
    thread: Annotated[Threads, Depends(get_object)],
    session: Annotated[Session, Depends(get_session)],
    thread_update: ThreadsUpdate,
) -> ThreadsRead:
    """Update thread.
    \f

    Parameters
    ----------
    thread
        Thread object returned by SQLAlchemy
    session
        SQL session
    thread_update
        Pydantic class containing the required updates

    Returns
    -------
    thread_return
        Updated thread instance
    """  # noqa: D205, D301, D400
    thread_data = thread_update.model_dump(exclude_unset=True)
    for key, value in thread_data.items():
        setattr(thread, key, value)
    session.add(thread)
    session.commit()
    session.refresh(thread)
    thread_return = ThreadsRead(**thread.__dict__)  # For mypy.
    return thread_return


@router.delete("/{thread_id}")
def delete_thread(
    _: Annotated[Threads, Depends(get_object)],
    session: Annotated[Session, Depends(get_session)],
    engine: Annotated[Engine, Depends(get_engine)],
    thread_id: str,
) -> dict[str, str]:
    """Delete the specified thread.
    \f

    Parameters
    ----------
    _
        Thread object returned by SQLAlchemy
    session
        SQL session
    engine
        SQL engine
    thread_id
        ID of the relevant thread

    Returns
    -------
    Acknowledgement of the deletion
    """  # noqa: D205, D301, D400
    metadata = MetaData()
    metadata.reflect(engine)
    for table in metadata.tables.values():
        if "thread_id" not in table.c.keys():
            continue
        # Delete from the checkpoint table
        query = table.delete().where(table.c.thread_id == thread_id)
        session.execute(query)

    session.commit()
    return {"Acknowledged": "true"}
