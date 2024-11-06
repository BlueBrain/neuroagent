"""Conversation related CRUD operations."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.db_utils import get_thread
from swarm_copy.app.database.schemas import ToolCallSchema
from swarm_copy.app.database.sql_schemas import Messages
from swarm_copy.app.dependencies import get_session, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Tool's CRUD"])


@router.get("/{thread_id}/{message_id}")
async def get_tool_calls(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
    message_id: str,
) -> list[ToolCallSchema]:
    """Get tool calls of a specific message."""
    await get_thread(user_id=user_id, thread_id=thread_id, session=session)

    messages_result = await session.execute(
        select(Messages)
        .where(Messages.thread.has(user_id=user_id), Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    db_messages = messages_result.scalars().all()

    # Find the message of interest.
    try:
        relevant_message = next(
            (
                i
                for i, message in enumerate(db_messages)
                if message.message_id == message_id
            )
        )
    except StopIteration:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Message not found.",
            },
        )

    # If not an AI response, there is no tool call associated.
    if db_messages[relevant_message].entity.value != "ai_message":
        return []

    # Get the nearest previous message that called the tools.
    previous_content_message = next(
        (
            i
            for i, message in reversed(list(enumerate(db_messages[:relevant_message])))
            if message.entity.value == "ai_tool"
        )
    )

    # We should maybe give back the messag_id, for easier search after.
    tool_calls = []
    tool_calls_dict = json.loads(db_messages[previous_content_message].content)
    for tool in tool_calls_dict["tool_calls"]:
        tool_calls.append(
            ToolCallSchema(
                tool_call_id=tool["id"],
                name=tool["function"]["name"],
                arguments=json.loads(tool["function"]["arguments"]),
            )
        )

    return tool_calls


@router.get("/output/{thread_id}/{tool_call_id}")
async def get_tool_returns(
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
    tool_call_id: str,
) -> list[dict[str, Any] | str]:
    """Given a tool id, return its output."""
    await get_thread(user_id=user_id, thread_id=thread_id, session=session)

    messages_result = await session.execute(
        select(Messages)
        .where(
            Messages.thread.has(user_id=user_id),
            Messages.thread_id == thread_id,
            Messages.entity == "TOOL",
        )
        .order_by(Messages.order)
    )
    tool_messages = messages_result.scalars().all()

    tool_output = []

    # We search all tool messages for matching tool_call id.
    # Maybe we should add also the arguments here ?
    for msg in tool_messages:
        if json.loads(msg.content).get("tool_call_id") == tool_call_id:
            tool_output.append(json.loads(msg.content))

    return tool_output
