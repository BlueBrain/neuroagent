"""Conversation related CRUD operations."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.db_utils import get_thread
from swarm_copy.app.database.schemas import ToolCallSchema
from swarm_copy.app.database.sql_schemas import Entity, Messages, Threads
from swarm_copy.app.dependencies import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Tool's CRUD"])


@router.get("/{thread_id}/{message_id}")
async def get_tool_calls(
    thread: Annotated[Threads, Depends(get_thread)],
    session: Annotated[AsyncSession, Depends(get_session)],
    message_id: str,
) -> list[ToolCallSchema]:
    """Get tool calls of a specific message."""
    # Find relevant messages
    relevant_message_result = await session.execute(
        select(Messages).where(
            Messages.thread_id == thread.thread_id, Messages.message_id == message_id
        )
    )
    relevant_message = relevant_message_result.scalar_one_or_none()

    # Check if message exists, and if of right type.
    if not relevant_message:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Message not found.",
            },
        )
    if relevant_message.entity != Entity.AI_MESSAGE:
        return []

    # Get the nearest previous message that called the tools.
    previous_user_message_result = await session.execute(
        select(Messages)
        .where(
            Messages.thread_id == thread.thread_id,
            Messages.order < relevant_message.order,
            Messages.entity == Entity.USER,
        )
        .order_by(Messages.order)
    )
    previous_user_message = previous_user_message_result.scalars().all()
    if not previous_user_message:
        return []
    last_user_message = previous_user_message[-1]

    # Get all the "AI_TOOL" messsages in between.
    tool_call_result = await session.execute(
        select(Messages)
        .where(
            Messages.thread_id == thread.thread_id,
            Messages.order < relevant_message.order,
            Messages.order > last_user_message.order,
            Messages.entity == Entity.AI_TOOL,
        )
        .order_by(Messages.order)
    )
    all_relevant_tool_calls = tool_call_result.scalars().all()

    # We should maybe give back the messag_id, for easier search after.
    tool_calls_response = []
    for tool_calls in all_relevant_tool_calls:
        tool_calls_dict = json.loads(tool_calls.content)
        for tool in tool_calls_dict["tool_calls"]:
            tool_calls_response.append(
                ToolCallSchema(
                    tool_call_id=tool["id"],
                    name=tool["function"]["name"],
                    arguments=json.loads(tool["function"]["arguments"]),
                )
            )

    return tool_calls_response


@router.get("/output/{thread_id}/{tool_call_id}")
async def get_tool_returns(
    session: Annotated[AsyncSession, Depends(get_session)],
    thread: Annotated[Threads, Depends(get_thread)],
    tool_call_id: str,
) -> list[dict[str, Any] | str]:
    """Given a tool id, return its output."""
    messages_result = await session.execute(
        select(Messages)
        .where(
            Messages.thread_id == thread.thread_id,
            Messages.entity == "TOOL",
        )
        .order_by(Messages.order)
    )
    tool_messages = messages_result.scalars().all()

    tool_output = []
    for msg in tool_messages:
        if json.loads(msg.content).get("tool_call_id") == tool_call_id:
            tool_output.append(json.loads(msg.content))

    return tool_output
