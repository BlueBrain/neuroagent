"""Conversation related CRUD operations."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends
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
    messages_result = messages_result.scalars().all()

    # Find the message of interest.
    message_id_order = None
    for msg in messages_result:
        if msg.message_id == message_id:
            message_id_order = msg.order

    # Find the AI call that calls tools.
    AI_tool_call = None
    for i in range(message_id_order - 1, 0, -1):
        if json.loads(messages_result[i].content)["role"] == "assistant":
            AI_tool_call = messages_result[i].order
            break

    tool_calls = []
    tool_calls_dict = json.loads(messages_result[AI_tool_call].content)
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
        .where(Messages.thread.has(user_id=user_id), Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    messages_result = messages_result.scalars().all()

    tool_output = []

    for msg in messages_result:
        if json.loads(msg.content).get("tool_call_id") == tool_call_id:
            tool_output.append(json.loads(msg.content))

    return tool_output
