"""Conversation related CRUD operations."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

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
    message_query = await session.execute(
        select(Messages)
        .where(Messages.thread.has(user_id=user_id), Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    db_messages = message_query.scalars().all()

    # Find the message of interest.
    message_id_order = None
    for msg in db_messages:
        if msg.message_id == message_id:
            message_id_order = msg.order

    # Find the AI call that calls tools.
    AI_tool_call = None
    for i in range(message_id_order - 1, 0, -1):
        if json.loads(db_messages[i].content)["role"] == "assistant":
            AI_tool_call = db_messages[i].order
            break

    tool_calls = []
    tool_calls_dict = json.loads(db_messages[AI_tool_call].content)
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
    message_query = await session.execute(
        select(Messages)
        .where(Messages.thread.has(user_id=user_id), Messages.thread_id == thread_id)
        .order_by(Messages.order)
    )
    db_messages = message_query.scalars().all()

    tool_output = []

    for msg in db_messages:
        if json.loads(msg.content).get("tool_call_id") == tool_call_id:
            tool_output.append(json.loads(msg.content))

    return tool_output
