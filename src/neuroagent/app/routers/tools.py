"""Conversation related CRUD operations."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from neuroagent.app.database.db_utils import get_thread
from neuroagent.app.database.schemas import ToolCallSchema
from neuroagent.app.database.sql_schemas import Entity, Messages, Threads, ToolCalls
from neuroagent.app.dependencies import get_session, get_starting_agent
from neuroagent.new_types import Agent, HILResponse, HILValidation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Tool's CRUD"])


@router.get("/{thread_id}/{message_id}")
async def get_tool_calls(
    _: Annotated[Threads, Depends(get_thread)],  # to check if thread exist
    session: Annotated[AsyncSession, Depends(get_session)],
    thread_id: str,
    message_id: str,
) -> list[ToolCallSchema]:
    """Get tool calls of a specific message."""
    # Find relevant messages
    relevant_message = await session.get(Messages, message_id)

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
            Messages.thread_id == thread_id,
            Messages.order < relevant_message.order,
            Messages.entity == Entity.USER,
        )
        .order_by(desc(Messages.order))
        .limit(1)
    )
    previous_user_message = previous_user_message_result.scalars().one_or_none()
    if not previous_user_message:
        return []

    # Get all the "AI_TOOL" messsages in between.
    ai_tool_messages_query = await session.execute(
        select(Messages)
        .where(
            Messages.thread_id == thread_id,
            Messages.order < relevant_message.order,
            Messages.order > previous_user_message.order,
            Messages.entity == Entity.AI_TOOL,
        )
        .order_by(Messages.order)
    )
    ai_tool_messages = ai_tool_messages_query.scalars().all()

    # We should maybe give back the message_id, for easier search after.
    tool_calls_response = []
    for ai_tool_message in ai_tool_messages:
        tool_calls = await ai_tool_message.awaitable_attrs.tool_calls
        for tool in tool_calls:
            tool_calls_response.append(
                ToolCallSchema(
                    tool_call_id=tool.tool_call_id,
                    name=tool.name,
                    arguments=json.loads(tool.arguments),
                )
            )

    return tool_calls_response


@router.get("/output/{thread_id}/{tool_call_id}")
async def get_tool_returns(
    _: Annotated[Threads, Depends(get_thread)],  # to check if thread exist
    session: Annotated[AsyncSession, Depends(get_session)],
    thread_id: str,
    tool_call_id: str,
) -> list[dict[str, Any] | str]:
    """Given a tool id, return its output."""
    messages_result = await session.execute(
        select(Messages)
        .where(
            Messages.thread_id == thread_id,
            Messages.entity == Entity.TOOL,
        )
        .order_by(Messages.order)
    )
    tool_messages = messages_result.scalars().all()

    tool_output = []
    for msg in tool_messages:
        msg_content = json.loads(msg.content)
        if msg_content.get("tool_call_id") == tool_call_id:
            tool_output.append(msg_content["content"])

    return tool_output


@router.get("/validation/{thread_id}/")
async def get_required_validation(
    _: Annotated[Threads, Depends(get_thread)],
    thread_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    starting_agent: Annotated[Agent, Depends(get_starting_agent)],
) -> list[HILResponse]:
    """List tool calls currently requiring validation in a thread."""
    message_query = await session.execute(
        select(Messages)
        .where(Messages.thread_id == thread_id)
        .order_by(desc(Messages.order))
        .limit(1)
    )
    message = message_query.scalar_one_or_none()
    if not message or message.entity != Entity.AI_TOOL:
        return []

    else:
        tool_calls = await message.awaitable_attrs.tool_calls
        need_validation = []
        for tool_call in tool_calls:
            tool = next(
                tool for tool in starting_agent.tools if tool.name == tool_call.name
            )
            if tool.hil and tool_call.validated is None:
                input_schema = tool.__annotations__["input_schema"](
                    **json.loads(tool_call.arguments)
                )
                need_validation.append(
                    HILResponse(
                        message="Please validate the following inputs before proceeding.",
                        name=tool_call.name,
                        inputs=input_schema.model_dump(),
                        tool_call_id=tool_call.tool_call_id,
                    )
                )
        return need_validation


@router.patch("/validation/{thread_id}/{tool_call_id}")
async def validate_input(
    user_request: HILValidation,
    _: Annotated[Threads, Depends(get_thread)],
    tool_call_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    starting_agent: Annotated[Agent, Depends(get_starting_agent)],
) -> ToolCallSchema:
    """Validate HIL inputs."""
    # We first find the AI TOOL message to modify.
    tool_call = await session.get(ToolCalls, tool_call_id)
    if not tool_call:
        raise HTTPException(status_code=404, detail="Specified tool call not found.")
    if tool_call.validated is not None:
        raise HTTPException(
            status_code=403, detail="The tool call has already been validated."
        )

    tool_call.validated = user_request.is_validated  # Accepted or rejected

    # If the user specified a json, take it as the new one
    # We modify only if the user validated
    if user_request.validated_inputs and user_request.is_validated:
        # Find the corresponding tool (class) to do input validation
        tool = next(
            tool for tool in starting_agent.tools if tool.name == tool_call.name
        )

        # Validate the input JSON provided by user
        try:
            tool.__annotations__["input_schema"](**user_request.validated_inputs)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        tool_call.arguments = json.dumps(user_request.validated_inputs)

    await session.commit()
    await session.refresh(tool_call)
    return ToolCallSchema(
        tool_call_id=tool_call.tool_call_id,
        name=tool_call.name,
        arguments=json.loads(tool_call.arguments),
    )
