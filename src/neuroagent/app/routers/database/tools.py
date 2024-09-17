"""Conversation related CRUD operations."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from neuroagent.app.dependencies import get_agent_memory
from neuroagent.app.routers.database.schemas import Threads, ToolCallSchema
from neuroagent.app.routers.database.sql import get_object

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["Tool's CRUD"])


@router.get("/{thread_id}/{message_id}")
async def get_tool_calls(
    _: Annotated[Threads, Depends(get_object)],
    memory: Annotated[AsyncSqliteSaver | None, Depends(get_agent_memory)],
    thread_id: str,
    message_id: str,
) -> list[ToolCallSchema]:
    """Get tool calls of a specific message.
    \f

    Parameters
    ----------
    _
        Thread object returned by SQLAlchemy
    memory
        Langgraph's checkpointer's instance.
    thread_id
        ID of the thread.
    message_id
        ID of the message.

    Returns
    -------
    tool_calls
        tools called to generate a message.

    Raises
    ------
    HTTPException
        If the thread is not from the current user.
    """  # noqa: D301, D400, D205
    if memory is None:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Couldn't connect to the SQL DB.",
            },
        )
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})
    messages = await memory.aget(config)
    if messages is None:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Message not found.",
            },
        )
    message_list = messages["channel_values"]["messages"]

    # Get the specified message index
    try:
        relevant_message = next(
            (i for i, message in enumerate(message_list) if message.id == message_id)
        )
    except StopIteration:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Message not found.",
            },
        )

    if isinstance(message_list[relevant_message], HumanMessage):
        return []

    # Get the nearest previous message that has content
    previous_content_message = next(
        (
            i
            for i, message in reversed(list(enumerate(message_list[:relevant_message])))
            if message.content and not isinstance(message, ToolMessage)
        )
    )

    # From sub list, extract tool calls
    tool_calls: list[ToolCallSchema] = []
    for message in message_list[previous_content_message + 1 : relevant_message]:
        if isinstance(message, AIMessage):
            tool_calls.extend(
                [
                    ToolCallSchema(
                        call_id=tool["id"], name=tool["name"], arguments=tool["args"]
                    )
                    for tool in message.tool_calls
                ]
            )

    return tool_calls


@router.get("/output/{thread_id}/{tool_call_id}")
async def get_tool_returns(
    _: Annotated[Threads, Depends(get_object)],
    memory: Annotated[AsyncSqliteSaver | None, Depends(get_agent_memory)],
    thread_id: str,
    tool_call_id: str,
) -> list[dict[str, Any] | str]:
    """Given a tool id, return its output.
    \f

    Parameters
    ----------
    _
        Thread object returned by SQLAlchemy
    memory
        Langgraph's checkpointer's instance.
    thread_id
        ID of the thread.
    tool_call_id
        ID of the tool call.

    Returns
    -------
    tool_returns
        Output of the selected tool call.

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
    if messages is None:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Message not found.",
            },
        )
    message_list = messages["channel_values"]["messages"]

    try:
        tool_output_str = next(
            (
                message.content
                for message in message_list
                if isinstance(message, ToolMessage)
                and message.tool_call_id == tool_call_id
            )
        )
    except StopIteration:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Tool call not found.",
            },
        )
    if isinstance(tool_output_str, str):
        try:
            tool_output = json.loads(tool_output_str)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail={
                    "detail": "There was an error decoding the tool output.",
                },
            )

        if isinstance(tool_output, list):
            return tool_output
        else:
            return [tool_output]
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "detail": (
                    "There was an error retrieving the content of the tool output."
                    " Please forward your request to the ML team."
                ),
            },
        )
