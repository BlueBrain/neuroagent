"""Endpoints for agent's question answering pipeline."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.app.database.db_utils import get_history, get_thread, save_history
from swarm_copy.app.database.schemas import ToolCallSchema
from swarm_copy.app.database.sql_schemas import Entity, Messages, Threads
from swarm_copy.app.dependencies import (
    get_agents_routine,
    get_context_variables,
    get_session,
    get_starting_agent,
    get_user_id,
)
from swarm_copy.new_types import (
    Agent,
    AgentRequest,
    AgentResponse,
    HILResponse,
    HILValidation,
)
from swarm_copy.run import AgentsRoutine
from swarm_copy.stream import stream_agent_response

router = APIRouter(prefix="/qa", tags=["Run the agent"])

logger = logging.getLogger(__name__)


@router.post("/run/", response_model=AgentResponse)
async def run_simple_agent(
    user_request: AgentRequest,
    agent_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
) -> AgentResponse:
    """Run a single agent query."""
    response = await agent_routine.arun(
        agent, [{"role": "user", "content": user_request.query}], context_variables
    )
    return AgentResponse(message=response.messages[-1]["content"])


@router.post("/chat/{thread_id}")
async def run_chat_agent(
    user_request: AgentRequest,
    agent_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
    messages: Annotated[list[dict[str, Any]], Depends(get_history)],
) -> AgentResponse | list[HILResponse]:
    """Run a single agent query."""
    not_ai_tool = not messages or not (
        messages[-1]["role"] == "assistant" and messages[-1]["tool_calls"]
    )
    if not_ai_tool:
        messages.append({"role": "user", "content": user_request.query})
    response, hil_messages = await agent_routine.arun(
        agent, messages, context_variables
    )
    offset = len(messages) - 1 if not_ai_tool else len(messages)

    await save_history(
        user_id=user_id,
        history=response.messages if not_ai_tool else response.messages[1:],
        offset=offset,
        thread_id=thread_id,
        session=session,
    )
    if hil_messages:
        return hil_messages
    return AgentResponse(message=response.messages[-1]["content"])


@router.post("/chat_streamed/{thread_id}")
async def stream_chat_agent(
    user_request: AgentRequest,
    agents_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread_id: str,
    messages: Annotated[list[dict[str, Any]], Depends(get_history)],
) -> StreamingResponse:
    """Run a single agent query in a streamed fashion."""
    messages.append({"role": "user", "content": user_request.query})
    stream_generator = stream_agent_response(
        agents_routine,
        agent,
        messages,
        context_variables,
        user_id,
        thread_id,
        session,
    )
    return StreamingResponse(stream_generator, media_type="text/event-stream")


@router.post("/validate/{thread_id}")
async def validate_input(
    user_request: HILValidation,
    _: Annotated[Threads, Depends(get_thread)],
    session: Annotated[AsyncSession, Depends(get_session)],
    thread_id: str,
) -> ToolCallSchema:
    """Validate HIL inputs."""
    response = await session.execute(
        select(Messages)
        .where(Messages.thread_id == thread_id, Messages.entity == Entity.AI_TOOL)
        .order_by(desc(Messages.order))
        .limit(1)
    )
    last_message_calling_tools = response.scalars().one_or_none()
    if not last_message_calling_tools:
        raise HTTPException(status_code=404, detail="Specified tool call not found.")
    last_message_calling_tools_content = json.loads(last_message_calling_tools.content)
    tool_calls = last_message_calling_tools_content["tool_calls"]
    try:
        relevant_tool_call_index = next(
            (
                i
                for i, tool_call in enumerate(tool_calls)
                if tool_call["id"] == user_request.tool_call_id
            )
        )
    except StopIteration:
        raise HTTPException(status_code=404, detail="Specified tool call not found.")

    # Replace the function call with the (potentially) modified one
    tool_calls[relevant_tool_call_index]["function"]["arguments"] = json.dumps(
        user_request.json
    )
    tool_calls[relevant_tool_call_index]["function"]["validated"] = True
    last_message_calling_tools_content["tool_calls"] = tool_calls
    last_message_calling_tools.content = json.dumps(last_message_calling_tools_content)
    await session.commit()
    return ToolCallSchema(
        tool_call_id=tool_calls[relevant_tool_call_index]["id"],
        name=tool_calls[relevant_tool_call_index]["function"]["name"],
        arguments=user_request.json,
    )
