"""Endpoints for agent's question answering pipeline."""

import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from neuroagent.agent_routine import AgentsRoutine
from neuroagent.app.database.db_utils import get_thread, save_history
from neuroagent.app.database.sql_schemas import Entity, Messages, Threads
from neuroagent.app.dependencies import (
    get_agents_routine,
    get_context_variables,
    get_session,
    get_starting_agent,
    get_user_id,
)
from neuroagent.new_types import Agent, AgentRequest, AgentResponse
from neuroagent.stream import stream_agent_response

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


@router.post("/chat/{thread_id}", response_model=AgentResponse)
async def run_chat_agent(
    user_request: AgentRequest,
    agent_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
    session: Annotated[AsyncSession, Depends(get_session)],
    thread: Annotated[Threads, Depends(get_thread)],
) -> AgentResponse:
    """Run a single agent query."""
    # Temporary solution
    context_variables["vlab_id"] = thread.vlab_id
    context_variables["project_id"] = thread.project_id

    messages: list[Messages] = await thread.awaitable_attrs.messages
    messages.append(
        Messages(
            order=len(messages),
            thread_id=thread.thread_id,
            entity=Entity.USER,
            content=json.dumps({"role": "user", "content": user_request.query}),
        )
    )
    response = await agent_routine.arun(agent, messages, context_variables)
    await save_history(
        history=response.messages,
        offset=len(messages),
        thread=thread,
        session=session,
    )
    return AgentResponse(message=response.messages[-1]["content"])


@router.post("/chat_streamed/{thread_id}")
async def stream_chat_agent(
    user_request: AgentRequest,
    agents_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread: Annotated[Threads, Depends(get_thread)],
) -> StreamingResponse:
    """Run a single agent query in a streamed fashion."""
    # Temporary solution
    context_variables["vlab_id"] = thread.vlab_id
    context_variables["project_id"] = thread.project_id
    messages: list[Messages] = await thread.awaitable_attrs.messages

    messages.append({"role": "user", "content": user_request.query})
    stream_generator = stream_agent_response(
        agents_routine,
        agent,
        messages,
        context_variables,
        user_id,
        thread.thread_id,
        session,
    )
    return StreamingResponse(stream_generator, media_type="text/event-stream")
