"""Endpoints for agent's question answering pipeline."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from swarm_copy.agent_routine import AgentsRoutine
from swarm_copy.app.database.db_utils import get_history, get_thread, save_history
from swarm_copy.app.database.sql_schemas import Threads
from swarm_copy.app.dependencies import (
    get_agents_routine,
    get_context_variables,
    get_session,
    get_starting_agent,
    get_user_id,
)
from swarm_copy.new_types import Agent, AgentRequest, AgentResponse
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


@router.post("/chat/{thread_id}", response_model=AgentResponse)
async def run_chat_agent(
    user_request: AgentRequest,
    agent_routine: Annotated[AgentsRoutine, Depends(get_agents_routine)],
    agent: Annotated[Agent, Depends(get_starting_agent)],
    context_variables: Annotated[dict[str, Any], Depends(get_context_variables)],
    session: Annotated[AsyncSession, Depends(get_session)],
    user_id: Annotated[str, Depends(get_user_id)],
    thread: Annotated[Threads, Depends(get_thread)],
    messages: Annotated[list[dict[str, Any]], Depends(get_history)],
) -> AgentResponse:
    """Run a single agent query."""
    # Temporary solution
    context_variables["vlab_id"] = thread.vlab_id
    context_variables["project_id"] = thread.project_id

    messages.append({"role": "user", "content": user_request.query})
    response = await agent_routine.arun(agent, messages, context_variables)
    await save_history(
        user_id=user_id,
        history=response.messages,
        offset=len(messages) - 1,
        thread_id=thread.thread_id,
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
    messages: Annotated[list[dict[str, Any]], Depends(get_history)],
) -> StreamingResponse:
    """Run a single agent query in a streamed fashion."""
    # Temporary solution
    context_variables["vlab_id"] = thread.vlab_id
    context_variables["project_id"] = thread.project_id

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
