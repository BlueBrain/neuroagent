"""Endpoints for agent's question answering pipeline."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from neuroagent.agents import (
    AgentOutput,
    BaseAgent,
    SimpleChatAgent,
)
from neuroagent.app.dependencies import (
    get_agent,
    get_chat_agent,
    get_connection_string,
    get_user_id,
)
from neuroagent.app.routers.database.schemas import Threads
from neuroagent.app.routers.database.sql import get_object
from neuroagent.app.schemas import AgentRequest

router = APIRouter(
    prefix="/qa", tags=["Run the agent"], dependencies=[Depends(get_user_id)]
)

logger = logging.getLogger(__name__)


@router.post("/run", response_model=AgentOutput)
async def run_agent(
    request: AgentRequest,
    agent: Annotated[BaseAgent, Depends(get_agent)],
) -> AgentOutput:
    """Run agent."""
    logger.info("Running agent query.")
    logger.info(f"User's query: {request.query}")
    return await agent.arun(request.query)


@router.post("/chat/{thread_id}", response_model=AgentOutput)
async def run_chat_agent(
    request: AgentRequest,
    _: Annotated[Threads, Depends(get_object)],
    agent: Annotated[SimpleChatAgent, Depends(get_chat_agent)],
    thread_id: str,
) -> AgentOutput:
    """Run chat agent."""
    logger.info("Running agent query.")
    logger.info(f"User's query: {request.query}")
    return await agent.arun(query=request.query, thread_id=thread_id)


@router.post("/chat_streamed/{thread_id}")
async def run_streamed_chat_agent(
    request: AgentRequest,
    _: Annotated[Threads, Depends(get_object)],
    agent: Annotated[BaseAgent, Depends(get_chat_agent)],
    connection_string: Annotated[str | None, Depends(get_connection_string)],
    thread_id: str,
) -> StreamingResponse:
    """Run agent in streaming mode."""
    logger.info("Running agent query.")
    logger.info(f"User's query: {request.query}")
    return StreamingResponse(
        agent.astream(
            query=request.query,
            thread_id=thread_id,
            connection_string=connection_string,
        )
    )
