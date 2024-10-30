"""App dependencies."""

import logging
from functools import cache
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, Request
from fastapi.security import HTTPBearer
from httpx import AsyncClient
from keycloak import KeycloakOpenID
from openai import AsyncOpenAI

from swarm_copy.app.config import Settings
from swarm_copy.new_types import Agent
from swarm_copy.run import AgentsRoutine
from swarm_copy.tools.electrophys_tool import ElectrophysTool
from swarm_copy.tools.get_me_model_tool import GetMEModelTool
from swarm_copy.tools.get_morpho_tool import GetMorphoTool
from swarm_copy.tools.kg_morpho_features_tool import KGMorphoFeatureTool
from swarm_copy.tools.literature_search_tool import LiteratureSearchTool
from swarm_copy.tools.morphology_features_tool import MorphologyFeatureTool
from swarm_copy.tools.resolve_entities_tool import ResolveEntitiesTool
from swarm_copy.tools.traces_tool import GetTracesTool

logger = logging.getLogger(__name__)


class HTTPBearerDirect(HTTPBearer):
    """HTTPBearer class that returns directly the token in the call."""

    async def __call__(self, request: Request) -> str | None:  # type: ignore
        """Intercept the bearer token in the headers."""
        auth_credentials = await super().__call__(request)
        return auth_credentials.credentials if auth_credentials else None


auth = HTTPBearerDirect(auto_error=False)


@cache
def get_settings() -> Settings:
    """Get the global settings."""
    logger.info("Reading the environment and instantiating settings")
    return Settings()


async def get_openai_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[AsyncOpenAI | None]:
    """Get the OpenAi Async client."""
    if not settings.openai.token:
        yield None
    else:
        try:
            client = AsyncOpenAI(api_key=settings.openai.token.get_secret_value())
            yield client
        finally:
            await client.close()


def get_starting_agent(settings: Annotated[Settings, Depends(get_settings)]) -> Agent:
    """Get the starting agent."""
    logger.info(f"Loading model {settings.openai.model}.")
    agent = Agent(
        name="Agent",
        instructions="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
        tools=[
            LiteratureSearchTool,
            ElectrophysTool,
            GetMEModelTool,
            GetMorphoTool,
            KGMorphoFeatureTool,
            MorphologyFeatureTool,
            ResolveEntitiesTool,
            GetTracesTool,
        ],
        model=settings.openai.model,
    )
    return agent


async def get_httpx_client(request: Request) -> AsyncIterator[AsyncClient]:
    """Manage the httpx client for the request."""
    client = AsyncClient(
        timeout=None,
        verify=False,
        headers={"x-request-id": request.headers["x-request-id"]},
    )
    yield client


def get_kg_token(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str | None, Depends(auth)],
) -> str:
    """Get a Knowledge graph token using Keycloak."""
    if token:
        return token
    else:
        instance = KeycloakOpenID(
            server_url=settings.keycloak.server_url,
            realm_name=settings.keycloak.realm,
            client_id=settings.keycloak.client_id,
        )
        return instance.token(
            username=settings.keycloak.username,
            password=settings.keycloak.password.get_secret_value(),  # type: ignore
        )["access_token"]


def get_context_variables(
    settings: Annotated[Settings, Depends(get_settings)],
    starting_agent: Annotated[Agent, Depends(get_starting_agent)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> dict[str, Any]:
    """Get the global context variables to feed the tool's metadata."""
    return {
        "user_id": 1234,
        "starting_agent": starting_agent,
        "section": "soma[0]",
        "offset": 0.5,
        "bluenaas_url": settings.tools.bluenaas.url,
        "token": token,
        "retriever_k": settings.tools.literature.retriever_k,
        "reranker_k": settings.tools.literature.reranker_k,
        "use_reranker": settings.tools.literature.use_reranker,
        "literature_search_url": settings.tools.literature.url,
        "knowledge_graph_url": settings.knowledge_graph.url,
        "me_model_search_size": settings.tools.me_model.search_size,
        "brainregion_path": settings.knowledge_graph.br_saving_path,
        "celltypes_path": settings.knowledge_graph.ct_saving_path,
        "morpho_search_size": settings.tools.kg_morpho_features.search_size,
        "trace_search_size": settings.tools.trace.search_size,
        "kg_sparql_url": settings.knowledge_graph.sparql_url,
        "kg_class_view_url": settings.knowledge_graph.class_view_url,
        "httpx_client": httpx_client,
    }


def get_agents_routine(
    openai: Annotated[AsyncOpenAI | None, Depends(get_openai_client)],
) -> AgentsRoutine:
    """Get the AgentRoutine client."""
    return AgentsRoutine(openai)
