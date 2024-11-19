"""App dependencies."""

import logging
from functools import cache
from typing import Annotated, Any, AsyncIterator

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from httpx import AsyncClient, HTTPStatusError
from keycloak import KeycloakOpenID
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from starlette.status import HTTP_401_UNAUTHORIZED


from swarm_copy.app.app_utils import validate_project
from swarm_copy.app.config import Settings
from swarm_copy.app.database.sql_schemas import Threads
from swarm_copy.cell_types import CellTypesMeta
from swarm_copy.new_types import Agent
from swarm_copy.run import AgentsRoutine
from swarm_copy.tools import (
    ElectrophysFeatureTool,
    GetMorphoTool,
    GetTracesTool,
    KGMorphoFeatureTool,
    LiteratureSearchTool,
    MEModelGetAllTool,
    MEModelGetOneTool,
    MorphologyFeatureTool,
    NowTool,
    ResolveEntitiesTool,
    SCSGetAllTool,
    SCSGetOneTool,
    SCSPostTool,
)
from swarm_copy.utils import RegionMeta, get_file_from_KG

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


async def get_httpx_client(request: Request) -> AsyncIterator[AsyncClient]:
    """Manage the httpx client for the request."""
    client = AsyncClient(
        timeout=None,
        verify=False,
        headers={"x-request-id": request.headers["x-request-id"]},
    )
    try:
        yield client
    finally:
        await client.aclose()


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


def get_connection_string(
    settings: Annotated[Settings, Depends(get_settings)],
) -> str | None:
    """Get the db interacting class for chat agent."""
    if settings.db.prefix:
        connection_string = settings.db.prefix
        if settings.db.user and settings.db.password:
            # Add authentication.
            connection_string += (
                f"{settings.db.user}:{settings.db.password.get_secret_value()}@"
            )
        if settings.db.host:
            # Either in file DB or connect to remote host.
            connection_string += settings.db.host
        if settings.db.port:
            # Add the port if remote host.
            connection_string += f":{settings.db.port}"
        if settings.db.name:
            # Add database name if specified.
            connection_string += f"/{settings.db.name}"
        return connection_string
    else:
        return None


def get_engine(request: Request) -> AsyncEngine | None:
    """Get the SQL engine."""
    return request.app.state.engine


async def get_session(
    engine: Annotated[AsyncEngine | None, Depends(get_engine)],
) -> AsyncIterator[AsyncSession]:
    """Yield a session per request."""
    if not engine:
        raise HTTPException(
            status_code=500,
            detail={
                "detail": "Couldn't connect to the SQL DB.",
            },
        )
    async with AsyncSession(engine) as session:
        yield session


async def get_user_id(
    token: Annotated[str, Depends(auth)],
    settings: Annotated[Settings, Depends(get_settings)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> str:
    """Validate JWT token and returns user ID."""
    if settings.keycloak.validate_token:
        if settings.keycloak.user_info_endpoint:
            try:
                response = await httpx_client.get(
                    settings.keycloak.user_info_endpoint,
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                user_info = response.json()
                return user_info["sub"]
            except HTTPStatusError:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED, detail="Invalid token."
                )
        else:
            raise HTTPException(status_code=404, detail="user info url not provided.")
    else:
        return "dev"


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


async def get_vlab_and_project(
    user_id: Annotated[str, Depends(get_user_id)],
    session: Annotated[AsyncSession, Depends(get_session)],
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
    token: Annotated[str, Depends(get_kg_token)],
) -> dict[str, str]:
    """Get the current vlab and project ID."""
    if "x-project-id" in request.headers and "x-virtual-lab-id" in request.headers:
        vlab_and_project = {
            "vlab_id": request.headers["x-virtual-lab-id"],
            "project_id": request.headers["x-project-id"],
        }
    elif not settings.keycloak.validate_token:
        vlab_and_project = {
            "vlab_id": "32c83739-f39c-49d1-833f-58c981ebd2a2",
            "project_id": "123251a1-be18-4146-87b5-5ca2f8bfaf48",
        }
    else:
        thread_id = request.path_params.get("thread_id")
        thread_result = await session.execute(
            select(Threads).where(
                Threads.user_id == user_id, Threads.thread_id == thread_id
            )
        )
        thread = thread_result.scalars().one_or_none()
        if not thread:
            raise HTTPException(
                status_code=404,
                detail={
                    "detail": "Thread not found.",
                },
            )
        if thread and thread.vlab_id and thread.project_id:
            vlab_and_project = {
                "vlab_id": thread.vlab_id,
                "project_id": thread.project_id,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="thread not found when trying to validate project ID.",
            )

    await validate_project(
        httpx_client=httpx_client,
        vlab_id=vlab_and_project["vlab_id"],
        project_id=vlab_and_project["project_id"],
        token=token,
        vlab_project_url=settings.virtual_lab.get_project_url,
    )
    return vlab_and_project


def get_starting_agent(
    # _: Annotated[None, Depends(get_vlab_and_project)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Agent:
    """Get the starting agent."""
    logger.info(f"Loading model {settings.openai.model}.")
    agent = Agent(
        name="Agent",
        instructions="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
        tools=[
            SCSGetAllTool,
            SCSGetOneTool,
            SCSPostTool,
            MEModelGetAllTool,
            MEModelGetOneTool,
            LiteratureSearchTool,
            ElectrophysFeatureTool,
            GetMorphoTool,
            KGMorphoFeatureTool,
            MorphologyFeatureTool,
            ResolveEntitiesTool,
            GetTracesTool,
            NowTool,
        ],
        model=settings.openai.model,
    )
    return agent


def get_context_variables(
    settings: Annotated[Settings, Depends(get_settings)],
    starting_agent: Annotated[Agent, Depends(get_starting_agent)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> dict[str, Any]:
    """Get the global context variables to feed the tool's metadata."""
    return {
        "starting_agent": starting_agent,
        "token": token,
        "vlab_id": "32c83739-f39c-49d1-833f-58c981ebd2a2",  # New god account vlab. Replaced by actual id in endpoint for now. Meant for usage without history
        "project_id": "123251a1-be18-4146-87b5-5ca2f8bfaf48",  # New god account proj. Replaced by actual id in endpoint for now. Meant for usage without history
        "retriever_k": settings.tools.literature.retriever_k,
        "reranker_k": settings.tools.literature.reranker_k,
        "use_reranker": settings.tools.literature.use_reranker,
        "literature_search_url": settings.tools.literature.url,
        "knowledge_graph_url": settings.knowledge_graph.url,
        "me_model_search_size": settings.tools.me_model.search_size,
        "brainregion_path": settings.knowledge_graph.br_saving_path,
        "celltypes_path": settings.knowledge_graph.ct_saving_path,
        "morpho_search_size": settings.tools.morpho.search_size,
        "kg_morpho_feature_search_size": settings.tools.kg_morpho_features.search_size,
        "trace_search_size": settings.tools.trace.search_size,
        "kg_sparql_url": settings.knowledge_graph.sparql_url,
        "kg_class_view_url": settings.knowledge_graph.class_view_url,
        "bluenaas_url": settings.tools.bluenaas.url,
        "httpx_client": httpx_client,
    }


async def get_redis_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[redis.Redis]:
    """Get the Redis client."""
    pool = redis.ConnectionPool.from_url(settings.hil.redis_uri)
    client = redis.Redis.from_pool(pool)

    try:
        yield client
    finally:
        await client.aclose()


def get_agents_routine(
    openai: Annotated[AsyncOpenAI | None, Depends(get_openai_client)],
    redis_client: Annotated[redis.Redis, Depends(get_redis_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgentsRoutine:
    """Get the AgentRoutine client."""
    return AgentsRoutine(
        redis_client=redis_client,
        client=openai,
        poll_interval=settings.hil.poll_interval,
        ttl=settings.hil.ttl,
    )


async def get_update_kg_hierarchy(
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    file_name: str = "brainregion.json",
) -> None:
    """Query file from KG and update the local hierarchy file."""
    file_url = f"<{settings.knowledge_graph.hierarchy_url}/brainregion>"
    KG_hierarchy = await get_file_from_KG(
        file_url=file_url,
        file_name=file_name,
        view_url=settings.knowledge_graph.sparql_url,
        token=token,
        httpx_client=httpx_client,
    )
    RegionMeta_temp = RegionMeta.from_KG_dict(KG_hierarchy)
    RegionMeta_temp.save_config(settings.knowledge_graph.br_saving_path)
    logger.info("Knowledge Graph Brain Regions Hierarchy file updated.")


async def get_cell_types_kg_hierarchy(
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    file_name: str = "celltypes.json",
) -> None:
    """Query file from KG and update the local hierarchy file."""
    file_url = f"<{settings.knowledge_graph.hierarchy_url}/celltypes>"
    hierarchy = await get_file_from_KG(
        file_url=file_url,
        file_name=file_name,
        view_url=settings.knowledge_graph.sparql_url,
        token=token,
        httpx_client=httpx_client,
    )
    celltypesmeta = CellTypesMeta.from_dict(hierarchy)
    celltypesmeta.save_config(settings.knowledge_graph.ct_saving_path)
    logger.info("Knowledge Graph Cell Types Hierarchy file updated.")
