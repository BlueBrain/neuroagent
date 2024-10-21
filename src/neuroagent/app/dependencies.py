"""Dependencies."""

import logging
from functools import cache
from typing import Annotated, Any, AsyncIterator

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from httpx import AsyncClient, HTTPStatusError
from keycloak import KeycloakOpenID
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from starlette.status import HTTP_401_UNAUTHORIZED

from neuroagent.agents import (
    BaseAgent,
    SimpleAgent,
    SimpleChatAgent,
)
from neuroagent.agents.base_agent import (
    AsyncPostgresSaverWithPrefix,
    AsyncSqliteSaverWithPrefix,
)
from neuroagent.app.app_utils import validate_project
from neuroagent.app.config import Settings
from neuroagent.app.routers.database.schemas import Threads
from neuroagent.cell_types import CellTypesMeta
from neuroagent.multi_agents import BaseMultiAgent, SupervisorMultiAgent
from neuroagent.tools import (
    BlueNaaSTool,
    ElectrophysFeatureTool,
    GetMEModelTool,
    GetMorphoTool,
    GetTracesTool,
    KGMorphoFeatureTool,
    LiteratureSearchTool,
    MorphologyFeatureTool,
    ResolveEntitiesTool,
)
from neuroagent.utils import RegionMeta, get_file_from_KG

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
    """Load all parameters.

    Note that this function is cached and environment will be read just once.
    """
    logger.info("Reading the environment and instantiating settings")
    return Settings()


async def get_httpx_client(request: Request) -> AsyncIterator[AsyncClient]:
    """Manage the httpx client for the request."""
    client = AsyncClient(
        timeout=None,
        verify=False,
        headers={"x-request-id": request.headers["x-request-id"]},
    )
    yield client


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
    if settings.keycloak.validate_token and settings.keycloak.user_info_endpoint:
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


def get_bluenaas_tool(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> BlueNaaSTool:
    """Load BlueNaaS tool."""
    tool = BlueNaaSTool(
        metadata={
            "url": settings.tools.bluenaas.url,
            "token": token,
            "httpx_client": httpx_client,
        }
    )
    return tool


def get_literature_tool(
    token: Annotated[str, Depends(get_kg_token)],
    settings: Annotated[Settings, Depends(get_settings)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> LiteratureSearchTool:
    """Load literature tool."""
    tool = LiteratureSearchTool(
        metadata={
            "url": settings.tools.literature.url,
            "httpx_client": httpx_client,
            "token": token,
            "retriever_k": settings.tools.literature.retriever_k,
            "reranker_k": settings.tools.literature.reranker_k,
            "use_reranker": settings.tools.literature.use_reranker,
        }
    )
    return tool


def get_entities_resolver_tool(
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResolveEntitiesTool:
    """Load resolve brain region tool."""
    tool = ResolveEntitiesTool(
        metadata={
            "token": token,
            "httpx_client": httpx_client,
            "kg_sparql_url": settings.knowledge_graph.sparql_url,
            "kg_class_view_url": settings.knowledge_graph.class_view_url,
        }
    )
    return tool


def get_morpho_tool(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> GetMorphoTool:
    """Load get morpho tool."""
    tool = GetMorphoTool(
        metadata={
            "url": settings.knowledge_graph.url,
            "token": token,
            "httpx_client": httpx_client,
            "search_size": settings.tools.morpho.search_size,
            "brainregion_path": settings.knowledge_graph.br_saving_path,
            "celltypes_path": settings.knowledge_graph.ct_saving_path,
        }
    )
    return tool


def get_kg_morpho_feature_tool(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> KGMorphoFeatureTool:
    """Load knowledge graph tool."""
    tool = KGMorphoFeatureTool(
        metadata={
            "url": settings.knowledge_graph.url,
            "token": token,
            "httpx_client": httpx_client,
            "search_size": settings.tools.kg_morpho_features.search_size,
            "brainregion_path": settings.knowledge_graph.br_saving_path,
        }
    )
    return tool


def get_traces_tool(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> GetTracesTool:
    """Load knowledge graph tool."""
    tool = GetTracesTool(
        metadata={
            "url": settings.knowledge_graph.url,
            "token": token,
            "httpx_client": httpx_client,
            "search_size": settings.tools.trace.search_size,
            "brainregion_path": settings.knowledge_graph.br_saving_path,
        }
    )
    return tool


def get_electrophys_feature_tool(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> ElectrophysFeatureTool:
    """Load morphology features tool."""
    tool = ElectrophysFeatureTool(
        metadata={
            "url": settings.knowledge_graph.url,
            "token": token,
            "httpx_client": httpx_client,
        }
    )
    return tool


def get_morphology_feature_tool(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> MorphologyFeatureTool:
    """Load morphology features tool."""
    tool = MorphologyFeatureTool(
        metadata={
            "url": settings.knowledge_graph.url,
            "token": token,
            "httpx_client": httpx_client,
        }
    )
    return tool


def get_me_model_tool(
    settings: Annotated[Settings, Depends(get_settings)],
    token: Annotated[str, Depends(get_kg_token)],
    httpx_client: Annotated[AsyncClient, Depends(get_httpx_client)],
) -> GetMEModelTool:
    """Load get ME model tool."""
    tool = GetMEModelTool(
        metadata={
            "url": settings.knowledge_graph.url,
            "token": token,
            "httpx_client": httpx_client,
            "search_size": settings.tools.me_model.search_size,
            "brainregion_path": settings.knowledge_graph.br_saving_path,
            "celltypes_path": settings.knowledge_graph.ct_saving_path,
        }
    )
    return tool


def get_language_model(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatOpenAI:
    """Get the language model."""
    logger.info(f"OpenAI selected. Loading model {settings.openai.model}.")
    return ChatOpenAI(
        model_name=settings.openai.model,
        temperature=settings.openai.temperature,
        openai_api_key=settings.openai.token.get_secret_value(),  # type: ignore
        max_tokens=settings.openai.max_tokens,
        seed=78,
        streaming=True,
    )


async def get_agent_memory(
    connection_string: Annotated[str | None, Depends(get_connection_string)],
) -> AsyncIterator[BaseCheckpointSaver[Any] | None]:
    """Get the agent checkpointer."""
    if connection_string:
        if connection_string.startswith("sqlite"):
            async with AsyncSqliteSaverWithPrefix.from_conn_string(
                connection_string
            ) as memory:
                await memory.setup()
                yield memory
                await memory.conn.close()

        elif connection_string.startswith("postgresql"):
            async with AsyncPostgresSaverWithPrefix.from_conn_string(
                connection_string
            ) as memory:
                await memory.setup()
                yield memory
                await memory.conn.close()
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "details": (
                        f"Database of type {connection_string.split(':')[0]} is not"
                        " supported."
                    )
                },
            )
    else:
        logger.warning("The SQL db_prefix needs to be set to use the SQL DB.")
        yield None


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
            "vlab_id": "430108e9-a81d-4b13-b7b6-afca00195908",
            "project_id": "eff09ea1-be16-47f0-91b6-52a3ea3ee575",
        }
    else:
        thread_id = request.path_params.get("thread_id")
        thread = await session.get(Threads, (thread_id, user_id))
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


def get_agent(
    _: Annotated[None, Depends(get_vlab_and_project)],
    llm: Annotated[ChatOpenAI, Depends(get_language_model)],
    literature_tool: Annotated[LiteratureSearchTool, Depends(get_literature_tool)],
    entities_resolver_tool: Annotated[
        ResolveEntitiesTool, Depends(get_entities_resolver_tool)
    ],
    morpho_tool: Annotated[GetMorphoTool, Depends(get_morpho_tool)],
    morphology_feature_tool: Annotated[
        MorphologyFeatureTool, Depends(get_morphology_feature_tool)
    ],
    kg_morpho_feature_tool: Annotated[
        KGMorphoFeatureTool, Depends(get_kg_morpho_feature_tool)
    ],
    electrophys_feature_tool: Annotated[
        ElectrophysFeatureTool, Depends(get_electrophys_feature_tool)
    ],
    traces_tool: Annotated[GetTracesTool, Depends(get_traces_tool)],
    me_model_tool: Annotated[GetMEModelTool, Depends(get_me_model_tool)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> BaseAgent | BaseMultiAgent:
    """Get the generative question answering service."""
    if settings.agent.model == "multi":
        logger.info("Load multi-agent chat")
        tools_list = [
            ("literature", [literature_tool]),
            (
                "morphologies",
                [
                    entities_resolver_tool,
                    morpho_tool,
                    morphology_feature_tool,
                    kg_morpho_feature_tool,
                ],
            ),
            ("traces", [entities_resolver_tool, electrophys_feature_tool, traces_tool]),
        ]
        return SupervisorMultiAgent(llm=llm, agents=tools_list)  # type: ignore
    else:
        tools = [
            literature_tool,
            entities_resolver_tool,
            morpho_tool,
            morphology_feature_tool,
            kg_morpho_feature_tool,
            electrophys_feature_tool,
            traces_tool,
            me_model_tool,
        ]
        logger.info("Load simple agent")
        return SimpleAgent(llm=llm, tools=tools)  # type: ignore


def get_chat_agent(
    _: Annotated[None, Depends(get_vlab_and_project)],
    llm: Annotated[ChatOpenAI, Depends(get_language_model)],
    memory: Annotated[BaseCheckpointSaver[Any], Depends(get_agent_memory)],
    bluenaas_tool: Annotated[BlueNaaSTool, Depends(get_bluenaas_tool)],
    literature_tool: Annotated[LiteratureSearchTool, Depends(get_literature_tool)],
    entities_resolver_tool: Annotated[
        ResolveEntitiesTool, Depends(get_entities_resolver_tool)
    ],
    morpho_tool: Annotated[GetMorphoTool, Depends(get_morpho_tool)],
    morphology_feature_tool: Annotated[
        MorphologyFeatureTool, Depends(get_morphology_feature_tool)
    ],
    me_model_tool: Annotated[GetMEModelTool, Depends(get_me_model_tool)],
    kg_morpho_feature_tool: Annotated[
        KGMorphoFeatureTool, Depends(get_kg_morpho_feature_tool)
    ],
    electrophys_feature_tool: Annotated[
        ElectrophysFeatureTool, Depends(get_electrophys_feature_tool)
    ],
    traces_tool: Annotated[GetTracesTool, Depends(get_traces_tool)],
) -> BaseAgent:
    """Get the generative question answering service."""
    logger.info("Load simple chat")
    tools = [
        bluenaas_tool,
        literature_tool,
        entities_resolver_tool,
        me_model_tool,
        morpho_tool,
        morphology_feature_tool,
        kg_morpho_feature_tool,
        electrophys_feature_tool,
        traces_tool,
    ]
    return SimpleChatAgent(llm=llm, tools=tools, memory=memory)  # type: ignore


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
