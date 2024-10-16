"""Base agent."""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, ConfigDict


class AgentStep(BaseModel):
    """Class for agent decision steps."""

    tool_name: str
    arguments: dict[str, Any] | str


class AgentOutput(BaseModel):
    """Class for agent response."""

    response: str
    steps: list[AgentStep]


class BaseAgent(BaseModel, ABC):
    """Base class for services."""

    llm: BaseChatModel
    tools: list[BaseTool]
    agent: Any

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> AgentOutput:
        """Run method of the service."""

    @abstractmethod
    async def arun(self, *args: Any, **kwargs: Any) -> AgentOutput:
        """Arun method of the service."""

    @abstractmethod
    def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Astream method of the service."""

    @staticmethod
    @abstractmethod
    def _process_output(*args: Any, **kwargs: Any) -> AgentOutput:
        """Format the output."""


class AsyncSqliteSaverWithPrefix(AsyncSqliteSaver):
    """Wrapper around the AsyncSqliteSaver that accepts a connection string with prefix."""

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls, conn_string: str
    ) -> AsyncIterator[AsyncSqliteSaver]:
        """Create a new AsyncSqliteSaver instance from a connection string.

        Args:
            conn_string (str): The async SQLite connection string. It can have the 'sqlite+aiosqlite:///' prefix.

        Yields
        ------
            AsyncSqliteSaverWithPrefix: A new connected AsyncSqliteSaverWithPrefix instance.
        """
        conn_string = conn_string.split("///")[-1]
        async with super().from_conn_string(conn_string) as memory:
            yield AsyncSqliteSaverWithPrefix(memory.conn)


class AsyncPostgresSaverWithPrefix(AsyncPostgresSaver):
    """Wrapper around the AsyncSqliteSaver that accepts a connection string with prefix."""

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        pipeline: bool = False,
        serde: SerializerProtocol | None = None,
    ) -> AsyncIterator[AsyncPostgresSaver]:
        """Create a new AsyncPostgresSaver instance from a connection string.

        Args:
            conn_string (str): The async Postgres connection string. It can have the 'postgresql+asyncpg://' prefix.

        Yields
        ------
            AsyncPostgresSaverWithPrefix: A new connected AsyncPostgresSaverWithPrefix instance.
        """
        prefix, body = conn_string.split("://", maxsplit=1)
        currated_prefix = prefix.split("+", maxsplit=1)[0]  # Still works if + not there
        conn_string = currated_prefix + "://" + body
        async with super().from_conn_string(
            conn_string, pipeline=pipeline, serde=serde
        ) as memory:
            yield AsyncPostgresSaverWithPrefix(memory.conn, memory.pipe, memory.serde)
