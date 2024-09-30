"""Base agent."""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
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
    ) -> AsyncIterator["AsyncSqliteSaver"]:
        """Create a new AsyncSqliteSaver instance from a connection string.

        Args:
            conn_string (str): The SQLite connection string. It can have the 'sqlite:///' prefix.

        Yields
        ------
            AsyncSqliteSaverWithPrefix: A new AsyncSqliteSaverWithPrefix instance.
        """
        conn_string = conn_string.split("///")[-1]
        async with super().from_conn_string(conn_string) as memory:
            yield AsyncSqliteSaverWithPrefix(memory.conn)
