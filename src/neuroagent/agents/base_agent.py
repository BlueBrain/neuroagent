"""Base agent."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
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
