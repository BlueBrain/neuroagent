"""Base multi-agent."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from pydantic import ConfigDict
from pydantic.v1 import BaseModel

from neuroagent.agents import AgentOutput
from neuroagent.tools.base_tool import BasicTool


class BaseMultiAgent(BaseModel, ABC):
    """Base class for multi agents."""

    llm: BaseLLM | BaseChatModel
    main_agent: Any
    agents: list[tuple[str, list[BasicTool]]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> AgentOutput:
        """Run method of the service."""

    @abstractmethod
    async def arun(self, *args: Any, **kwargs: Any) -> AgentOutput:
        """Arun method of the service."""

    @abstractmethod
    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Astream method of the service."""

    @staticmethod
    @abstractmethod
    def _process_output(*args: Any, **kwargs: Any) -> AgentOutput:
        """Format the output."""
