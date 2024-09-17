"""Base agent."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict

BASE_PROMPT = ChatPromptTemplate(
    input_variables=["agent_scratchpad", "input"],
    input_types={
        "chat_history": list[
            AIMessage
            | HumanMessage
            | ChatMessage
            | SystemMessage
            | FunctionMessage
            | ToolMessage
        ],
        "agent_scratchpad": list[
            AIMessage
            | HumanMessage
            | ChatMessage
            | SystemMessage
            | FunctionMessage
            | ToolMessage
        ],
    },
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="""You are a helpful assistant helping scientists with neuro-scientific questions.
                You must always specify in your answers from which brain regions the information is extracted.
                Do no blindly repeat the brain region requested by the user, use the output of the tools instead.""",
            )
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(input_variables=["input"], template="{input}")
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)


class AgentStep(BaseModel):
    """Class for agent decision steps."""

    tool_name: str
    arguments: dict[str, Any] | str


class AgentOutput(BaseModel):
    """Class for agent response."""

    response: str
    steps: list[AgentStep]
    plan: str | None = None


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
    async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[str]:
        """Astream method of the service."""

    @staticmethod
    @abstractmethod
    def _process_output(*args: Any, **kwargs: Any) -> AgentOutput:
        """Format the output."""
