"""Base tool."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from openai.lib._tools import pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict


class BaseMetadata(BaseModel):
    """Base class for metadata."""

    model_config = ConfigDict(extra="ignore")


class BaseTool(BaseModel, ABC):
    """Base class for the tools."""

    name: ClassVar[str]
    description: ClassVar[str]
    metadata: BaseModel
    input_schema: BaseModel

    @classmethod
    def pydantic_to_openai_schema(cls) -> ChatCompletionToolParam:
        """Convert pydantic schema to OpenAI json."""
        return pydantic_function_tool(
            model=cls.__annotations__["input_schema"],
            name=cls.name,
            description=cls.description,
        )

    @abstractmethod
    async def arun(self) -> Any:
        """Run the tool."""


class BaseToolOutput(BaseModel):
    """Base class for tool outputs."""

    def __repr__(self) -> str:
        """Representation method."""
        return self.model_dump_json()
