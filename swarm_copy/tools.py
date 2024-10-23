import asyncio
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from openai.lib._tools import pydantic_function_tool
from pydantic import BaseModel, ConfigDict, Field


class BaseMetadata(BaseModel):
    """Base class for metadata"""

    model_config = ConfigDict(extra="ignore")


class BaseTool(BaseModel, ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    metadata: BaseModel
    input_schema: BaseModel

    @classmethod
    def pydantic_to_openai_schema(cls):
        return pydantic_function_tool(
            model=cls.__annotations__.get("input_schema"),
            name=cls.name,
            description=cls.description,
        )

    @abstractmethod
    async def arun(self) -> Any:
        """Run the tool"""


class AccountDetailInput(BaseModel):
    """Inputs for the account detail tool"""

    account_name: str = Field(
        description="Name the the person to whom the account belongs."
    )


class AccountDetailMetadata(BaseModel):
    """Metadata class for the account detail tool"""

    user_id: int


class PrintAccountDetailsTool(BaseTool):
    name: ClassVar[str] = "print-account-details-tool"
    description: ClassVar[str] = "Print the account details"
    input_schema: AccountDetailInput
    metadata: AccountDetailMetadata

    async def arun(self):
        user_id = self.metadata.user_id
        await asyncio.sleep(0.5)
        print(f"Account Details: {self.input_schema.account_name} {user_id}")
        return "Success"
