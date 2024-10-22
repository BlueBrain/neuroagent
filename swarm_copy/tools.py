from typing import Any, ClassVar

from openai.lib._tools import pydantic_function_tool
from pydantic import BaseModel, Field


class BaseTool(BaseModel):
    name: str
    description: str
    input_schema: type[BaseModel]
    metadata: dict[str, Any]

    @classmethod
    def pydantic_to_openai_schema(cls):
        return pydantic_function_tool(
            model=cls.__annotations__.get("input_schema"),
            name=cls.name,
            description=cls.description,
        )

    def run(self) -> Any:
        """Run the tool"""
        pass


class AccountDetailInput(BaseModel):
    """Inputs for the account detail tool"""

    account_name: str = Field(
        description="Name the the person to whom the account belongs."
    )


class PrintAccountDetailsTool(BaseTool):
    name: ClassVar[str] = "print-account-details-tool"
    description: ClassVar[str] = "Print the account details"
    input_schema: AccountDetailInput

    def run(self):
        user_id = self.metadata.get("user_id", None)
        print(f"Account Details: {self.input_schema.account_name} {user_id}")
        return "Success"
