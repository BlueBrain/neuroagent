"""Base tool."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from httpx import AsyncClient
from openai.lib._tools import pydantic_function_tool
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

logger = logging.getLogger(__name__)


EtypesLiteral = Literal[
    "bSTUT",
    "dSTUT",
    "bNAC",
    "cSTUT",
    "dNAC",
    "bAC",
    "cIR",
    "cAC",
    "cACint",
    "bIR",
    "cNAC",
    "cAD",
    "cADpyr",
    "cAD_ltb",
    "cNAD_ltb",
    "cAD_noscltb",
    "cNAD_noscltb",
    "dAD_ltb",
    "dNAD_ltb",
]
ETYPE_IDS = {
    "bSTUT": "http://uri.interlex.org/base/ilx_0738200",
    "dSTUT": "http://uri.interlex.org/base/ilx_0738202",
    "bNAC": "http://uri.interlex.org/base/ilx_0738203",
    "cSTUT": "http://uri.interlex.org/base/ilx_0738198",
    "dNAC": "http://uri.interlex.org/base/ilx_0738205",
    "bAC": "http://uri.interlex.org/base/ilx_0738199",
    "cIR": "http://uri.interlex.org/base/ilx_0738204",
    "cAC": "http://uri.interlex.org/base/ilx_0738197",
    "cACint": "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
    "bIR": "http://uri.interlex.org/base/ilx_0738206",
    "cNAC": "http://uri.interlex.org/base/ilx_0738201",
    "cAD": "http://uri.interlex.org/base/ilx_0738207",  # Both are the same id, what's the purpose ?
    "cADpyr": "http://uri.interlex.org/base/ilx_0738207",  # Both are the same id, what's the purpose ?
    "cAD_ltb": "http://uri.interlex.org/base/ilx_0738255",
    "cNAD_ltb": "http://uri.interlex.org/base/ilx_0738254",
    "cAD_noscltb": "http://uri.interlex.org/base/ilx_0738250",
    "cNAD_noscltb": "http://uri.interlex.org/base/ilx_0738249",
    "dAD_ltb": "http://uri.interlex.org/base/ilx_0738258",
    "dNAD_ltb": "http://uri.interlex.org/base/ilx_0738256",
}


class BaseMetadata(BaseModel):
    """Base class for metadata."""

    httpx_client: AsyncClient
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


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


def process_validation_error(error: ValidationError) -> str:
    """Handle validation errors when tool inputs are wrong."""
    error_list = []
    name = error.title
    # We have to iterate, in case there are multiple errors.
    try:
        for err in error.errors():
            if err["type"] == "literal_error":
                error_list.append(
                    {
                        "Validation error": (
                            f'Wrong value: provided {err["input"]} for input'
                            f' {err["loc"][0]}. Try again and change this problematic'
                            " input."
                        )
                    }
                )
            elif err["type"] == "missing":
                error_list.append(
                    {
                        "Validation error": (
                            f'Missing input : {err["loc"][0]}. Try again and add this'
                            " input."
                        )
                    }
                )
            else:
                error_list.append(
                    {"Validation error": f'{err["loc"][0]}. {err["msg"]}'}
                )

    except (KeyError, IndexError) as e:
        error_list.append({"Validation error": f"Error in {name} : {str(e)}"})
        logger.error(
            "UNTREATED ERROR !! PLEASE CONTACT ML TEAM AND FOWARD THEM THE REQUEST !!"
        )

    logger.warning(f"VALIDATION ERROR: Wrong input in {name}. {error_list}")

    return json.dumps(error_list)


def process_tool_error(error: Exception) -> str:
    """Handle errors inside tools."""
    logger.warning(
        f"TOOL ERROR: Error in tool {error.args[1]}. Error: {str(error.args[0])}"
    )
    dict_output = {error.args[1]: error.args[0]}
    return json.dumps(dict_output)


class BasicTool(BaseTool):
    """Basic class for tools."""

    @model_validator(mode="before")
    @classmethod
    def handle_errors(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Instantiate the clients upon class creation."""
        data["handle_validation_error"] = process_validation_error
        data["handle_tool_error"] = process_tool_error
        return data