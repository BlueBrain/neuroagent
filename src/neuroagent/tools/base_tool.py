"""Base tool (to handle errors)."""

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ValidationError, model_validator

logger = logging.getLogger(__name__)


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


def process_tool_error(error: ToolException) -> str:
    """Handle errors inside tools."""
    logger.warning(
        f"TOOL ERROR: Error in tool {error.args[1]}. Error: {str(error.args[0])}"
    )
    dict_output = {error.args[1]: error.args[0]}
    return json.dumps(dict_output)


class BasicTool(BaseTool):
    """Basic class for tools."""

    name: str = "base"
    description: str = "Base tool from which regular tools should inherit."

    @model_validator(mode="before")
    @classmethod
    def handle_errors(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Instantiate the clients upon class creation."""
        data["handle_validation_error"] = process_validation_error
        data["handle_tool_error"] = process_tool_error
        return data


class BaseToolOutput(BaseModel):
    """Base class for tool outputs."""

    def __repr__(self) -> str:
        """Representation method."""
        return self.model_dump_json()
