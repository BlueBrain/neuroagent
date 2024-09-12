"""Base tool (to handle errors)."""

import json
import logging
from typing import Any

from langchain_core.pydantic_v1 import ValidationError
from langchain_core.tools import BaseTool, ToolException
from pydantic.v1 import BaseModel, root_validator

logger = logging.getLogger(__name__)


def process_validation_error(error: ValidationError) -> str:
    """Handle validation errors when tool inputs are wrong."""
    error_list = []

    # not happy with this solution but it is to extract the name of the input class
    name = str(error.model).split(".")[-1].strip(">")
    # We have to iterate, in case there are multiple errors.
    try:
        for err in error.errors():
            if "ctx" in err:
                error_list.append(
                    {
                        "Validation error": (
                            f'Wrong value: {err["ctx"]["given"]} for input'
                            f' {err["loc"][0]}. Try again and change this problematic'
                            " input."
                        )
                    }
                )
            elif "loc" in err and err["msg"] == "field required":
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

    @root_validator(pre=True)
    def handle_errors(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Instantiate the clients upon class creation."""
        values["handle_validation_error"] = process_validation_error
        values["handle_tool_error"] = process_tool_error
        return values


class BaseToolOutput(BaseModel):
    """Base class for tool outputs."""

    def __repr__(self) -> str:
        """Representation method."""
        return self.json()
