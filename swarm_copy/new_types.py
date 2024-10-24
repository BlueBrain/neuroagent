"""New types."""

from typing import Callable

# Third-party imports
from pydantic import BaseModel

from swarm_copy.tools import BaseTool


class Agent(BaseModel):
    """Agent class."""

    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str | Callable[[], str] = "You are a helpful agent."
    functions: list[type[BaseTool]] = []
    tool_choice: str | None = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    """Agent response."""

    messages: list = []
    agent: Agent | None = None
    context_variables: dict = {}


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes
    ----------
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Agent | None = None
    context_variables: dict = {}
