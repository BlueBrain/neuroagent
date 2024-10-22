from typing import Any, Callable, List, Optional, Union

# Third-party imports
from pydantic import BaseModel, ConfigDict


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[Any] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
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
    agent: Optional[Agent] = None
    context_variables: dict = {}
