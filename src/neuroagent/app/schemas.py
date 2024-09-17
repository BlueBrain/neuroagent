"""Schemas."""

from typing import Any

from pydantic import BaseModel


class AgentRequest(BaseModel):
    """Class for agent request."""

    inputs: str
    parameters: dict[str, Any]
