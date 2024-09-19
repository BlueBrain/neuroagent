"""Schemas."""

from pydantic import BaseModel


class AgentRequest(BaseModel):
    """Class for agent request."""

    query: str
