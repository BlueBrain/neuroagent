"""Pydantic schemas for the database operations."""

import datetime
from typing import Any, Literal

from pydantic import BaseModel

from neuroagent.app.database.sql_schemas import Entity


class ThreadsRead(BaseModel):
    """Data class to read chatbot conversations in the db."""

    thread_id: str
    user_id: str
    vlab_id: str
    project_id: str
    title: str
    creation_date: datetime.datetime
    update_date: datetime.datetime


class ThreadUpdate(BaseModel):
    """Data class for the update of a thread."""

    title: str


class MessagesRead(BaseModel):
    """Output of the conversation listing crud."""

    message_id: str
    order: int
    creation_date: datetime.datetime
    msg_content: str
    entity: Literal[Entity.USER, Entity.AI_MESSAGE]


class ToolCallSchema(BaseModel):
    """Tool call crud's output."""

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
