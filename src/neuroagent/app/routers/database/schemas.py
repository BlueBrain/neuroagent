"""Schemas for the chatbot."""

import datetime
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def uuid_to_str() -> str:
    """Turn a uuid into a string."""
    return uuid.uuid4().hex


class Threads(Base):
    """SQL table for the chatbot's user."""

    __tablename__ = "Threads"
    # langgraph's tables work with strings, so we must too
    thread_id = Column(String, primary_key=True, default=uuid_to_str)
    user_sub = Column(String, default=None, primary_key=True)
    title = Column(String, default="title")
    timestamp = Column(DateTime, default=datetime.datetime.now)


class ThreadsUpdate(BaseModel):
    """Class to update the conversation's title in the db."""

    title: Optional[str] = None


class ThreadsRead(BaseModel):
    """Data class to read chatbot conversations in the db."""

    thread_id: str
    user_sub: str
    title: str = "title"
    timestamp: datetime.datetime = datetime.datetime.now()


class GetThreadsOutput(BaseModel):
    """Output of the conversation listing crud."""

    message_id: str
    entity: Literal["Human", "AI"]
    message: str


class ToolCallSchema(BaseModel):
    """Tool call crud's output."""

    call_id: str
    name: str
    arguments: dict[str, Any]
