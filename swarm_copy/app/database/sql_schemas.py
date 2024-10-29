"""Schemas for the chatbot."""

import datetime
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, String, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def uuid_to_str() -> str:
    """Turn a uuid into a string."""
    return uuid.uuid4().hex


class Threads(Base):
    """SQL table for the users thread / conversations."""

    __tablename__ = "Threads"
    thread_id = Column(String, primary_key=True, default=uuid_to_str)
    vlab_id = Column(String, default="430108e9-a81d-4b13-b7b6-afca00195908") # only default for now ! 
    project_id = Column(String, default="eff09ea1-be16-47f0-91b6-52a3ea3ee575") # only default for now ! 
    title = Column(String, default="New chat")
    creation_date = Column(DateTime, default=datetime.datetime.now)
    update_date = Column(DateTime, default=datetime.datetime.now)

    user_id = Column(String, nullable=False)
    messages = relationship('Messages', back_populates='thread', cascade="all, delete-orphan")


class Messages(Base):
    __tablename__ = 'Messages'
    message_id = Column(String, primary_key=True, default=uuid_to_str)
    message_order = Column(Integer, nullable=False)
    creation_date = Column(DateTime, default=datetime.datetime.now)
    update_date = Column(DateTime, default=datetime.datetime.now)
    message_content = Column(String, nullable=False)  

    thread_id = Column(String, ForeignKey('Threads.thread_id'), nullable=False)
    thread = relationship('Threads', back_populates='messages')


# class ThreadsUpdate(BaseModel):
#     """Class to update the conversation's title in the db."""

#     title: Optional[str] = None


# class ThreadsRead(BaseModel):
#     """Data class to read chatbot conversations in the db."""

#     thread_id: str
#     user_sub: str
#     vlab_id: str
#     project_id: str
#     title: str = "title"
#     creation_date: datetime.datetime
#     update_date: datetime.datetime = datetime.datetime.now



# class GetThreadsOutput(BaseModel):
#     """Output of the conversation listing crud."""

#     message_id: str
#     entity: Literal["Human", "AI"]
#     message: str


# class ToolCallSchema(BaseModel):
#     """Tool call crud's output."""

#     call_id: str
#     name: str
#     arguments: dict[str, Any]
