"""Schemas for the chatbot."""

import datetime
import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def uuid_to_str() -> str:
    """Turn a uuid into a string."""
    return uuid.uuid4().hex


class Threads(Base):
    """SQL table for the users thread / conversations."""

    __tablename__ = "threads"
    thread_id = Column(String, primary_key=True, default=uuid_to_str)
    vlab_id = Column(
        String, default="430108e9-a81d-4b13-b7b6-afca00195908"
    )  # only default for now !
    project_id = Column(
        String, default="eff09ea1-be16-47f0-91b6-52a3ea3ee575"
    )  # only default for now !
    title = Column(String, default="New chat")
    creation_date = Column(DateTime, default=datetime.datetime.now)
    update_date = Column(DateTime, default=datetime.datetime.now)

    user_id = Column(String, nullable=False)
    messages = relationship(
        "Messages", back_populates="thread", cascade="all, delete-orphan"
    )


class Messages(Base):
    """SQL table for the messaages in the threads."""

    __tablename__ = "messages"
    message_id = Column(String, primary_key=True, default=uuid_to_str)
    order = Column(Integer, nullable=False)
    creation_date = Column(DateTime, default=datetime.datetime.now)
    content = Column(String, nullable=False)

    thread_id = Column(String, ForeignKey("threads.thread_id"), nullable=False)
    thread = relationship("Threads", back_populates="messages")
