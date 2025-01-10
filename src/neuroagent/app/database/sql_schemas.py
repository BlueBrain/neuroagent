"""Schemas for the chatbot."""

import datetime
import enum
import uuid

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime.datetime:
    """Return the utc time."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class Entity(enum.Enum):
    """Calss to restrict entity collumn."""

    USER = "user"
    AI_TOOL = "ai_tool"
    TOOL = "tool"
    AI_MESSAGE = "ai_message"


class Base(AsyncAttrs, DeclarativeBase):
    """Base declarative base for SQLAlchemy."""


class Threads(Base):
    """SQL table for the users thread / conversations."""

    __tablename__ = "threads"
    thread_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: uuid.uuid4().hex
    )
    vlab_id: Mapped[str] = mapped_column(
        String, default="430108e9-a81d-4b13-b7b6-afca00195908", nullable=False
    )  # only default for now !
    project_id: Mapped[str] = mapped_column(
        String, default="eff09ea1-be16-47f0-91b6-52a3ea3ee575", nullable=False
    )  # only default for now !
    title: Mapped[str] = mapped_column(String, default="New chat")
    creation_date: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)
    update_date: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)

    user_id: Mapped[str] = mapped_column(String, nullable=False)
    messages: Mapped[list["Messages"]] = relationship(
        "Messages",
        back_populates="thread",
        order_by="Messages.order",  # get messages in order.
        cascade="all, delete-orphan",
    )


class Messages(Base):
    """SQL table for the messsages in the threads."""

    __tablename__ = "messages"
    message_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: uuid.uuid4().hex
    )
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    creation_date: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)
    entity: Mapped[Entity] = mapped_column(Enum(Entity), nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)

    thread_id: Mapped[str] = mapped_column(
        String, ForeignKey("threads.thread_id"), nullable=False
    )
    thread: Mapped[Threads] = relationship("Threads", back_populates="messages")
    tool_calls: Mapped[list["ToolCalls"]] = relationship(
        "ToolCalls", back_populates="message", cascade="all, delete-orphan"
    )


class ToolCalls(Base):
    """SQL table used for tool call parameters."""

    __tablename__ = "tool_calls"
    tool_call_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    arguments: Mapped[str] = mapped_column(String, nullable=False)
    validated: Mapped[bool] = mapped_column(Boolean, nullable=True)

    message_id: Mapped[str] = mapped_column(String, ForeignKey("messages.message_id"))
    message: Mapped[Messages] = relationship("Messages", back_populates="tool_calls")
