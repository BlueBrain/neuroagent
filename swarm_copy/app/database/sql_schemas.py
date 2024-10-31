"""Schemas for the chatbot."""

import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative base for SQLAlchemy."""

    pass


def uuid_to_str() -> str:
    """Turn a uuid into a string."""
    return uuid.uuid4().hex


def utc_now() -> datetime.datetime:
    """Return the utc time."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class Threads(Base):
    """SQL table for the users thread / conversations."""

    __tablename__ = "threads"
    thread_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=uuid_to_str
    )
    vlab_id: Mapped[str] = mapped_column(
        String, default="430108e9-a81d-4b13-b7b6-afca00195908"
    )  # only default for now !
    project_id: Mapped[str] = mapped_column(
        String, default="eff09ea1-be16-47f0-91b6-52a3ea3ee575"
    )  # only default for now !
    title: Mapped[str] = mapped_column(String, default="New chat")
    creation_date: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)
    update_date: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)

    user_id: Mapped[str] = mapped_column(String, nullable=False)
    messages: Mapped[list["Messages"]] = relationship(
        "Messages", back_populates="thread", cascade="all, delete-orphan"
    )


class Messages(Base):
    """SQL table for the messaages in the threads."""

    __tablename__ = "messages"
    message_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=uuid_to_str
    )
    order: Mapped[int] = mapped_column(Integer, nullable=False)
    creation_date: Mapped[datetime.datetime] = mapped_column(DateTime, default=utc_now)
    content: Mapped[str] = mapped_column(String, nullable=False)

    thread_id: Mapped[str] = mapped_column(
        String, ForeignKey("threads.thread_id"), nullable=False
    )
    thread: Mapped["Threads"] = relationship("Threads", back_populates="messages")
