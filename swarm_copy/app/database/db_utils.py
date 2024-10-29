import json 

from typing import Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import  AsyncSession
from swarm_copy.app.database.sql_schemas import Messages

async def put_messages_in_db(history: list[dict[str, Any]], thread_id: str, session: AsyncSession) -> None:
    # Add messages to DB.
    for i, messages in enumerate(history):
        new_msg = Messages(
            message_order=i,
            thread_id=thread_id,
            message_content=json.dumps(messages),
        )
        session.add(new_msg)
        await session.commit()
        await session.refresh(new_msg)

async def get_messages_from_db(thread_id: str,  session: AsyncSession) -> list[dict[str, Any] | None ]:
    # get messages from history.
    message_query = await session.execute(
        select(Messages).where(Messages.thread_id == thread_id)
    )
    db_messages = message_query.scalars().all()

    messages = []
    if db_messages:
        for msg in db_messages:
            messages.append(json.loads(msg.message_content))
    if messages:
        return messages
    else:
        return []