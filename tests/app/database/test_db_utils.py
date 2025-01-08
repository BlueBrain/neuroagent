import json

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from neuroagent.app.app_utils import setup_engine
from neuroagent.app.config import Settings
from neuroagent.app.database.db_utils import get_thread, save_history
from neuroagent.app.database.sql_schemas import Base, Entity, Messages, Threads
from neuroagent.app.dependencies import get_session


@pytest.mark.asyncio
@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
async def test_get_thread(patch_required_env, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    valid_thread_id = "test_thread_id"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=valid_thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()

    try:
        thread = await get_thread(
            user_id=user_id,
            thread_id=valid_thread_id,
            session=session,
        )
        assert thread.user_id == user_id
        assert thread.thread_id == valid_thread_id
        assert thread.title == "test_title"
    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
async def test_get_thread_invalid_thread_id(patch_required_env, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    valid_thread_id = "test_thread_id"
    invalid_thread_id = "wrong_thread_id"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=valid_thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()

    try:
        with pytest.raises(HTTPException) as exc_info:
            await get_thread(
                user_id=user_id,
                thread_id=invalid_thread_id,
                session=session,
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["detail"] == "Thread not found."
    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.httpx_mock(can_send_already_matched_responses=True)
async def test_get_thread_invalid_user_id(patch_required_env, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    valid_thread_id = "test_thread_id"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=valid_thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()

    try:
        with pytest.raises(HTTPException) as exc_info:
            await get_thread(
                user_id="wrong_user",
                thread_id=valid_thread_id,
                session=session,
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["detail"] == "Thread not found."

    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_history(patch_required_env, db_connection):
    test_settings = Settings(db={"prefix": db_connection})
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    thread_id = "test_thread_id"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    try:
        history = [
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "AI message"},
        ]
        await save_history(
            history=history, offset=0, thread=new_thread, session=session
        )

        result = await session.execute(
            select(Messages).where(Messages.thread_id == thread_id)
        )
        messages = result.scalars().all()

        assert len(messages) == len(history)
        assert messages[0].entity == Entity.USER
        assert messages[0].content == json.dumps(history[0])
        assert messages[1].entity == Entity.AI_MESSAGE
        assert messages[1].content == json.dumps(history[1])

        updated_thread = await get_thread(
            user_id=user_id, thread_id=thread_id, session=session
        )
        assert updated_thread.update_date is not None

    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_history_with_tool_messages(patch_required_env, db_connection):
    test_settings = Settings(db={"prefix": db_connection})
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    thread_id = "test_thread_id"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    try:
        history = [
            {"role": "tool", "content": "Tool invoked"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "arguments": '{"Hi": "Hello"}',
                            "name": "great-name",
                        },
                        "id": "call_id123",
                        "type": "function",
                    }
                ],
            },
        ]
        await save_history(
            history=history, offset=0, thread=new_thread, session=session
        )

        result = await session.execute(
            select(Messages).where(Messages.thread_id == thread_id)
        )
        messages = result.scalars().all()

        assert len(messages) == len(history)
        assert messages[0].entity == Entity.TOOL
        assert messages[0].content == json.dumps(history[0])
        assert messages[1].entity == Entity.AI_TOOL
        assert messages[1].content == json.dumps(history[1])

    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_history_invalid_message_entity(patch_required_env, db_connection):
    test_settings = Settings(db={"prefix": db_connection})
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    thread_id = "test_thread_id"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()

    try:
        history = [{"role": "unknown", "content": "Invalid entity message"}]

        with pytest.raises(HTTPException) as exc_info:
            await save_history(
                history=history, offset=0, thread=new_thread, session=session
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Unknown message entity."

    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_history_with_offset(patch_required_env, db_connection):
    test_settings = Settings(db={"prefix": db_connection})
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    thread_id = "test_thread_id"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title",
    )
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    try:
        history = [
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First AI message"},
        ]
        await save_history(
            history=history, offset=5, thread=new_thread, session=session
        )

        result = await session.execute(
            select(Messages).where(Messages.thread_id == thread_id)
        )
        messages = result.scalars().all()

        assert len(messages) == len(history)
        assert messages[0].order == 5
        assert messages[0].content == json.dumps(history[0])
        assert messages[1].order == 6
        assert messages[1].content == json.dumps(history[1])

    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_history_empty_thread(patch_required_env, db_connection):
    test_settings = Settings(db={"prefix": db_connection})
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    thread_id = "empty_thread"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title_empty",
    )
    session.add(new_thread)
    await session.commit()
    await session.refresh(new_thread)

    try:
        thread = await get_thread(user_id=user_id, thread_id=thread_id, session=session)
        history = await thread.awaitable_attrs.messages

        assert history == []

    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_history_with_messages(patch_required_env, db_connection):
    test_settings = Settings(db={"prefix": db_connection})
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    thread_id = "valid_thread"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title_valid",
    )
    session.add(new_thread)
    await session.commit()

    messages_to_add = [
        Messages(
            order=1,
            thread_id=thread_id,
            entity=Entity.USER,
            content=json.dumps({"role": "user", "content": "User message"}),
        ),
        Messages(
            order=2,
            thread_id=thread_id,
            entity=Entity.AI_MESSAGE,
            content=json.dumps({"role": "assistant", "content": "AI message"}),
        ),
    ]
    session.add_all(messages_to_add)
    await session.commit()

    try:
        thread = await get_thread(user_id=user_id, thread_id=thread_id, session=session)
        history = await thread.awaitable_attrs.messages

        assert len(history) == 2
        assert json.loads(history[0].content) == {
            "role": "user",
            "content": "User message",
        }
        assert json.loads(history[1].content) == {
            "role": "assistant",
            "content": "AI message",
        }

    finally:
        await session.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_get_history_with_malformed_json(patch_required_env, db_connection):
    test_settings = Settings(db={"prefix": db_connection})
    engine = setup_engine(test_settings, db_connection)
    session = await anext(get_session(engine))
    user_id = "test_user"
    thread_id = "malformed_thread"

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    new_thread = Threads(
        user_id=user_id,
        thread_id=thread_id,
        vlab_id="test_vlab_DB",
        project_id="project_id_DB",
        title="test_title_malformed",
    )
    session.add(new_thread)
    await session.commit()

    messages_to_add = [
        Messages(
            order=1,
            thread_id=thread_id,
            entity=Entity.USER,
            content=json.dumps({"role": "user", "content": "Valid message"}),
        ),
        Messages(
            order=2,
            thread_id=thread_id,
            entity=Entity.AI_MESSAGE,
            content="MALFORMED_JSON",
        ),  # Malformed JSON
    ]
    session.add_all(messages_to_add)
    await session.commit()

    try:
        thread = await get_thread(user_id=user_id, thread_id=thread_id, session=session)
        with pytest.raises(json.JSONDecodeError):
            history = await thread.awaitable_attrs.messages
            json.loads(history[-1].content)

    finally:
        await session.close()
        await engine.dispose()
