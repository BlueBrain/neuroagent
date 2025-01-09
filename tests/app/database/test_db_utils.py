import pytest
from fastapi import HTTPException

from neuroagent.app.app_utils import setup_engine
from neuroagent.app.config import Settings
from neuroagent.app.database.db_utils import get_thread
from neuroagent.app.database.sql_schemas import Base, Threads
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
