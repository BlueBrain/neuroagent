"""Test of the thread router."""

import pytest
from neuroagent.app.config import Settings
from neuroagent.app.dependencies import get_language_model, get_settings
from neuroagent.app.main import app
from neuroagent.app.routers.database.schemas import GetThreadsOutput
from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import Select


def test_create_thread(patch_required_env, app_client, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    with app_client as app_client:
        # Create a thread
        create_output = app_client.post("/threads/").json()
    assert create_output["thread_id"]
    assert create_output["user_sub"] == "dev"
    assert create_output["title"] == "title"
    assert create_output["timestamp"]


def test_get_threads(patch_required_env, app_client, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    with app_client as app_client:
        threads = app_client.get("/threads/").json()
        assert not threads
        # Create a thread
        create_output_1 = app_client.post("/threads/").json()
        create_output_2 = app_client.post("/threads/").json()
        threads = app_client.get("/threads/").json()

    assert len(threads) == 2
    assert threads[0] == create_output_1
    assert threads[1] == create_output_2


@pytest.mark.asyncio
async def test_get_thread(
    patch_required_env, fake_llm_with_tools, app_client, db_connection
):
    # Put data in the db
    llm, _, _ = await anext(fake_llm_with_tools)
    app.dependency_overrides[get_language_model] = lambda: llm
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    with app_client as app_client:
        wrong_response = app_client.get("/threads/test")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post("/threads/").json()
        thread_id = create_output["thread_id"]

        # Fill the thread
        app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
        )

        create_output = app_client.post("/threads/").json()
        empty_thread_id = create_output["thread_id"]
        empty_messages = app_client.get(f"/threads/{empty_thread_id}").json()
        assert empty_messages == []

        # Get the messages of the thread
        messages = app_client.get(f"/threads/{thread_id}").json()

    assert messages == [
        GetThreadsOutput(
            message_id=messages[0]["message_id"],
            entity="Human",
            message="This is my query",
        ).model_dump(),
        GetThreadsOutput(
            message_id="run-42768b30-044a-4263-8c5c-da61429aa9da-0",
            entity="AI",
            message="Great answer",
        ).model_dump(),
    ]


def test_update_threads(patch_required_env, app_client, db_connection):
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    with app_client as app_client:
        wrong_response = app_client.patch("/threads/test", json={"title": "New title"})
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post("/threads/").json()
        thread_id = create_output["thread_id"]
        app_client.patch(f"/threads/{thread_id}", json={"title": "New title"})
        threads = app_client.get("/threads/").json()

    assert threads[0]["title"] == "New title"


@pytest.mark.asyncio
async def test_delete_thread(
    patch_required_env, fake_llm_with_tools, app_client, db_connection
):
    # Put data in the db
    llm, _, _ = await anext(fake_llm_with_tools)
    app.dependency_overrides[get_language_model] = lambda: llm
    test_settings = Settings(
        db={"prefix": db_connection},
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    with app_client as app_client:
        wrong_response = app_client.delete("/threads/test")
        assert wrong_response.status_code == 404
        assert wrong_response.json() == {"detail": {"detail": "Thread not found."}}

        # Create a thread
        create_output = app_client.post("/threads/").json()
        thread_id = create_output["thread_id"]
        # Fill the thread
        app_client.post(
            f"/qa/chat/{thread_id}",
            json={"query": "This is my query"},
            params={"thread_id": thread_id},
        )
        # Get the messages of the thread
        messages = app_client.get(f"/threads/{thread_id}").json()
        threads = app_client.get("/threads").json()
        assert messages
        assert threads
        delete_response = app_client.delete(f"/threads/{thread_id}")
        assert delete_response.json() == {"Acknowledged": "true"}
        messages = app_client.get(f"/threads/{thread_id}").json()
        threads = app_client.get("/threads").json()
        assert messages == {"detail": {"detail": "Thread not found."}}
        assert not threads

    # Double check with pure sqlalchemy
    metadata = MetaData()
    engine = create_engine(test_settings.db.prefix)
    metadata.reflect(engine)

    with Session(engine) as session:
        for table in metadata.tables.values():
            if "thread_id" in table.c.keys():
                query = Select(table).where(  # type: ignore
                    table.c.thread_id == thread_id
                )
                row = session.execute(query).one_or_none()
                assert row is None
