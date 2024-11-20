from unittest.mock import AsyncMock, Mock, patch

import pytest

from swarm_copy.app.database.sql_schemas import utc_now


@pytest.mark.asyncio
async def test_create_thread(app_client, settings):
    mock_validate_project = AsyncMock()
    mock_session = AsyncMock()

    def add_thread(thread):
        thread.creation_date = utc_now()
        thread.update_date = utc_now()
        thread.thread_id = "thread_id"

    mock_session.add = add_thread
    user_id = "user_id"
    token = "token"
    title = "title"
    project_id = "project_id"
    virtual_lab_id = "virtual_lab_id"
    with patch("swarm_copy.app.app_utils.validate_project", mock_validate_project):
        from swarm_copy.app.routers.threads import create_thread
        await create_thread(app_client, settings,
                            token,
                            virtual_lab_id,
                            project_id,
                            mock_session,
                            user_id,
                            title)

    assert mock_session.commit.called
    assert mock_session.refresh.called


@pytest.mark.asyncio
async def test_get_threads():
    user_id = "user_id"
    title = "title"
    thread_id = "uuid"
    project_id = "project_id"
    virtual_lab_id = "virtual_lab_id"
    creation_date = utc_now()
    update_date = utc_now()
    mock_threads = [
        Mock(user_id=user_id,
             title=title,
             vlab_id=virtual_lab_id,
             project_id=project_id,
             thread_id=thread_id,
             creation_date=creation_date,
             update_date=update_date)
    ]
    mock_session = AsyncMock()
    scalars_mock = Mock()
    scalars_mock.all.return_value = mock_threads
    mock_thread_result = Mock()
    mock_thread_result.scalars.return_value = scalars_mock
    mock_session.execute.return_value = mock_thread_result
    from swarm_copy.app.routers.threads import get_threads
    thread_reads = await get_threads(mock_session, user_id)
    thread_read = thread_reads[0]
    assert thread_read.thread_id == thread_id
    assert thread_read.user_id == user_id
    assert thread_read.vlab_id == virtual_lab_id
    assert thread_read.project_id == project_id
    assert thread_read.title == title
    assert thread_read.creation_date == creation_date
    assert thread_read.update_date == update_date


@pytest.mark.asyncio
async def test_update_thread_title():
    user_id = "user_id"
    title = "title"
    thread_id = "uuid"
    project_id = "project_id"
    virtual_lab_id = "virtual_lab_id"
    creation_date = utc_now()
    update_date = utc_now()
    mock_session = AsyncMock()
    mock_thread_result = Mock()
    mock_session.execute.return_value = mock_thread_result
    mock_update_thread = Mock()
    mock_update_thread.model_dump.return_value = {
        "user_id": user_id,
        "title": title,
        "vlab_id": virtual_lab_id,
        "project_id": project_id,
        "thread_id": thread_id,
        "creation_date": creation_date,
        "update_date": update_date
    }
    mock_thread = Mock()
    from swarm_copy.app.routers.threads import update_thread_title
    thread_read = await update_thread_title(mock_session, mock_update_thread, mock_thread)
    assert mock_session.commit.called
    assert mock_session.refresh.called
    assert thread_read.thread_id == thread_id
    assert thread_read.user_id == user_id
    assert thread_read.vlab_id == virtual_lab_id
    assert thread_read.project_id == project_id
    assert thread_read.title == title
    assert thread_read.creation_date == creation_date
    assert thread_read.update_date == update_date


@pytest.mark.asyncio
async def test_delete_thread():
    mock_session = AsyncMock()
    mock_thread_result = Mock()
    mock_session.execute.return_value = mock_thread_result
    mock_thread = Mock()
    from swarm_copy.app.routers.threads import delete_thread
    await delete_thread(mock_session, mock_thread)
    assert mock_session.delete.called
    assert mock_session.commit.called


@pytest.fixture(autouse=True)
def stop_patches():
    yield
    patch.stopall()
