import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException

from swarm_copy.app.database.db_utils import get_thread, save_history, get_history
from swarm_copy.app.database.sql_schemas import Entity, Messages, Threads


@pytest.mark.asyncio
async def test_get_thread():
    user_id = "0"
    thread_id = "0"
    mock_thread_result = Mock()
    mock_scalars_return = Mock()
    mock_thread = Mock()
    mock_scalars_return.one_or_none.return_value = mock_thread
    mock_thread_result.scalars.return_value = mock_scalars_return
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_thread_result
    result = await get_thread(user_id=user_id, thread_id=thread_id, session=mock_session)
    assert result == mock_thread


@pytest.mark.asyncio
async def test_get_thread_exception():
    user_id = "0"
    thread_id = "0"
    mock_thread_result = Mock()
    mock_scalars_return = Mock()
    mock_scalars_return.one_or_none.return_value = None
    mock_thread_result.scalars.return_value = mock_scalars_return
    mock_session = AsyncMock()
    mock_session.execute.return_value = mock_thread_result
    with pytest.raises(HTTPException):
        await get_thread(user_id=user_id, thread_id=thread_id, session=mock_session)


@pytest.mark.parametrize("message_role,expected_entity,content", [
    ('user', Entity.USER, False),
    ('tool', Entity.TOOL, False),
    ('assistant', Entity.AI_MESSAGE, True),
    ('assistant', Entity.AI_TOOL, False)
])
@pytest.mark.asyncio
async def test_save_history(message_role, expected_entity, content):
    history = [{"role": message_role, "content": content}]
    user_id, thread_id, offset = "test_user", "test_thread", 0

    mock_session = AsyncMock()
    mock_thread = AsyncMock()

    async def mock_get_thread(**kwargs):
        return mock_thread

    with patch("swarm_copy.app.database.db_utils.get_thread", mock_get_thread):
        await save_history(history, user_id, thread_id, offset, mock_session)

    assert mock_session.add.called

    called_with_param = mock_session.add.call_args[0][0]
    assert isinstance(called_with_param, Messages)
    assert called_with_param.order == 0
    assert called_with_param.thread_id == thread_id
    assert called_with_param.entity == expected_entity
    assert called_with_param.content == json.dumps(history[0])

    assert mock_session.commit.called


@pytest.mark.asyncio
async def test_save_history_exception():
    history = [{"role": "bad role", "content": None}]
    user_id, thread_id, offset = "test_user", "test_thread", 0

    mock_session = AsyncMock()

    with pytest.raises(HTTPException):
        await save_history(history, user_id, thread_id, offset, mock_session)


@pytest.mark.asyncio
async def test_get_history():
    msg1 = Mock()
    msg1.content = json.dumps("message1")
    msg2 = Mock()
    msg2.content = json.dumps("message2")
    mock_thread = AsyncMock()
    messages = [msg1, msg2]

    async def mock_messages():
        return messages

    mock_thread.awaitable_attrs.messages = mock_messages()
    results = await get_history(mock_thread)
    assert results == [json.loads(msg.content) for msg in messages]
