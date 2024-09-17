"""SQL related functions."""

from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic
from sqlalchemy.orm import Session

from neuroagent.app.dependencies import get_session, get_user_id
from neuroagent.app.routers.database.schemas import Threads

security = HTTPBasic()


def get_object(
    session: Annotated[Session, Depends(get_session)],
    thread_id: str,
    user_id: Annotated[str, Depends(get_user_id)],
) -> Threads:
    """Get an SQL object. Also useful to correlate user_id to thread_id.

    Parameters
    ----------
    session
        Session object connected to the SQL instance.
    thread_id
        ID of the thread, provided by the user.
    user_id
        ID of the user.

    Returns
    -------
    object
        Relevant row of the relevant table in the SQL DB.
    """
    sql_object = session.get(Threads, (thread_id, user_id))
    if not sql_object:
        raise HTTPException(
            status_code=404,
            detail={
                "detail": "Thread not found.",
            },
        )
    return sql_object
