"""Middleware."""

from typing import Any, Callable

from fastapi import Request, Response

from neuroagent.app.dependencies import get_settings


async def strip_path_prefix(
    request: Request, call_next: Callable[[Any], Any]
) -> Response:
    """Optionally strip a prefix from a request path.

    Parameters
    ----------
    request
        Request sent by the user.
    call_next
        Function executed to get the output of the endpoint.

    Returns
    -------
    response: Response of the request after potentially stripping prefix from path and applying other middlewares
    """
    if request.base_url in (
        "http://testserver/",
        "http://test/",
    ) and "healthz" not in str(request.url):
        settings = request.app.dependency_overrides[get_settings]()
    else:
        settings = get_settings()
    prefix = settings.misc.application_prefix
    if prefix is not None and len(prefix) > 0 and request.url.path.startswith(prefix):
        new_path = request.url.path[len(prefix) :]
        scope = request.scope
        scope["path"] = new_path
        request = Request(scope, request.receive)
    return await call_next(request)
