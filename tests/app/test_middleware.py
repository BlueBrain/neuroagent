"""Test middleware"""

from unittest.mock import patch

import pytest
from fastapi.requests import Request
from fastapi.responses import Response

from neuroagent.app.config import Settings
from neuroagent.app.middleware import strip_path_prefix


@pytest.mark.parametrize(
    "path,prefix,trimmed_path",
    [
        ("/suggestions", "", "/suggestions"),
        ("/literature/suggestions", "/literature", "/suggestions"),
    ],
)
@pytest.mark.asyncio
async def test_strip_path_prefix(path, prefix, trimmed_path, patch_required_env):
    test_settings = Settings(misc={"application_prefix": prefix})

    scope = {
        "type": "http",
        "path": path,
        "query_string": b"best_query_string_i_have_ever_seen,_woah",
        "method": "POST",
        "headers": [
            (b"host", b"example.com"),
        ],
        "scheme": "http",
        "server": ("example.com", 80),
    }

    request = Request(scope=scope)

    async def async_callable(request):
        return Response(content=request.url.path, media_type="text/plain")

    with patch("neuroagent.app.middleware.get_settings", lambda: test_settings):
        response = await strip_path_prefix(request, async_callable)

    assert response.body.decode("utf-8") == trimmed_path
