"""Get the current time in the UTC timezone in the format of ISO 8601."""

import datetime
import logging
from typing import ClassVar

from pydantic import BaseModel
from swarm_copy.tools.base_tool import BaseMetadata, BaseTool


logger = logging.getLogger(__name__)


class NowMetadata(BaseMetadata):
    """Metadata for tool calling."""


class NowInput(BaseModel):
    """Input for tool calling."""


class NowTool(BaseTool):
    """Thought count tool for calling."""

    name: ClassVar[str] = "get-now"
    description: ClassVar[str] = (
        "This tool returns the current time in the UTC timezone in the format of ISO 8601."
    )
    hil: ClassVar[bool] = True
    metadata: NowMetadata
    input_schema: NowInput

    def run(self) -> str:
        """Run the tool."""
        raise NotImplementedError

    async def arun(self) -> str:
        """Run the tool asynchronously."""
        logger.info("Calling the now tool")

        return datetime.datetime.now(datetime.timezone.utc).isoformat()
