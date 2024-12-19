"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from neuroagent.bluenaas_models import (
    PaginatedResponseUnionMEModelResponseSynaptomeModelResponse,
)
from neuroagent.tools.base_tool import BaseMetadata, BaseTool

logger = logging.getLogger(__name__)


class MEModelGetAllMetadata(BaseMetadata):
    """Metadata class for the get all me models api."""

    token: str
    vlab_id: str
    project_id: str
    bluenaas_url: str


class InputMEModelGetAll(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    offset: int = Field(default=0, description="Pagination offset")
    page_size: int = Field(
        default=20, description="Number of results returned by the API."
    )
    memodel_type: Literal["single-neuron-simulation", "synaptome-simulation"] = Field(
        default="single-neuron-simulation",
        description="Type of simulation to retrieve.",
    )


class MEModelGetAllTool(BaseTool):
    """Class defining the MEModelGetAll tool."""

    name: ClassVar[str] = "memodelgetall-tool"
    description: ClassVar[str] = """Get multiple me models from the user.
    Returns `page_size` ME-models that belong to the user's project.
    If the user requests an ME-model with specific criteria, use this tool
    to retrieve multiple of its ME-models and chose yourself the one(s) that fit the user's request."""
    metadata: MEModelGetAllMetadata
    input_schema: InputMEModelGetAll

    async def arun(self) -> dict[str, Any]:
        """Run the MEModelGetAll tool."""
        logger.info(
            f"Running MEModelGetAll tool with inputs {self.input_schema.model_dump()}"
        )

        response = await self.metadata.httpx_client.get(
            url=f"{self.metadata.bluenaas_url}/neuron-model/{self.metadata.vlab_id}/{self.metadata.project_id}/me-models",
            params={
                "simulation_type": self.input_schema.memodel_type,
                "offset": self.input_schema.offset,
                "page_size": self.input_schema.page_size,
            },
            headers={"Authorization": f"Bearer {self.metadata.token}"},
        )
        return PaginatedResponseUnionMEModelResponseSynaptomeModelResponse(
            **response.json()
        ).model_dump()
