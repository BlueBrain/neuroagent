"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""

import logging
from typing import Any, ClassVar
from urllib.parse import quote_plus

from pydantic import BaseModel, Field

from neuroagent.bluenaas_models import MEModelResponse
from neuroagent.tools.base_tool import BaseMetadata, BaseTool

logger = logging.getLogger(__name__)


class MEModelGetOneMetadata(BaseMetadata):
    """Metadata class for the get one me models api."""

    token: str
    vlab_id: str
    project_id: str
    bluenaas_url: str


class InputMEModelGetOne(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    memodel_id: str = Field(
        description="ID of the model to retrieve. Should be an https link."
    )


class MEModelGetOneTool(BaseTool):
    """Class defining the MEModelGetOne tool."""

    name: ClassVar[str] = "memodelgetone-tool"
    description: ClassVar[str] = """Get one specific me model from a user.
    The id can be retrieved using the 'memodelgetall-tool' or directly specified by the user."""
    metadata: MEModelGetOneMetadata
    input_schema: InputMEModelGetOne

    async def arun(self) -> dict[str, Any]:
        """Run the MEModelGetOne tool."""
        logger.info(
            f"Running MEModelGetOne tool with inputs {self.input_schema.model_dump()}"
        )

        response = await self.metadata.httpx_client.get(
            url=f"{self.metadata.bluenaas_url}/neuron-model/{self.metadata.vlab_id}/{self.metadata.project_id}/{quote_plus(self.input_schema.memodel_id)}",
            headers={"Authorization": f"Bearer {self.metadata.token}"},
        )

        return MEModelResponse(**response.json()).model_dump()
