"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""

import logging
from typing import ClassVar

from pydantic import BaseModel, Field

from swarm_copy.bluenaas_models import SimulationDetailsResponse
from swarm_copy.tools.base_tool import BaseMetadata, BaseTool

logger = logging.getLogger(__name__)


class SCSGetOneMetadata(BaseMetadata):
    """Metadata class for the get all simulations api."""

    token: str
    vlab_id: str
    project_id: str
    bluenaas_url: str


class InputSCSGetOne(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    simulation_id: str = Field(description="ID of the simulation to retrieve.")


class SCSGetOneTool(BaseTool):
    """Class defining the SCSGetOne tool."""

    name: ClassVar[str] = "scsgetone-tool"
    description: ClassVar[
        str
    ] = """Get one specific simulations from a user based on its id.
    The id can be retrieved using the 'scsgetall-tool'."""
    metadata: SCSGetOneMetadata
    input_schema: InputSCSGetOne

    async def arun(self) -> SimulationDetailsResponse:
        """Run the SCSGetOne tool."""
        logger.info(
            f"Running SCSGetOne tool with inputs {self.input_schema.model_dump()}"
        )

        response = await self.metadata.httpx_client.get(
            url=f"{self.metadata.bluenaas_url}/simulation/single-neuron/{self.metadata.vlab_id}/{self.metadata.project_id}/{self.input_schema.simulation_id}",
            headers={"Authorization": f"Bearer {self.metadata.token}"},
        )

        return SimulationDetailsResponse(**response.json())
