"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

import logging
from typing import Any, Type
from pydantic import BaseModel
from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_kg_data

logger = logging.getLogger(__name__)

class SynapseConfig(BaseModel):
    id: str
    delay: int
    duration: int
    frequency: int
    weightScalar: float

class StimulusConfig(BaseModel):
    stimulusType: Literal["current_clamp", "voltage_clamp"]
    stimulusProtocol: Literal["ap_waveform", "idrest", "iv", "fire_pattern", "pos_cheops", "neg_cheops"]
    amplitudes: List[float]

class CurrentInjectionConfig(BaseModel):
    injectTo: str
    stimulus: StimulusConfig

class RecordFromConfig(BaseModel):
    section: str
    offset: float

class ConditionsConfig(BaseModel):
    celsius: float
    vinit: float
    hypamp: float
    max_time: int
    time_step: float
    seed: int

class InputBlueNaaS(BaseModel):
    model_id: str
    synapses: List[SynapseConfig]
    currentInjection: CurrentInjectionConfig
    recordFrom: List[RecordFromConfig]
    conditions: ConditionsConfig
    type: Literal["single-neuron-simulation"]
    simulationDuration: int

class BlueNaaSOutput(BaseModel):
    status: str
    result: Optional[dict]
    error: Optional[str]


class BlueNaaSTool(BasicTool):
    name: str = "blue-naas-tool"
    description: str = """Runs a single-neuron simulation using the BlueNaaS service.
    Requires a 'model_id' which can be fetched using the 'get-me-model-tool'.
    The input configuration should be provided by the user otherwise agent will probe the user
    with the selected default values."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputBlueNaaS

    async def _arun(self, **kwargs) -> BlueNaaSOutput:
        """Run a single-neuron simulation using the BlueNaaS service."""
        logger.info(f"Running BlueNaaS tool with inputs: {kwargs}")
        try:
            response = await self.metadata["httpx_client"].post(
                url=self.metadata["url"],
                headers={"Authorization": f"Bearer {self.metadata['token']}"},
                json=kwargs,
            )
            response_data = response.json()
            return BlueNaaSOutput(status="success", result=response_data)
        except Exception as e:
            logger.error(f"Error running BlueNaaS tool: {e}")
            return BlueNaaSOutput(status="error", error=str(e))