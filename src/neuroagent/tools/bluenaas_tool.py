"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""
from typing import List, Literal, Optional, Annotated
from pydantic import BaseModel, Field, PositiveInt
from annotated_types import Len

import logging
from typing import Any, Type
from pydantic import BaseModel
from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_kg_data

logger = logging.getLogger(__name__)

class SynapseSimulationConfig(BaseModel):
    id: str
    delay: int
    duration: int
    frequency: PositiveInt
    weightScalar: int

class SimulationStimulusConfig(BaseModel):
    stimulusType: Literal["current_clamp", "voltage_clamp", "conductance"]
    stimulusProtocol: Literal["ap_waveform", "idrest", "iv", "fire_pattern"]
    amplitudes: Annotated[list[float], Len(min_length=1, max_length=15)]

class CurrentInjectionConfig(BaseModel):
    injectTo: str
    stimulus: SimulationStimulusConfig

class RecordingLocation(BaseModel):
    section: str
    offset: float

class SimulationConditionsConfig(BaseModel):
    celsius: float
    vinit: float
    hypamp: float
    max_time: float
    time_step: float
    seed: int

class SimulationWithSynapseBody(BaseModel):
    directCurrentConfig: CurrentInjectionConfig
    synapseConfigs: list[SynapseSimulationConfig]


class InputBlueNaaS(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    model_id: str = Field(
        description=(
            "ID of the neuron model to be used in the simulation. The model ID can be"
            " fetched using the 'get-me-model-tool'."
        )
    )
    synapses: list[SynapseSimulationConfig] = Field(
        description=(
            "List of synapse configurations. Each synapse configuration includes the"
            " synapse ID, delay, duration, frequency, and weight scalar."
        )
    )
    currentInjection: CurrentInjectionConfig = Field(
        description=(
            "Configuration for current injection. Includes the target section to inject"
            " to and the stimulus configuration."
        )
    )
    recordFrom: list[RecordingLocation] = Field(
        description=(
            "List of sections to record from during the simulation. Each record"
            " configuration includes the section name and offset."
        )
    )
    conditions: SimulationConditionsConfig = Field(
        description=(
            "Simulation conditions including temperature (celsius), initial voltage"
            " (vinit, in mV), hyperpolarizing current (hypamp, in nA), maximum simulation time"
            " (max_time, in ms), time step (time_step, in ms), and random seed (seed)."
        )
    )
    #TODO: implement synaptome simulation
    simulationType: Literal["single-neuron-simulation","synaptome-simulation"] = Field(
        description=(
            "Type of the simulation. it can be single neuron simulation for simulation"
            " without synapse placement or synaptome-simulation to put synapses on the morphology."
        )
    )
    simulationDuration: int = Field(
        description=(
            "Duration of the simulation in milliseconds."
        )
    )


class BlueNaaSOutput(BaseModel):
    status: str
    result: Optional[dict]
    error: Optional[str]


class BlueNaaSTool(BasicTool):
    name: str = "bluenaas-tool"
    description: str = """Runs a single-neuron simulation using the BlueNaaS service.
    Requires a 'model_id' which can be fetched using the 'get-me-model-tool'.
    The input configuration should be provided by the user otherwise agent 
    will probe the user with the selected default values."""
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