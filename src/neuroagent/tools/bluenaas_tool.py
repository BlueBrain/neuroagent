"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""
from typing import Literal, Annotated
from pydantic import BaseModel, Field, PositiveInt

import logging
from pydantic import BaseModel
from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_kg_data

logger = logging.getLogger(__name__)

class SimulationStimulusConfig(BaseModel):
    stimulusType: Literal["current_clamp", "voltage_clamp", "conductance"] = Field(
        description="Type of stimulus to be used."
    )
    stimulusProtocol: Literal["ap_waveform", "idrest", "iv", "fire_pattern"] = Field(
        description="Stimulus protocol to be used."
    )

    amplitudes: list[float]  = Field(default_factory=list, min_items=1, description="List of amplitudes for the stimulus")

class CurrentInjectionConfig(BaseModel):
    injectTo: str = Field(
        description="Section to inject the current to."
    )
    stimulus: SimulationStimulusConfig

class RecordingLocation(BaseModel):
    section: str #TODO: how to constrain this, we need to query available section ids from the obtained me model's morphology
    offset: float = Field(ge=0, le=1, description="Offset in the section to record from")

class SimulationConditionsConfig(BaseModel):
    celsius: int = Field(default=34, ge=0, le=50, description="Temperature in celsius")
    vinit: int = Field(default=-73, description="Initial voltage in mV")
    hypamp: int = Field(default=0, description="Holding current in nA")
    max_time: int = Field(default=100, le=3000, description="Maximum simulation time in ms")
    time_step: float = Field(default=0.05, ge=0.001, le=10, description="Time step in ms")
    seed: int = Field(default=100, description="Random seed")



class InputBlueNaaS(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    me_model_id: str = Field(
        description=(
            "ID of the neuron model to be used in the simulation. The model ID can be"
            " fetched using the 'get-me-model-tool'."
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
    simulationType: Literal["single-neuron-simulation"] = Field(
        description=(
            "Type of the simulation. For now , its set to only single-neuron-simulation"
        )
    )
    simulationDuration: int = Field(
        description=(
            "Duration of the simulation in milliseconds."
        )
    )

