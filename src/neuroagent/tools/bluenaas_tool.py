"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""

from typing import Literal, Annotated
from pydantic import BaseModel, Field, PositiveInt

import logging
from pydantic import BaseModel
from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_kg_data

logger = logging.getLogger(__name__)


class SimulationStimulusConfig(BaseModel):
    stimulus_type: Literal["current_clamp", "voltage_clamp", "conductance"] = Field(
        default="current_clamp", description="Type of stimulus to be used."
    )
    stimulus_protocol: Literal["ap_waveform", "idrest", "iv", "fire_pattern"] = Field(
        default="ap_waveform", description="Stimulus protocol to be used."
    )

    amplitudes: list[float] = Field(
        default_factory=lambda: [0.1],
        min_items=1,
        description="List of amplitudes for the stimulus",
    )


class CurrentInjectionConfig(BaseModel):
    inject_to: str = Field(
        default="soma[0]", description="Section to inject the current to."
    )
    stimulus: SimulationStimulusConfig = SimulationStimulusConfig()


class RecordingLocation(BaseModel):
    section: str = Field(default="soma[0]", description="Section to record from")
    offset: float = Field(
        default=0.5, ge=0, le=1, description="Offset in the section to record from"
    )


class SimulationConditionsConfig(BaseModel):
    celsius: int = Field(default=34, ge=0, le=50, description="Temperature in celsius")
    vinit: int = Field(default=-73, description="Initial voltage in mV")
    hypamp: int = Field(default=0, description="Holding current in nA")
    max_time: int = Field(
        default=100, le=3000, description="Maximum simulation time in ms"
    )
    time_step: float = Field(
        default=0.05, ge=0.001, le=10, description="Time step in ms"
    )
    seed: int = Field(default=100, description="Random seed")


class InputBlueNaaS(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    me_model_id: str = Field(
        description=(
            "ID of the neuron model to be used in the simulation. The model ID can be"
            " fetched using the 'get-me-model-tool'."
        )
    )

    current_injection: CurrentInjectionConfig = Field(
        default=CurrentInjectionConfig(),
        description=(
            "Configuration for current injection. Includes the target section to inject"
            " to and the stimulus configuration."
        ),
    )
    record_from: list[RecordingLocation] = Field(
        default_factory=lambda: [RecordingLocation()],
        description=(
            "List of sections to record from during the simulation. Each record"
            " configuration includes the section name and offset."
        ),
    )
    conditions: SimulationConditionsConfig = Field(
        default=SimulationConditionsConfig(),
        description=(
            "Simulation conditions including temperature (celsius), initial voltage"
            " (vinit, in mV), hyperpolarizing current (hypamp, in nA), maximum simulation time"
            " (max_time, in ms), time step (time_step, in ms), and random seed (seed)."
        ),
    )


class BlueNaaSOutput(BaseToolOutput):
    """Should return a successful POST request"""

    status: Literal["success", "pending", "error"]


class BlueNaaSTool(BasicTool):
    name: str = "bluenaas-tool"
    description: str = """Runs a single-neuron simulation using the BlueNaaS service.
    Requires a "me_model_id" which must be fetched by get-me-model-tool.

    The tool 
    """
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputBlueNaaS

    def _run(self) -> None:
        pass

    async def _arun(
        self,
        me_model_id: str,
        current_injection: CurrentInjectionConfig,
        record_from: list[RecordingLocation],
        conditions: SimulationConditionsConfig,
    ) -> BaseToolOutput:
        """Run the BlueNaaS tool."""
        logger.info(f"Running BlueNaaS tool")

        try:
            response = await self.metadata["httpx_client"].post(
                url=self.metadata["url"],
                params={"model_id": me_model_id},
                headers={"Authorization": f"Bearer {self.metadata["token"]}"},
                json={
                    "currentInjection": current_injection.model_dump(),
                    "recordFrom": [rec.model_dump() for rec in recordFrom],
                    "conditions": conditions.model_dump(),
                    "simulationType": "single-neuron-simulation",
                    "simulationDuration": conditions.max_time,
                },
                timeout=5.0,
            )

            return BlueNaaSOutput(status="success")

        except Exception as e:
            raise ToolException(str(e), self.name)
