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

#TODO : since bluenaaas has multiple endpoints for synapse generation ,nexus query, simulation, how should we handle those?
# class SynapseSimulationConfig(BaseModel):
#     id: str
#     delay: int
#     duration: int
#     frequency: PositiveInt
#     weightScalar: int

class SimulationStimulusConfig(BaseModel):
    stimulusType: Literal["current_clamp", "voltage_clamp", "conductance"]
    stimulusProtocol: Literal["ap_waveform", "idrest", "iv", "fire_pattern"]
    amplitudes: Annotated[list[float], Len(min_length=1, max_length=15)]

class CurrentInjectionConfig(BaseModel):
    injectTo: str #TODO: could be soma, dendrite/basal, apical, AIS or how its named in the platform
    stimulus: SimulationStimulusConfig

class RecordingLocation(BaseModel):
    section: str #TODO: how to constrain this, we need to query available section ids from the obtained me model's morphology
    offset: float # TODO: should be between 0-1

class SimulationConditionsConfig(BaseModel):
    celsius: float # TODO: ideally this should be controlled as models dont perform well outside calibrated temperature range
    vinit: float # TODO: usually default value is set to -70 mV but can change depending on the simulation
    hypamp: float # this is usually set to experimentally defined value !
    max_time: float # user defined
    time_step: float # usually 0.025 ms
    seed: int # can be any nubmer

# class SimulationWithSynapseBody(BaseModel):
#     directCurrentConfig: CurrentInjectionConfig
#     synapseConfigs: list[SynapseSimulationConfig]


class InputBlueNaaS(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    me_model_id: str = Field(
        description=(
            "ID of the neuron model to be used in the simulation. The model ID can be"
            " fetched using the 'get-me-model-tool'."
        )
    )
    # synapses: list[SynapseSimulationConfig] = Field(
    #     description=(
    #         "List of synapse configurations. Each synapse configuration includes the"
    #         " synapse ID, delay, duration, frequency, and weight scalar."
    #     )
    # )
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


class BlueNaaSOutput(BaseModel):
    """Should return a successful POST request"""
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

    def get_default_values(self) -> dict:
        return {
            "me_model_id": "default_model_id",
            "currentInjection": {
                "injectTo": "soma",
                "stimulus": {
                    "stimulusType": "current_clamp",
                    "stimulusProtocol": "ap_waveform",
                    "amplitudes": [0.1]
                }
            },
            "recordFrom": [
                {"section": "soma", "offset": 0.5}
            ],
            "conditions": {
                "celsius": 36.0,
                "vinit": -70.0,
                "hypamp": 0.1,
                "max_time": 1000.0,
                "time_step": 0.025,
                "seed": 42
            },
            "simulationType": "single-neuron-simulation",
            "simulationDuration": 1000
        }

    async def prompt_user_for_approval(self, default_values: dict) -> bool:
        # This is a placeholder for the actual implementation
        # You might use a chat interface or some other method to get user approval
        user_response = await self.metadata["llm"].ainvoke({
            "messages": [
                {"role": "system", "content": "The following default values will be used for the simulation:"},
                {"role": "system", "content": str(default_values)},
                {"role": "user", "content": "Do you approve these values? (yes/no)"}
            ]
        })
        return user_response.lower() == "yes"

    async def _arun(self,
                    me_model_id: Optional[str] = None,
                    currentInjection: Optional[CurrentInjectionConfig] = None,
                    recordFrom: Optional[List[RecordingLocation]] = None,
                    conditions: Optional[SimulationConditionsConfig] = None,
                    simulationType: Optional[Literal["single-neuron-simulation"]] = None,
                    simulationDuration: Optional[int] = None
                    ) -> BaseToolOutput:
        """
        Run the BlueNaaS tool.

        Args:
            me_model_id: ID of the neuron model to be used in the simulation.
            currentInjection: Configuration for current injection.
            recordFrom: List of sections to record from during the simulation.
            conditions: Simulation conditions.
            simulationType: Type of the simulation.
            simulationDuration: Duration of the simulation in milliseconds.

        Returns:
            BaseToolOutput: Output of the BlueNaaS tool.
        """
        logger.info(f"Running BlueNaaS tool with inputs: {locals()}")

        # Get default values
        default_values = self.get_default_values()

        # Use provided values or default values
        me_model_id = me_model_id or default_values["me_model_id"]
        currentInjection = currentInjection or CurrentInjectionConfig(**default_values["currentInjection"])
        recordFrom = recordFrom or [RecordingLocation(**rec) for rec in default_values["recordFrom"]]
        conditions = conditions or SimulationConditionsConfig(**default_values["conditions"])
        simulationType = simulationType or default_values["simulationType"]
        simulationDuration = simulationDuration or default_values["simulationDuration"]

        # Prompt user for approval
        if not await self.prompt_user_for_approval({
            "me_model_id": me_model_id,
            "currentInjection": currentInjection.dict(),
            "recordFrom": [rec.dict() for rec in recordFrom],
            "conditions": conditions.dict(),
            "simulationType": simulationType,
            "simulationDuration": simulationDuration
        }):
            return BlueNaaSOutput(status="error", error="User did not approve the default values.")

        try:
            response = await self.metadata["httpx_client"].post(
                url=self.metadata["url"],
                headers={"Authorization": f"Bearer {self.metadata['token']}"},
                json={
                    "model_id": me_model_id,
                    "currentInjection": currentInjection.dict(),
                    "recordFrom": [rec.dict() for rec in recordFrom],
                    "conditions": conditions.dict(),
                    "type": simulationType,
                    "simulationDuration": simulationDuration
                },
            )
            response_data = response.json()
            return BlueNaaSOutput(status="success", result=response_data)
        except Exception as e:
            logger.error(f"Error running BlueNaaS tool: {e}")
            return BlueNaaSOutput(status="error", error=str(e))