"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""

import logging
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from neuroagent.tools.base_tool import BaseToolOutput, BasicTool

logger = logging.getLogger(__name__)


class RecordingLocation(BaseModel):
    """Configuration for the recording location in the simulation."""

    section: str = Field(default="soma[0]", description="Section to record from")
    offset: float = Field(
        default=0.5, ge=0, le=1, description="Offset in the section to record from"
    )


class InputBlueNaaS(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    me_model_id: str = Field(
        description=(
            "ID of the neuron model to be used in the simulation. The model ID can be"
            " fetched using the 'get-me-model-tool'."
        )
    )
    current_injection__inject_to: str = Field(
        default="soma[0]", description="Section to inject the current to."
    )
    current_injection__stimulus__stimulus_type: Literal[
        "current_clamp", "voltage_clamp", "conductance"
    ] = Field(default="current_clamp", description="Type of stimulus to be used.")
    current_injection__stimulus__stimulus_protocol: Literal[
        "ap_waveform", "idrest", "iv", "fire_pattern"
    ] = Field(default="ap_waveform", description="Stimulus protocol to be used.")

    current_injection__stimulus__amplitudes: list[float] = Field(
        default=[0.1],
        min_length=1,
        description="List of amplitudes for the stimulus",
    )
    record_from: list[RecordingLocation] = Field(
        default=[RecordingLocation()],
        description=(
            "List of sections to record from during the simulation. Each record"
            " configuration includes the section name and offset."
        ),
    )
    conditions__celsius: int = Field(
        default=34, ge=0, le=50, description="Temperature in celsius"
    )
    conditions__vinit: int = Field(default=-73, description="Initial voltage in mV")
    conditions__hypamp: int = Field(default=0, description="Holding current in nA")
    conditions__max_time: int = Field(
        default=100, le=3000, description="Maximum simulation time in ms"
    )
    conditions__time_step: float = Field(
        default=0.05, ge=0.001, le=10, description="Time step in ms"
    )
    conditions__seed: int = Field(default=100, description="Random seed")
    messages: Annotated[list[BaseMessage], InjectedState("messages")]


class BlueNaaSValidatedOutput(BaseToolOutput):
    """Should return a successful POST request."""

    status: Literal["success", "pending", "error"]


class BlueNaaSInvalidatedOutput(BaseModel):
    """Response to the user if the simulation has not been validated yet."""

    inputs: dict[str, Any]

    def __str__(self) -> str:
        """Format the response passed to the LLM."""
        return f"A simulation will be ran with the following inputs {self.inputs}. \n Please confirm that you are satisfied by the simulation parameters, or correct them accordingly."


class BlueNaaSTool(BasicTool):
    """Class defining the BlueNaaS tool."""

    name: str = "bluenaas-tool"
    description: str = """Runs a single-neuron simulation using the BlueNaaS service.
    Requires a "me_model_id" which must be fetched by get-me-model-tool.
    Optionally, the user can specify simulation parameters.
    """
    metadata: dict[str, Any]
    args_schema: type[BaseModel] = InputBlueNaaS
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def _run(self) -> None:
        pass

    async def _arun(
        self,
        me_model_id: str,
        messages: Annotated[list[BaseMessage], InjectedState("messages")],
        current_injection__inject_to: str = "soma[0]",
        current_injection__stimulus__stimulus_type: Literal[
            "current_clamp", "voltage_clamp", "conductance"
        ] = "current_clamp",
        current_injection__stimulus__stimulus_protocol: Literal[
            "ap_waveform", "idrest", "iv", "fire_pattern"
        ] = "ap_waveform",
        current_injection__stimulus__amplitudes: list[float] | None = None,
        record_from: list[RecordingLocation] | None = None,
        conditions__celsius: int = 34,
        conditions__vinit: int = -73,
        conditions__hypamp: int = 0,
        conditions__max_time: int = 100,
        conditions__time_step: float = 0.05,
        conditions__seed: int = 100,
    ) -> tuple[BaseToolOutput | BaseModel, dict[str, bool]]:
        """Run the BlueNaaS tool."""
        logger.info("Running BlueNaaS tool")
        try:
            # Get the last bluenaas call
            last_bluenaas_call = next(
                (
                    message
                    for message in reversed(messages)
                    if isinstance(message, ToolMessage)
                    and message.name == "bluenaas-tool"
                )
            )
        except StopIteration:
            last_bluenaas_call = None

        last_messages = messages[-4:-1] if len(messages) > 3 else None
        if last_messages is not None:
            recently_validated = (
                isinstance(last_messages[-1], HumanMessage)  # Approval from the human
                and isinstance(
                    last_messages[-2], AIMessage
                )  # AI sending the second validated tool call
                and isinstance(
                    last_messages[-3], ToolMessage
                )  # First tool call not validated
                and last_messages[-3].name == "bluenaas-tool"
            )
        else:
            recently_validated = False

        json_api = self.create_json_api(
            current_injection__inject_to=current_injection__inject_to,
            current_injection__stimulus__stimulus_type=current_injection__stimulus__stimulus_type,
            current_injection__stimulus__stimulus_protocol=current_injection__stimulus__stimulus_protocol,
            current_injection__stimulus__amplitudes=current_injection__stimulus__amplitudes,
            record_from=record_from,
            conditions__celsius=conditions__celsius,
            conditions__vinit=conditions__vinit,
            conditions__hypamp=conditions__hypamp,
            conditions__max_time=conditions__max_time,
            conditions__time_step=conditions__time_step,
            conditions__seed=conditions__seed,
        )
        # The tool is called for the first time -> need validation
        if last_bluenaas_call is None:
            # We send the config for validation. We assume validated from now on
            return BlueNaaSInvalidatedOutput(inputs=json_api), {"is_validated": True}

        else:
            # The tool is not called for the first time -> check if already validated
            if last_bluenaas_call.artifact.get("is_validated") and recently_validated:
                try:
                    await self.metadata["httpx_client"].post(
                        url=self.metadata["url"],
                        params={"model_id": me_model_id},
                        headers={"Authorization": f'Bearer {self.metadata["token"]}'},
                        json=json_api,
                        timeout=5.0,
                    )

                    return BlueNaaSValidatedOutput(status="success"), {
                        "is_validated": False
                    }

                except Exception as e:
                    raise ToolException(str(e), self.name)
            else:
                return BlueNaaSInvalidatedOutput(inputs=json_api), {
                    "is_validated": True
                }

    @staticmethod
    def create_json_api(
        current_injection__inject_to: str = "soma[0]",
        current_injection__stimulus__stimulus_type: Literal[
            "current_clamp", "voltage_clamp", "conductance"
        ] = "current_clamp",
        current_injection__stimulus__stimulus_protocol: Literal[
            "ap_waveform", "idrest", "iv", "fire_pattern"
        ] = "ap_waveform",
        current_injection__stimulus__amplitudes: list[float] | None = None,
        record_from: list[RecordingLocation] | None = None,
        conditions__celsius: int = 34,
        conditions__vinit: int = -73,
        conditions__hypamp: int = 0,
        conditions__max_time: int = 100,
        conditions__time_step: float = 0.05,
        conditions__seed: int = 100,
    ) -> dict[str, Any]:
        """Based on the simulation config, create a valid JSON for the API."""
        if not current_injection__stimulus__amplitudes:
            current_injection__stimulus__amplitudes = [0.1]
        if not record_from:
            record_from = [RecordingLocation()]
        json_api = {
            "currentInjection": {
                "injectTo": current_injection__inject_to,
                "stimulus": {
                    "stimulusType": current_injection__stimulus__stimulus_type,
                    "stimulusProtocol": current_injection__stimulus__stimulus_protocol,
                    "amplitudes": current_injection__stimulus__amplitudes,
                },
            },
            "recordFrom": [recording.model_dump() for recording in record_from],
            "conditions": {
                "celsius": conditions__celsius,
                "vinit": conditions__vinit,
                "hypamp": conditions__hypamp,
                "max_time": conditions__max_time,
                "time_step": conditions__time_step,
                "seed": conditions__seed,
            },
            "type": "single-neuron-simulation",
            "simulationDuration": conditions__max_time,
        }
        return json_api
