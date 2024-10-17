"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""

import json
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
        return f"A simulation will be ran with the following inputs <json>{self.inputs}</json>. \n Please confirm that you are satisfied by the simulation parameters, or correct them accordingly."


class BlueNaaSTool(BasicTool):
    """Class defining the BlueNaaS tool."""

    name: str = "bluenaas-tool"
    description: str = """Runs a single-neuron simulation using the BlueNaaS service.
    Requires a "me_model_id" which must be fetched by get-me-model-tool.
    Optionally, the user can specify simulation parameters.
    The tool will always ask for config validation from the user before running.
    If the user mentions an existing configuration, it must always be passed in the tool first to get user's approval.
    Specify ALL of the parameters everytime you enter this tool.
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

        if self.is_validated(messages, json_api):
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
            return BlueNaaSInvalidatedOutput(inputs=json_api), {"is_validated": True}

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

    @staticmethod
    def is_validated(messages: list[BaseMessage], json_api: dict[str, Any]) -> bool:
        """Decide whether the current configuration has been validated by the user.

        Parameters
        ----------
        messages
            List of Langgraph messages extracted from the graph state.
        json_api
            Simulation configuration that the tool will run if it has been validated.

        Returns
        -------
        is_validated
            Boolean stating wether or not the configuration has been validated by the user.
        """
        # If it is the first time the tool is called in the conversation, need validation
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
            return False

        # Verify if the tool has been recently called to ask for validation
        # There has to be at least 3 messages in the state otherwise cannot be validated
        if len(messages) > 3:
            last_messages = messages[-4:-1]
            recently_validated = (
                isinstance(last_messages[-1], HumanMessage)  # Approval from the human
                and isinstance(
                    last_messages[-2], AIMessage
                )  # AI answering the human and asking for validation
                and isinstance(
                    last_messages[-3], ToolMessage
                )  # First tool call not validated
                and last_messages[-3].name == "bluenaas-tool"
            )
            # If it hasn't been recently validated, we need more validation
            if not recently_validated:
                return False
        # If there is not enough messages in the state to have a potential validation
        else:
            return False

        # If the previous simulation was started, ask for validation on the new one
        if not last_bluenaas_call.artifact.get("is_validated"):
            return False

        # Verify if the config has changed since previous call. Validate again if so
        old_config = json.loads(
            last_bluenaas_call.content.split("<json>")[-1]  # type: ignore
            .split("</json>")[0]
            .replace("'", '"')
        )
        return old_config == json_api
