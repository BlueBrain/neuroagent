"""BlueNaaS single cell stimulation, simulation and synapse placement tool."""

import logging
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from neuroagent.tools.base_tool import BaseMetadata, BaseTool

logger = logging.getLogger(__name__)


class SCSPostMetadata(BaseMetadata):
    """Metadata class for the get all simulations api."""

    token: str
    vlab_id: str
    project_id: str
    bluenaas_url: str


class RecordingLocation(BaseModel):
    """Configuration for the recording location in the simulation."""

    section: str = Field(default="soma[0]", description="Section to record from")
    offset: float = Field(
        default=0.5, ge=0, le=1, description="Offset in the section to record from"
    )


class InputSCSPost(BaseModel):
    """Inputs for the BlueNaaS single-neuron simulation."""

    me_model_id: str = Field(
        description=(
            "ID of the neuron model to be used in the simulation. The model ID can be"
            " fetched using the 'memodelgetall-tool'."
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


class SCSPostOutput(BaseModel):
    """Should return a successful POST request."""

    id: str
    name: str
    status: Literal["success", "pending", "error"]
    error: str | None


class SCSPostTool(BaseTool):
    """Class defining the SCSPost tool."""

    name: ClassVar[str] = "scspost-tool"
    description: ClassVar[str] = """Runs a single-neuron simulation.
    Requires a "me_model_id" which must be fetched through the 'memodelgetall-tool' or directly provided by the user.
    Optionally, the user can specify simulation parameters.
    Returns the id of the simulation along with metadatas to fetch the simulation result and analyse it at a later stage.
    """
    metadata: SCSPostMetadata
    input_schema: InputSCSPost
    hil: ClassVar[bool] = True

    async def arun(self) -> dict[str, Any]:
        """Run the SCSPost tool."""
        logger.info(
            f"Running SCSPost tool with inputs {self.input_schema.model_dump()}"
        )

        json_api = self.create_json_api(
            current_injection__inject_to=self.input_schema.current_injection__inject_to,
            current_injection__stimulus__stimulus_type=self.input_schema.current_injection__stimulus__stimulus_type,
            current_injection__stimulus__stimulus_protocol=self.input_schema.current_injection__stimulus__stimulus_protocol,
            current_injection__stimulus__amplitudes=self.input_schema.current_injection__stimulus__amplitudes,
            record_from=self.input_schema.record_from,
            conditions__celsius=self.input_schema.conditions__celsius,
            conditions__vinit=self.input_schema.conditions__vinit,
            conditions__hypamp=self.input_schema.conditions__hypamp,
            conditions__max_time=self.input_schema.conditions__max_time,
            conditions__time_step=self.input_schema.conditions__time_step,
            conditions__seed=self.input_schema.conditions__seed,
        )

        response = await self.metadata.httpx_client.post(
            url=f"{self.metadata.bluenaas_url}/simulation/single-neuron/{self.metadata.vlab_id}/{self.metadata.project_id}/run",
            params={"model_id": self.input_schema.me_model_id, "realtime": "False"},
            headers={"Authorization": f"Bearer {self.metadata.token}"},
            json=json_api,
        )
        json_response = response.json()
        return SCSPostOutput(
            id=json_response["id"],
            status=json_response["status"],
            name=json_response["name"],
            error=json_response["error"],
        ).model_dump()

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
            "current_injection": {
                "inject_to": current_injection__inject_to,
                "stimulus": {
                    "stimulus_type": current_injection__stimulus__stimulus_type,
                    "stimulus_protocol": current_injection__stimulus__stimulus_protocol,
                    "amplitudes": current_injection__stimulus__amplitudes,
                },
            },
            "record_from": [recording.model_dump() for recording in record_from],
            "conditions": {
                "celsius": conditions__celsius,
                "vinit": conditions__vinit,
                "hypamp": conditions__hypamp,
                "max_time": conditions__max_time,
                "time_step": conditions__time_step,
                "seed": conditions__seed,
            },
            "type": "single-neuron-simulation",
            "duration": conditions__max_time,
        }
        return json_api
