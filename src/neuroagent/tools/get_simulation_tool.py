"""Tool to retrieve private traces (from simulations) in the knowledge graph."""

import logging
from typing import Any, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from neuroagent.tools.base_tool import BaseToolOutput, BasicTool

logger = logging.getLogger(__name__)


class InputGetSimulation(BaseModel):
    """Inputs of the tool."""

    me_model_id: str | None = Field(
        default=None,
        description="ID of the ME model used for the simulation. Usually obtained from the 'get-me-model-tool'.",
    )
    name: str | None = Field(
        default=None, description="Name of the simulation. Can be partial."
    )
    description: str | None = Field(
        default=None, description="Partial description of the simulation."
    )
    created_by: str | None = Field(
        default=None, description="Username of the person that created the simulation."
    )
    injection_location: str | None = Field(
        default=None,
        description="Location where the current was injected in the simulation. Typically composed of two parts: the name of the cell part and the relative location on it. The cell part is typically 'soma', 'dend' or 'apic' and the relative location is indicated by an integer between square bracket. For instance a valid injection location could be 'soma[0]', 'dend[12]' or 'apic[78]'.",
    )
    recording_locations: list[str] | None = Field(
        default=None,
        description="Locations where the recording happened in the simulation. Typically composed of three parts: the name of the cell part, the relative location on it and the input current. The cell part is typically 'soma', 'dend' or 'apic', the relative location is indicated by an integer between square bracket and the amplitude (in nA) is appended with an underscore. For instance a valid recording location could be ['soma[0]_0.5'], or ['dend[12]_0.2', 'apic[78]_0.07'].",
    )


class SimulationMetadataOutput(BaseToolOutput):
    """Output schema for simulation campaign."""

    simulation_id: str
    me_model_id: str | None  # me_model_id not provided for public me_model for now.

    created_by: str
    created_at: str

    name: str
    description: str
    injection_location: str
    recording_locations: list[str]


class GetSimulationTool(BasicTool):
    """Class defining the logic to obtain simulation ids."""

    name: str = "get-simulation-tool"
    description: str = """Retrieve electrical traces that have been simulated by a user.
    The tool should be used if the user requests a simulation result, or mentions his/her personal trace(s)/simulation result or a collegue's one.
    The target simulation can be described by:
    - A brain region id obtained through the 'resolve-brain-region' tool.
    - The explicit name of the trace. Can be a partial name.
    - A description of the simulation campaign that the user wrote when saving it. Can be partial.
    - The username of a person that created the simulation campaign.
    - The location where the current has been injected in the simulation.
    - The recording location in the simulation.
    Only specify the above if explicitely mentioned by the user.
    The tool returns similar informations as the ones described above."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputGetSimulation

    def _run(self, query: str) -> list[SimulationMetadataOutput]:  # type: ignore
        pass

    async def _arun(
        self,
        me_model_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        created_by: str | None = None,
        injection_location: str | None = None,
        recording_locations: list[str] | None = None,
    ) -> list[SimulationMetadataOutput] | dict[str, str]:
        """From partial information extract the relevant simulation.

        Parameters
        ----------
        brain_region_id
            ID of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        name
            Name of the simulation campaign.
        description
            (Partial) description of the simulation campaign.
        created_by
            Username of the person that created the simulation.
        injection_location
            Location where the current has been injected in the simulation.
        recording_locations
            Locations where the simulation recorded the response to the injected current.

        Returns
        -------
            list of SimulationMetadataOutput to describe the simulation, or an error dict.
        """
        logger.info(
            f"Entering get simulation tool. Inputs: {me_model_id=}, {name=}, {description=}, {created_by=}, {injection_location=}, {recording_locations=}"
        )
        try:
            # Create the ES query to query the KG with resolved descendants
            entire_query = self.create_query(
                me_model_id=me_model_id,
                name=name,
                description=description,
                created_by=created_by,
                injection_location=injection_location,
                recording_locations=recording_locations,
            )

            # Send the query to the KG
            response = await self.metadata["httpx_client"].post(
                url=self.metadata["url"],
                headers={"Authorization": f"Bearer {self.metadata['token']}"},
                json=entire_query,
            )
            return self._process_output(response.json())
        except Exception as e:
            raise ToolException(str(e), self.name)

    def create_query(
        self,
        me_model_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        created_by: str | None = None,
        injection_location: str | None = None,
        recording_locations: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create ES query.

        Parameters
        ----------
        brain_region_id
            ID of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        name
            (Partial) name of the simulation campaign.
        description
            (Partial) description of the simulation campaign.
        created_by
            Username of the person that created the simulation.
        injection_location
            Location where the current has been injected in the simulation.
        recording_locations
            Locations where the simulation recorded the response to the injected current.

        Returns
        -------
            dict containing the ES query to send to the KG.
        """
        # At least one of the children brain region should match.
        conditions = []

        # Add me_model id condition
        if me_model_id:
            # Warning: For now the field is called 'emodel' but should be 'memodel'.
            # Frontend might change that and we will have to adapt.
            conditions.append(
                {"term": {"singleNeuronSimulation.emodel.@id.keyword": me_model_id}}
            )

        # Add simulation name condition
        if name:
            # Can be a partial name so we match
            conditions.append({"match": {"name": name}})

        # Add (partial) description condition
        if description:
            conditions.append({"match": {"description": description}})

        # Add condition on user that created the simulation
        if created_by:
            conditions.append(
                {
                    "term": {
                        "createdBy.keyword": f"https://openbluebrain.com/api/nexus/v1/realms/SBO/users/{created_by.lower()}"
                    }
                }
            )

        # Add condition on injection location
        if injection_location:
            conditions.append({"term": {"injectionLocation": injection_location}})

        # Add condition on recording location. AND matching (they must at least all be there)
        if recording_locations:
            for recording_location in recording_locations:
                conditions.append(
                    {"term": {"recordingLocation.keyword": recording_location}}
                )

        # Unwrap everything into the main query
        entire_query = {
            "size": self.metadata["search_size"],
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        *conditions,
                        {
                            "term": {
                                "@type.keyword": "https://bbp.epfl.ch/ontologies/core/bmo/SingleNeuronSimulation"
                            }
                        },
                        {"term": {"deprecated": False}},
                    ]
                }
            },
        }
        return entire_query

    @staticmethod
    def _process_output(output: Any) -> list[SimulationMetadataOutput]:
        """Process output to fit the SimulationMetadataOutput pydantic class defined above.

        Parameters
        ----------
        output
            Raw output of the _arun method, which comes from the KG

        Returns
        -------
            list of SimulationMetadataOutput to describe the simulation and its metadata.
        """
        results = [
            SimulationMetadataOutput(
                simulation_id=res["_source"]["@id"],
                me_model_id=res["_source"]["singleNeuronSimulation"]["emodel"]["@id"]
                if "emodel" in res["_source"]["singleNeuronSimulation"]
                else None,
                created_by=res["_source"]["createdBy"],
                created_at=res["_source"]["createdAt"],
                name=res["_source"]["name"],
                description=res["_source"]["description"],
                injection_location=res["_source"]["injectionLocation"],
                recording_locations=res["_source"]["recordingLocation"]
                if isinstance(res["_source"]["recordingLocation"], list)
                else [res["_source"]["recordingLocation"]],
            )
            for res in output["hits"]["hits"]
        ]
        return results
