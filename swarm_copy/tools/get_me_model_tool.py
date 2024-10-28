"""Module defining the Get ME Model tool."""

import logging
from typing import Any, ClassVar

from httpx import AsyncClient
from langchain_core.tools import ToolException
from pydantic import BaseModel, ConfigDict, Field

from neuroagent.cell_types import get_celltypes_descendants
from neuroagent.utils import get_descendants_id
from swarm_copy.tools.base_tool import BaseMetadata, BaseTool, BaseToolOutput

logger = logging.getLogger(__name__)


class GetMEModelInput(BaseModel):
    """Inputs of the knowledge graph API."""

    brain_region_id: str = Field(
        description="ID of the brain region of interest. To get this ID, please use the `resolve-entities-tool` first."
    )
    mtype_id: str | None = Field(
        default=None,
        description="ID of the M-type of interest. To get this ID, please use the `resolve-entities-tool` first.",
    )
    etype_id: str | None = Field(
        default=None,
        description="ID of the electrical type of the cell. Can be obtained through the 'resolve-entities-tool'.",
    )


class GetMEModelMetadata(BaseMetadata):
    """Metadata class for GetMEModelTool."""

    knowledge_graph_url: str
    token: str
    httpx_client: AsyncClient
    me_model_search_size: int
    brainregion_path: str
    celltypes_path: str
    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)


class MEModelOutput(BaseToolOutput):
    """Output schema for the knowledge graph API."""

    me_model_id: str
    me_model_name: str | None
    me_model_description: str | None
    mtype: str | None
    etype: str | None

    brain_region_id: str
    brain_region_label: str | None

    subject_species_label: str | None
    subject_age: str | None


class GetMEModelTool(BaseTool):
    """Class defining the Get ME Model logic."""

    name: ClassVar[str] = "get-me-model-tool"
    description: ClassVar[
        str
    ] = """Searches a neuroscience based knowledge graph to retrieve neuron morpho-electric model (ME models) names, IDs and descriptions.
    Requires a 'brain_region_id' which is the ID of the brain region of interest as registered in the knowledge graph.
    Optionally accepts an mtype_id and/or an etype_id.
    The output is a list of ME models, containing:
    - The brain region ID.
    - The brain region name.
    - The subject species name.
    - The subject age.
    - The model ID.
    - The model name.
    - The model description.
    The model ID is in the form of an HTTP(S) link such as 'https://bbp.epfl.ch/data/bbp/mmb-point-neuron-framework-model/...'."""
    input_schema: GetMEModelInput
    metadata: GetMEModelMetadata

    def _run(self) -> None:
        pass

    async def _arun(
        self,
    ) -> list[MEModelOutput]:
        """From a brain region ID, extract ME models."""
        logger.info(
            f"Entering Get ME Model tool. Inputs: {self.input_schema.brain_region_id=}, {self.input_schema.mtype_id=}, {self.input_schema.etype_id=}"
        )
        try:
            # From the brain region ID, get the descendants.
            hierarchy_ids = get_descendants_id(
                self.input_schema.brain_region_id,
                json_path=self.metadata.brainregion_path,
            )
            logger.info(
                f"Found {len(list(hierarchy_ids))} children of the brain ontology."
            )

            if self.input_schema.mtype_id:
                mtype_ids = get_celltypes_descendants(
                    self.input_schema.mtype_id, self.metadata.celltypes_path
                )
                logger.info(
                    f"Found {len(list(mtype_ids))} children of the cell types ontology for mtype."
                )
            else:
                mtype_ids = None

            # Create the ES query to query the KG.
            entire_query = self.create_query(
                brain_regions_ids=hierarchy_ids,
                mtype_ids=mtype_ids,
                etype_id=self.input_schema.etype_id,
            )

            # Send the query to get ME models.
            response = await self.metadata.httpx_client.post(
                url=self.metadata.knowledge_graph_url,
                headers={"Authorization": f"Bearer {self.metadata.token}"},
                json=entire_query,
            )

            # Process the output and return.
            return self._process_output(response.json())

        except Exception as e:
            raise ToolException(str(e), self.name)

    def create_query(
        self,
        brain_regions_ids: set[str],
        mtype_ids: set[str] | None = None,
        etype_id: str | None = None,
    ) -> dict[str, Any]:
        """Create ES query out of the BR, mtype, and etype IDs.

        Parameters
        ----------
        brain_regions_ids
            IDs of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        mtype_id
            ID of the mtype of the model
        etype_id
            ID of the etype of the model

        Returns
        -------
            dict containing the elasticsearch query to send to the KG.
        """
        # At least one of the children brain region should match.
        conditions = [
            {
                "bool": {
                    "should": [
                        {"term": {"brainRegion.@id.keyword": hierarchy_id}}
                        for hierarchy_id in brain_regions_ids
                    ]
                }
            },
            {"term": {"@type.keyword": "https://neuroshapes.org/MEModel"}},
            {"term": {"deprecated": False}},
        ]

        if mtype_ids:
            # The correct mtype should match. For now
            # It is a one term should condition, but eventually
            # we will resolve the subclasses of the mtypes.
            # They will all be appended here.
            conditions.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"mType.@id.keyword": mtype_id}}
                            for mtype_id in mtype_ids
                        ]
                    }
                }
            )

        if etype_id:
            # The correct etype should match.
            conditions.append({"term": {"eType.@id.keyword": etype_id}})

        # Assemble the query to return ME models.
        entire_query = {
            "size": self.metadata.me_model_search_size,
            "track_total_hits": True,
            "query": {"bool": {"must": conditions}},
        }
        return entire_query

    @staticmethod
    def _process_output(output: Any) -> list[MEModelOutput]:
        """Process output to fit the MEModelOutput pydantic class defined above.

        Parameters
        ----------
        output
            Raw output of the _arun method, which comes from the KG

        Returns
        -------
            list of MEModelOutput to describe the model and its metadata.
        """
        formatted_output = [
            MEModelOutput(
                me_model_id=res["_source"]["_self"],
                me_model_name=res["_source"].get("name"),
                me_model_description=res["_source"].get("description"),
                mtype=(
                    res["_source"]["mType"].get("label")
                    if "mType" in res["_source"]
                    else None
                ),
                etype=(
                    res["_source"]["eType"].get("label")
                    if "eType" in res["_source"]
                    else None
                ),
                brain_region_id=res["_source"]["brainRegion"]["@id"],
                brain_region_label=res["_source"]["brainRegion"].get("label"),
                subject_species_label=(
                    res["_source"]["subjectSpecies"].get("label")
                    if "subjectSpecies" in res["_source"]
                    else None
                ),
                subject_age=(
                    res["_source"]["subjectAge"].get("label")
                    if "subjectAge" in res["_source"]
                    else None
                ),
            )
            for res in output["hits"]["hits"]
        ]
        return formatted_output
