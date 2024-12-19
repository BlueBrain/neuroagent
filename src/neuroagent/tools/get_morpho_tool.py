"""Get Morpho tool."""

import logging
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from neuroagent.cell_types import get_celltypes_descendants
from neuroagent.tools.base_tool import BaseMetadata, BaseTool
from neuroagent.utils import get_descendants_id

logger = logging.getLogger(__name__)


class GetMorphoInput(BaseModel):
    """Inputs of the knowledge graph API."""

    brain_region_id: str = Field(
        description="ID of the brain region of interest. To get this ID, please use the `resolve-entities-tool` first."
    )
    mtype_id: str | None = Field(
        default=None,
        description="ID of the M-type of interest. To get this ID, please use the `resolve-entities-tool` first.",
    )


class GetMorphoMetadata(BaseMetadata):
    """Metadata class for GetMorphoTool."""

    knowledge_graph_url: str
    token: str
    morpho_search_size: int
    brainregion_path: str | Path
    celltypes_path: str | Path


class KnowledgeGraphOutput(BaseModel):
    """Output schema for the knowledge graph API."""

    morphology_id: str
    morphology_name: str | None
    morphology_description: str | None
    mtype: list[str] | None

    brain_region_id: str
    brain_region_label: str | None

    subject_species_label: str | None
    subject_age: str | None


class GetMorphoTool(BaseTool):
    """Class defining the Get Morpho logic."""

    name: ClassVar[str] = "get-morpho-tool"
    description: ClassVar[
        str
    ] = """Searches a neuroscience based knowledge graph to retrieve neuron morphology names, IDs and descriptions.
    Requires a 'brain_region_id' which is the ID of the brain region of interest as registered in the knowledge graph.
    Optionally accepts an mtype_id.
    The output is a list of morphologies, containing:
    - The brain region ID.
    - The brain region name.
    - The subject species name.
    - The subject age.
    - The morphology ID.
    - The morphology name.
    - the morphology description.
    The morphology ID is in the form of an HTTP(S) link such as 'https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies...'."""
    input_schema: GetMorphoInput
    metadata: GetMorphoMetadata

    async def arun(self) -> list[dict[str, Any]]:
        """From a brain region ID, extract morphologies.

        Returns
        -------
            list of KnowledgeGraphOutput to describe the morphology and its metadata, or an error dict.
        """
        logger.info(
            f"Entering Get Morpho tool. Inputs: {self.input_schema.brain_region_id=}, {self.input_schema.mtype_id=}"
        )
        # From the brain region ID, get the descendants.
        hierarchy_ids = get_descendants_id(
            self.input_schema.brain_region_id,
            json_path=self.metadata.brainregion_path,
        )
        logger.info(f"Found {len(list(hierarchy_ids))} children of the brain ontology.")

        # Create the ES query to query the KG.
        mtype_ids = (
            get_celltypes_descendants(
                self.input_schema.mtype_id, self.metadata.celltypes_path
            )
            if self.input_schema.mtype_id
            else None
        )
        entire_query = self.create_query(
            brain_regions_ids=hierarchy_ids, mtype_ids=mtype_ids
        )

        # Send the query to get morphologies.
        response = await self.metadata.httpx_client.post(
            url=self.metadata.knowledge_graph_url,
            headers={"Authorization": f"Bearer {self.metadata.token}"},
            json=entire_query,
        )

        # Process the output and return.
        return self._process_output(response.json())

    def create_query(
        self, brain_regions_ids: set[str], mtype_ids: set[str] | None = None
    ) -> dict[str, Any]:
        """Create ES query out of the BR and mtype IDs.

        Parameters
        ----------
        brain_regions_ids
            IDs of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        mtype_ids
            IDs the the mtype of the morphology

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
            }
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

        # Assemble the query to return morphologies.
        entire_query = {
            "size": self.metadata.morpho_search_size,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        *conditions,
                        {
                            "term": {
                                "@type.keyword": "https://neuroshapes.org/ReconstructedNeuronMorphology"
                            }
                        },
                        {"term": {"deprecated": False}},
                        {"term": {"curated": True}},
                    ]
                }
            },
        }
        return entire_query

    @staticmethod
    def _process_output(output: Any) -> list[dict[str, Any]]:
        """Process output to fit the KnowledgeGraphOutput pydantic class defined above.

        Parameters
        ----------
        output
            Raw output of the arun method, which comes from the KG

        Returns
        -------
            list of KGMorphoFeatureOutput to describe the morphology and its metadata.
        """
        formatted_output = [
            KnowledgeGraphOutput(
                morphology_id=res["_source"]["@id"],
                morphology_name=res["_source"].get("name"),
                morphology_description=res["_source"].get("description"),
                mtype=(
                    [res["_source"]["mType"].get("label")]
                    if isinstance(res["_source"].get("mType"), dict)
                    else [item.get("label") for item in res["_source"]["mType"]]
                    if isinstance(res["_source"].get("mType"), list)
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
            ).model_dump()
            for res in output["hits"]["hits"]
        ]
        return formatted_output
