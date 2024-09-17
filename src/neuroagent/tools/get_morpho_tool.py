"""Get Morpho tool."""

import logging
from typing import Any, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from neuroagent.cell_types import get_celltypes_descendants
from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_descendants_id

logger = logging.getLogger(__name__)


class InputGetMorpho(BaseModel):
    """Inputs of the knowledge graph API."""

    brain_region_id: str = Field(description="ID of the brain region of interest.")
    mtype_id: Optional[str] = Field(
        default=None, description="ID of the M-type of interest."
    )


class KnowledgeGraphOutput(BaseToolOutput):
    """Output schema for the knowledge graph API."""

    morphology_id: str
    morphology_name: str | None
    morphology_description: str | None
    mtype: str | None

    brain_region_id: str
    brain_region_label: str | None

    subject_species_label: str | None
    subject_age: str | None


class GetMorphoTool(BasicTool):
    """Class defining the Get Morpho logic."""

    name: str = "get-morpho-tool"
    description: str = """Searches a neuroscience based knowledge graph to retrieve neuron morphology names, IDs and descriptions.
    Requires a 'brain_region_id' which is the ID of the brain region of interest as registered in the knowledge graph. To get this ID, please use the `resolve-brain-region-tool` first.
    The output is a list of morphologies, containing:
    - The brain region ID.
    - The brain region name.
    - The subject species name.
    - The subject age.
    - The morphology ID.
    - The morphology name.
    - the morphology description.
    The morphology ID is in the form of an HTTP(S) link such as 'https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies...'."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputGetMorpho

    def _run(self) -> None:
        pass

    async def _arun(
        self, brain_region_id: str, mtype_id: str | None = None
    ) -> list[KnowledgeGraphOutput]:
        """From a brain region ID, extract morphologies.

        Parameters
        ----------
        brain_region_id
            ID of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        mtype_id
            ID of the mtype of the morphology

        Returns
        -------
            list of KnowledgeGraphOutput to describe the morphology and its metadata, or an error dict.
        """
        logger.info(
            f"Entering Get Morpho tool. Inputs: {brain_region_id=}, {mtype_id=}"
        )
        try:
            # From the brain region ID, get the descendants.
            hierarchy_ids = get_descendants_id(
                brain_region_id, json_path=self.metadata["brainregion_path"]
            )
            logger.info(
                f"Found {len(list(hierarchy_ids))} children of the brain ontology."
            )

            # Create the ES query to query the KG.
            mtype_ids = (
                get_celltypes_descendants(mtype_id, self.metadata["celltypes_path"])
                if mtype_id
                else None
            )
            entire_query = self.create_query(
                brain_regions_ids=hierarchy_ids, mtype_ids=mtype_ids
            )

            # Send the query to get morphologies.
            response = await self.metadata["httpx_client"].post(
                url=self.metadata["url"],
                headers={"Authorization": f"Bearer {self.metadata['token']}"},
                json=entire_query,
            )

            # Process the output and return.
            return self._process_output(response.json())

        except Exception as e:
            raise ToolException(str(e), self.name)

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
            "size": self.metadata["search_size"],
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
    def _process_output(output: Any) -> list[KnowledgeGraphOutput]:
        """Process output to fit the KnowledgeGraphOutput pydantic class defined above.

        Parameters
        ----------
        output
            Raw output of the _arun method, which comes from the KG

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
                    res["_source"]["mType"].get("label")
                    if "mType" in res["_source"]
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
