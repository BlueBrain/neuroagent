"""Module defining the Get ME Model tool."""

import logging
from typing import Any, Literal, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from neuroagent.cell_types import get_celltypes_descendants
from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_descendants_id

logger = logging.getLogger(__name__)


class InputGetMEModel(BaseModel):
    """Inputs of the knowledge graph API."""

    brain_region_id: str = Field(description="ID of the brain region of interest.")
    mtype_id: Optional[str] = Field(
        default=None, description="ID of the M-type of interest."
    )
    etype_id: Optional[
        Literal[
            "bAC",
            "bIR",
            "bNAC",
            "bSTUT",
            "cAC",
            "cIR",
            "cNAC",
            "cSTUT",
            "dNAC",
            "dSTUT",
        ]
    ] = Field(default=None, description="ID of the E-type of interest.")


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


class GetMEModelTool(BasicTool):
    """Class defining the Get ME Model logic."""

    name: str = "get-me-model-tool"
    description: str = """Searches a neuroscience based knowledge graph to retrieve neuron morpho-electric model names, IDs and descriptions.
    Requires a 'brain_region_id' which is the ID of the brain region of interest as registered in the knowledge graph. To get this ID, please use the `resolve-brain-region-tool` first.
    Ideally, the user should also provide an 'mtype_id' and/or an 'etype_id' to filter the search results. But in case they are not provided, the search will return all models that match the brain region.
    The output is a list of ME models, containing:
    - The brain region ID.
    - The brain region name.
    - The subject species name.
    - The subject age.
    - The model ID.
    - The model name.
    - The model description.
    The model ID is in the form of an HTTP(S) link such as 'https://bbp.epfl.ch/data/bbp/mmb-point-neuron-framework-model/...'."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputGetMEModel

    def _run(self) -> None:
        pass

    async def _arun(
        self,
        brain_region_id: str,
        mtype_id: str | None = None,
        etype_id: str | None = None,
    ) -> list[MEModelOutput]:
        """From a brain region ID, extract ME models.

        Parameters
        ----------
        brain_region_id
            ID of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        mtype_id
            ID of the mtype of the model
        etype_id
            ID of the etype of the model

        Returns
        -------
            list of MEModelOutput to describe the model and its metadata, or an error dict.
        """
        logger.info(
            f"Entering Get ME Model tool. Inputs: {brain_region_id=}, {mtype_id=}, {etype_id=}"
        )
        try:
            # From the brain region ID, get the descendants.
            hierarchy_ids = get_descendants_id(
                brain_region_id, json_path=self.metadata["brainregion_path"]
            )
            logger.info(
                f"Found {len(list(hierarchy_ids))} children of the brain ontology."
            )

            if mtype_id:
                mtype_ids = set(
                    get_celltypes_descendants(mtype_id, self.metadata["celltypes_path"])
                )
                logger.info(
                    f"Found {len(list(mtype_ids))} children of the cell types ontology for mtype."
                )
            else:
                mtype_ids = None

            if etype_id:
                etype_ids = set(
                    get_celltypes_descendants(etype_id, self.metadata["celltypes_path"])
                )
                logger.info(
                    f"Found {len(list(etype_ids))} children of the cell types ontology for etype."
                )
            else:
                etype_ids = None

            # Create the ES query to query the KG.
            entire_query = self.create_query(
                brain_regions_ids=hierarchy_ids,
                mtype_ids=mtype_ids,
                etype_ids=etype_ids,
            )

            # Send the query to get ME models.
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
        self,
        brain_regions_ids: set[str],
        mtype_ids: set[str] | None = None,
        etype_ids: set[str] | None = None,
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
                            {"match": {"mType.label": mtype_id}}
                            for mtype_id in mtype_ids
                        ]
                    }
                }
            )

        if etype_ids:
            # The correct etype should match.
            conditions.append(
                {
                    "bool": {
                        "should": [
                            {"match": {"eType.label": etype_id}}
                            for etype_id in etype_ids
                        ]
                    }
                }
            )

        # Assemble the query to return ME models.
        entire_query = {
            "size": self.metadata["search_size"],
            "track_total_hits": True,
            "query": {"bool": {"must": conditions}},
            "sort": {"createdAt": {"order": "desc", "unmapped_type": "keyword"}},
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
