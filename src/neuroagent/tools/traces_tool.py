"""Traces tool."""

import logging
from typing import Any, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from neuroagent.tools.base_tool import (
    BaseToolOutput,
    BasicTool,
)
from neuroagent.utils import get_descendants_id

logger = logging.getLogger(__name__)


class InputGetTraces(BaseModel):
    """Inputs of the knowledge graph API."""

    brain_region_id: str = Field(
        description="ID of the brain region of interest. Can be obtained from 'resolve-entities-tool'."
    )
    etype_id: str | None = Field(
        default=None,
        description=(
            "ID of the electrical type of the cell. Can be obtained through the 'resolve-entities-tool'."
        ),
    )


class TracesOutput(BaseToolOutput):
    """Output schema for the traces."""

    trace_id: str

    brain_region_id: str
    brain_region_label: str | None

    etype: str | None

    subject_species_id: str | None
    subject_species_label: str | None
    subject_age: str | None


class GetTracesTool(BasicTool):
    """Class defining the logic to obtain traces ids."""

    name: str = "get-traces-tool"
    description: str = """Searches a neuroscience based knowledge graph to retrieve traces names, IDs and descriptions.
    Requires a 'brain_region_id' which is the ID of the brain region of interest as registered in the knowledge graph.
    Optionally accepts an e-type id.
    The output is a list of traces, containing:
    - The trace id.
    - The brain region ID.
    - The brain region name.
    - The etype of the excited cell
    - The subject species ID.
    - The subject species name.
    - The subject age.
    The trace ID is in the form of an HTTP(S) link such as 'https://bbp.epfl.ch/neurosciencegraph/data/traces...'."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputGetTraces

    def _run(self, query: str) -> list[TracesOutput]:  # type: ignore
        pass

    async def _arun(
        self,
        brain_region_id: str,
        etype_id: str | None = None,
    ) -> list[TracesOutput] | dict[str, str]:
        """From a brain region ID, extract traces.

        Parameters
        ----------
        brain_region_id
            ID of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        etype
            Name of the etype of interest (in plain english)

        Returns
        -------
            list of TracesOutput to describe the trace and its metadata, or an error dict.
        """
        logger.info(f"Entering get trace tool. Inputs: {brain_region_id=}, {etype_id=}")
        try:
            # Get descendants of the brain region specified as input
            hierarchy_ids = get_descendants_id(
                brain_region_id, json_path=self.metadata["brainregion_path"]
            )
            logger.info(
                f"Found {len(list(hierarchy_ids))} children of the brain ontology."
            )

            # Create the ES query to query the KG with resolved descendants
            entire_query = self.create_query(
                brain_region_ids=hierarchy_ids, etype_id=etype_id
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
        brain_region_ids: set[str],
        etype_id: str | None = None,
    ) -> dict[str, Any]:
        """Create ES query.

        Parameters
        ----------
        brain_region_ids
            IDs of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        etype
            Name of the etype of interest (in plain english)

        Returns
        -------
            dict containing the ES query to send to the KG.
        """
        # At least one of the children brain region should match.
        conditions = [
            {
                "bool": {
                    "should": [
                        {"term": {"brainRegion.@id.keyword": hierarchy_id}}
                        for hierarchy_id in brain_region_ids
                    ]
                }
            }
        ]

        # Optionally constraint the output on the etype of the cell
        if etype_id is not None:
            logger.info(f"etype selected: {etype_id}")
            conditions.append({"term": {"eType.@id.keyword": etype_id}})  # type: ignore

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
                                "@type.keyword": "https://bbp.epfl.ch/ontologies/core/bmo/ExperimentalTrace"
                            }
                        },
                        {"term": {"curated": True}},
                        {"term": {"deprecated": False}},
                    ]
                }
            },
        }
        return entire_query

    @staticmethod
    def _process_output(output: Any) -> list[TracesOutput]:
        """Process output to fit the TracesOutput pydantic class defined above.

        Parameters
        ----------
        output
            Raw output of the _arun method, which comes from the KG

        Returns
        -------
            list of TracesOutput to describe the trace and its metadata.
        """
        results = [
            TracesOutput(
                trace_id=res["_source"]["@id"],
                brain_region_id=res["_source"]["brainRegion"]["@id"],
                brain_region_label=res["_source"]["brainRegion"]["label"],
                etype=(
                    res["_source"]["eType"].get("label")
                    if "eType" in res["_source"]
                    else None
                ),
                subject_species_id=(
                    res["_source"]["subjectSpecies"]["@id"]
                    if "subjectSpecies" in res["_source"]
                    else None
                ),
                subject_species_label=(
                    res["_source"]["subjectSpecies"]["label"]
                    if "subjectSpecies" in res["_source"]
                    else None
                ),
                subject_age=(
                    f"{res['_source']['subjectAge']['value']} {res['_source']['subjectAge']['unit']}"
                    if "subjectAge" in res["_source"]
                    else None
                ),
            )
            for res in output["hits"]["hits"]
        ]
        return results
