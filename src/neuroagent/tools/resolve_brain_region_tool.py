"""Tool to resolve the brain region from natural english to a KG ID."""

import logging
from typing import Any, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from neuroagent.resolving import resolve_query
from neuroagent.tools.base_tool import BaseToolOutput, BasicTool

logger = logging.getLogger(__name__)


class InputResolveBR(BaseModel):
    """Inputs of the Resolve Brain Region tool.."""

    brain_region: str = Field(
        description="Brain region of interest specified by the user in natural english."
    )
    mtype: Optional[str] = Field(
        default=None,
        description="M-type of interest specified by the user in natural english.",
    )


class BRResolveOutput(BaseToolOutput):
    """Output schema for the Brain region resolver."""

    brain_region_name: str
    brain_region_id: str


class MTypeResolveOutput(BaseToolOutput):
    """Output schema for the Mtype resolver."""

    mtype_name: str
    mtype_id: str


class ResolveBrainRegionTool(BasicTool):
    """Class defining the Brain Region Resolving logic."""

    name: str = "resolve-brain-region-tool"
    description: str = """From a brain region name written in natural english, search a knowledge graph to retrieve its corresponding ID.
    Optionaly resolve the mtype name from natural english to its corresponding ID too.
    You MUST use this tool when a brain region is specified in natural english because in that case the output of this tool is essential to other tools.
    returns a dictionary containing the brain region name, id and optionaly the mtype name and id.
    Brain region related outputs are stored in the class `BRResolveOutput` while the mtype related outputs are stored in the class `MTypeResolveOutput`."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputResolveBR

    def _run(self) -> None:
        """Not implemented yet."""
        pass

    async def _arun(
        self, brain_region: str, mtype: str | None = None
    ) -> list[BRResolveOutput | MTypeResolveOutput]:
        """Given a brain region in natural language, resolve its ID.

        Parameters
        ----------
        brain_region
            Name of the brain region to resolve (in english)
        mtype
            Name of the mtype to resolve (in english)

        Returns
        -------
            Mapping from BR/mtype name to ID.
        """
        logger.info(
            f"Entering Brain Region resolver tool. Inputs: {brain_region=}, {mtype=}"
        )
        try:
            # Prepare the output list.
            output: list[BRResolveOutput | MTypeResolveOutput] = []

            # First resolve the brain regions.
            brain_regions = await resolve_query(
                sparql_view_url=self.metadata["kg_sparql_url"],
                token=self.metadata["token"],
                query=brain_region,
                resource_type="nsg:BrainRegion",
                search_size=10,
                httpx_client=self.metadata["httpx_client"],
                es_view_url=self.metadata["kg_class_view_url"],
            )
            # Extend the resolved BRs.
            output.extend(
                [
                    BRResolveOutput(
                        brain_region_name=br["label"], brain_region_id=br["id"]
                    )
                    for br in brain_regions
                ]
            )

            # Optionally resolve the mtypes.
            if mtype is not None:
                mtypes = await resolve_query(
                    sparql_view_url=self.metadata["kg_sparql_url"],
                    token=self.metadata["token"],
                    query=mtype,
                    resource_type="bmo:BrainCellType",
                    search_size=10,
                    httpx_client=self.metadata["httpx_client"],
                    es_view_url=self.metadata["kg_class_view_url"],
                )
                # Extend the resolved mtypes.
                output.extend(
                    [
                        MTypeResolveOutput(
                            mtype_name=mtype["label"], mtype_id=mtype["id"]
                        )
                        for mtype in mtypes
                    ]
                )

            return output
        except Exception as e:
            raise ToolException(str(e), self.name)
