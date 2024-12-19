"""Tool to resolve the brain region from natural english to a KG ID."""

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from neuroagent.resolving import resolve_query
from neuroagent.tools.base_tool import (
    ETYPE_IDS,
    BaseMetadata,
    BaseTool,
    EtypesLiteral,
)

logger = logging.getLogger(__name__)


class ResolveBRInput(BaseModel):
    """Inputs of the Resolve Brain Region tool."""

    brain_region: str = Field(
        description="Brain region of interest specified by the user in natural english."
    )
    mtype: str | None = Field(
        default=None,
        description="M-type of interest specified by the user in natural english.",
    )
    etype: EtypesLiteral | None = Field(
        default=None,
        description=(
            "E-type of interest specified by the user in natural english. Possible values:"
            f" {', '.join(list(ETYPE_IDS.keys()))}. The first letter meaning c: continuous,"
            "b: bursting or d: delayed, The other letters in capital meaning AC: accomodating,"
            "NAC: non-accomodating, AD: adapting, NAD: non-adapting, STUT: stuttering,"
            "IR: irregular spiking. Optional suffixes in lowercase can exist:"
            "pyr: pyramidal, int: interneuron, _ltb: low threshold bursting,"
            "_noscltb: non-oscillatory low-threshold bursting. Examples: "
            "cADpyr: continuous adapting pyramidal. dAD_ltb: delayed adapting low-threshold bursting"
        ),
    )


class BRResolveOutput(BaseModel):
    """Output schema for the Brain region resolver."""

    brain_region_name: str
    brain_region_id: str


class MTypeResolveOutput(BaseModel):
    """Output schema for the Mtype resolver."""

    mtype_name: str
    mtype_id: str


class EtypeResolveOutput(BaseModel):
    """Output schema for the Mtype resolver."""

    etype_name: str
    etype_id: str


class ResolveBRMetadata(BaseMetadata):
    """Metadata for ResolveEntitiesTool."""

    token: str
    kg_sparql_url: str
    kg_class_view_url: str


class ResolveEntitiesTool(BaseTool):
    """Class defining the Brain Region Resolving logic."""

    name: ClassVar[str] = "resolve-entities-tool"
    description: ClassVar[
        str
    ] = """From a brain region name written in natural english, search a knowledge graph to retrieve its corresponding ID.
    Optionaly resolve the mtype name from natural english to its corresponding ID too.
    You MUST use this tool when a brain region is specified in natural english because in that case the output of this tool is essential to other tools.
    returns a dictionary containing the brain region name, id and optionaly the mtype name and id.
    Brain region related outputs are stored in the class `BRResolveOutput` while the mtype related outputs are stored in the class `MTypeResolveOutput`."""
    input_schema: ResolveBRInput
    metadata: ResolveBRMetadata

    async def arun(
        self,
    ) -> list[dict[str, Any]]:
        """Given a brain region in natural language, resolve its ID."""
        logger.info(
            f"Entering Brain Region resolver tool. Inputs: {self.input_schema.brain_region=}, "
            f"{self.input_schema.mtype=}, {self.input_schema.etype=}"
        )
        # Prepare the output list.
        output: list[dict[str, Any]] = []

        # First resolve the brain regions.
        brain_regions = await resolve_query(
            sparql_view_url=self.metadata.kg_sparql_url,
            token=self.metadata.token,
            query=self.input_schema.brain_region,
            resource_type="nsg:BrainRegion",
            search_size=10,
            httpx_client=self.metadata.httpx_client,
            es_view_url=self.metadata.kg_class_view_url,
        )
        # Extend the resolved BRs.
        output.extend(
            [
                BRResolveOutput(
                    brain_region_name=br["label"], brain_region_id=br["id"]
                ).model_dump()
                for br in brain_regions
            ]
        )

        # Optionally resolve the mtypes.
        if self.input_schema.mtype is not None:
            mtypes = await resolve_query(
                sparql_view_url=self.metadata.kg_sparql_url,
                token=self.metadata.token,
                query=self.input_schema.mtype,
                resource_type="bmo:BrainCellType",
                search_size=10,
                httpx_client=self.metadata.httpx_client,
                es_view_url=self.metadata.kg_class_view_url,
            )
            # Extend the resolved mtypes.
            output.extend(
                [
                    MTypeResolveOutput(
                        mtype_name=mtype["label"], mtype_id=mtype["id"]
                    ).model_dump()
                    for mtype in mtypes
                ]
            )

        # Optionally resolve the etype
        if self.input_schema.etype is not None:
            output.append(
                EtypeResolveOutput(
                    etype_name=self.input_schema.etype,
                    etype_id=ETYPE_IDS[self.input_schema.etype],
                ).model_dump()
            )

        return output
