"""Morphology features tool."""

import logging
from typing import Any, Type

import neurom
import numpy as np
from langchain_core.tools import ToolException
from neurom.io.utils import load_morphology
from pydantic import BaseModel, Field

from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_kg_data

logger = logging.getLogger(__name__)


class InputNeuroM(BaseModel):
    """Inputs of the NeuroM API."""

    morphology_id: str = Field(
        description=(
            "ID of the morphology of interest. A morphology ID is an HTTP(S) link, it"
            " should therefore match the following regex pattern:"
            r" 'https?://\S+[a-zA-Z0-9]'"
        )
    )


class MorphologyFeatureOutput(BaseToolOutput):
    """Output schema for the neurom tool."""

    brain_region: str
    feature_dict: dict[str, Any]


class MorphologyFeatureTool(BasicTool):
    """Class defining the morphology feature retrieval logic."""

    name: str = "morpho-features-tool"
    description: str = """Given a morphology ID, fetch data about the features of the morphology. You need to know a morphology ID to use this tool and they can only come from the 'get-morpho-tool'. Therefore this tool should only be used if you already called the 'knowledge-graph-tool'.
    Here is an exhaustive list of features that can be retrieved with this tool:
    Soma radius, Soma surface area, Number of neurites, Number of sections, Number of sections per neurite, Section lengths, Segment lengths, Section radial distance, Section path distance, Local bifurcation angles, Remote bifurcation angles."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputNeuroM

    def _run(self) -> None:
        """Not implemented yet."""
        pass

    async def _arun(self, morphology_id: str) -> list[MorphologyFeatureOutput]:
        """Give features about morphology.

        Parameters
        ----------
        morphology_id
            ID of the morphology of interest

        Returns
        -------
            Dict containing feature_name: value.
        """
        logger.info(f"Entering morphology feature tool. Inputs: {morphology_id=}")
        try:
            # Download the .swc file describing the morphology from the KG
            morphology_content, metadata = await get_kg_data(
                object_id=morphology_id,
                httpx_client=self.metadata["httpx_client"],
                url=self.metadata["url"],
                token=self.metadata["token"],
                preferred_format="swc",
            )

            # Extract the features from it
            features = self.get_features(morphology_content, metadata.file_extension)
            return [
                MorphologyFeatureOutput(
                    brain_region=metadata.brain_region, feature_dict=features
                )
            ]
        except Exception as e:
            raise ToolException(str(e), self.name)

    def get_features(self, morphology_content: bytes, reader: str) -> dict[str, Any]:
        """Get features from a morphology.

        Parameters
        ----------
        morphology_content
            Bytes of the file containing the morphology info (comes from the KG)
        reader
            type of file (i.e. its extension)

        Returns
        -------
            Dict containing feature_name: value.
        """
        # Load the morphology
        morpho = load_morphology(morphology_content.decode(), reader=reader)

        # Compute soma radius and soma surface area
        features = {
            "soma_radius [µm]": neurom.get("soma_radius", morpho),
            "soma_surface_area [µm^2]": neurom.get("soma_surface_area", morpho),
        }

        # Prepare a list of features that have a unique value (no statistics)
        f1 = [
            ("number_of_neurites", "Number of neurites"),
            ("number_of_sections", "Number of sections"),
            ("number_of_sections_per_neurite", "Number of sections per neurite"),
        ]

        # For each neurite type, compute the above features
        for neurite_type in neurom.NEURITE_TYPES:
            for get_name, name in f1:
                features[f"{name} ({neurite_type.name})"] = neurom.get(
                    get_name, morpho, neurite_type=neurite_type
                )

        # Prepare a list of features that are defined by statistics
        f2 = [
            ("section_lengths", "Section lengths [µm]"),
            ("segment_lengths", "Segment lengths [µm]"),
            ("section_radial_distances", "Section radial distance [µm]"),
            ("section_path_distances", "Section path distance [µm]"),
            ("local_bifurcation_angles", "Local bifurcation angles [˚]"),
            ("remote_bifurcation_angles", "Remote bifurcation angles [˚]"),
        ]

        # For each neurite, compute the feature values and return their statistics
        for neurite_type in neurom.NEURITE_TYPES:
            for get_name, name in f2:
                try:
                    array = neurom.get(get_name, morpho, neurite_type=neurite_type)
                    if len(array) == 0:
                        continue
                    features[f"{name} ({neurite_type.name})"] = self.get_stats(array)
                except (IndexError, ValueError):
                    continue
        return features

    @staticmethod
    def get_stats(array: list[int | float]) -> dict[str, int | np.float64]:
        """Get summary stats for the array.

        Parameters
        ----------
        array
            Array of feature's statistics of a morphology

        Returns
        -------
            Dict with length, mean, sum, standard deviation, min and max of data
        """
        return {
            "len": len(array),
            "mean": np.mean(array),
            "sum": np.sum(array),
            "std": np.std(array),
            "min": np.min(array),
            "max": np.max(array),
        }
