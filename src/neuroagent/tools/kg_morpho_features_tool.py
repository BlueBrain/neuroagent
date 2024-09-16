"""KG Morpho Feature tool."""

import logging
from typing import Any, Literal, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, model_validator

from neuroagent.tools.base_tool import BaseToolOutput, BasicTool
from neuroagent.utils import get_descendants_id

LABEL = Literal[
    "Neurite Max Radial Distance",
    "Number Of Sections",
    "Number Of Bifurcations",
    "Number Of Leaves",
    "Total Length",
    "Total Area",
    "Total Volume",
    "Section Lengths",
    "Section Term Lengths",
    "Section Bif Lengths",
    "Section Branch Orders",
    "Section Bif Branch Orders",
    "Section Term Branch Orders",
    "Section Path Distances",
    "Section Taper Rates",
    "Local Bifurcation Angles",
    "Remote Bifurcation Angles",
    "Partition Asymmetry",
    "Partition Asymmetry Length",
    "Sibling Ratios",
    "Diameter Power Relations",
    "Section Radial Distances",
    "Section Term Radial Distances",
    "Section Bif Radial Distances",
    "Terminal Path Lengths",
    "Section Volumes",
    "Section Areas",
    "Section Tortuosity",
    "Section Strahler Orders",
    "Soma Surface Area",
    "Soma Radius",
    "Soma Number Of Points",
    "Morphology Max Radial Distance",
    "Number Of Sections Per Neurite",
    "Total Length Per Neurite",
    "Total Area Per Neurite",
    "Total Height",
    "Total Width",
    "Total Depth",
    "Number Of Neurites",
]

STATISTICS = {
    "Morphology Max Radial Distance": ["raw"],
    "Neurite Max Radial Distance": ["raw"],
    "Number Of Sections": ["raw"],
    "Number Of Bifurcations": ["raw"],
    "Number Of Leaves": ["raw"],
    "Total Length": ["raw"],
    "Total Area": ["raw"],
    "Total Volume": ["raw"],
    "Section Lengths": ["min", "max", "median", "mean", "std"],
    "Section Term Lengths": ["min", "max", "median", "mean", "std"],
    "Section Bif Lengths": ["min", "max", "median", "mean", "std"],
    "Section Branch Orders": ["min", "max", "median", "mean", "std"],
    "Section Bif Branch Orders": ["min", "max", "median", "mean", "std"],
    "Section Term Branch Orders": ["min", "max", "median", "mean", "std"],
    "Section Path Distances": ["min", "max", "median", "mean", "std"],
    "Section Taper Rates": ["min", "max", "median", "mean", "std"],
    "Local Bifurcation Angles": ["min", "max", "median", "mean", "std"],
    "Remote Bifurcation Angles": ["min", "max", "median", "mean", "std"],
    "Partition Asymmetry": ["min", "max", "median", "mean", "std"],
    "Partition Asymmetry Length": ["min", "max", "median", "mean", "std"],
    "Sibling Ratios": ["min", "max", "median", "mean", "std"],
    "Diameter Power Relations": ["min", "max", "median", "mean", "std"],
    "Section Radial Distances": ["min", "max", "median", "mean", "std"],
    "Section Term Radial Distances": ["min", "max", "median", "mean", "std"],
    "Section Bif Radial Distances": ["min", "max", "median", "mean", "std"],
    "Terminal Path Lengths": ["min", "max", "median", "mean", "std"],
    "Section Volumes": ["min", "max", "median", "mean", "std"],
    "Section Areas": ["min", "max", "median", "mean", "std"],
    "Section Tortuosity": ["min", "max", "median", "mean", "std"],
    "Section Strahler Orders": ["min", "max", "median", "mean", "std"],
    "Soma Surface Area": ["raw"],
    "Soma Radius": ["raw"],
    "Number Of Sections Per Neurite": ["min", "max", "median", "mean", "std"],
    "Total Length Per Neurite": ["min", "max", "median", "mean", "std"],
    "Total Area Per Neurite": ["min", "max", "median", "mean", "std"],
    "Total Height": ["raw"],
    "Total Width": ["raw"],
    "Total Depth": ["raw"],
    "Number Of Neurites": ["raw"],
    "Soma Number Of Points": ["N"],
}

logger = logging.getLogger(__name__)


class FeatRangeInput(BaseModel):
    """Features Range input class."""

    min_value: float | int | None = None
    max_value: float | int | None = None


class FeatureInput(BaseModel):
    """Class defining the scheme of inputs the agent should use for the features."""

    label: LABEL
    compartment: (
        Literal["Axon", "BasalDendrite", "ApicalDendrite", "NeuronMorphology", "Soma"]
        | None
    ) = Field(
        default=None,
        description=(
            "Compartment of the cell. Leave as None if not explicitely stated by the"
            " user"
        ),
    )
    feat_range: FeatRangeInput | None = None

    @model_validator(mode="before")
    @classmethod
    def check_if_list(cls, data: Any) -> dict[str, str | list[float | int] | None]:
        """Validate that the values passed to the constructor are a dictionary."""
        if isinstance(data, list) and len(data) == 1:
            data_dict = data[0]
        else:
            data_dict = data
        return data_dict


class InputKGMorphoFeatures(BaseModel):
    """Inputs of the knowledge graph API when retrieving features of morphologies."""

    brain_region_id: str = Field(description="ID of the brain region of interest.")
    features: FeatureInput = Field(
        description="""Definition of the feature and values expected by the user.
        The input consists of a dictionary with three keys. The first one is the label (or name) of the feature specified by the user.
        The second one is the compartment in which the feature is calculated. It MUST be None if not explicitly specified by the user.
        The third one consists of a min_value and a max_value which encapsulate the range of values the user expects for this feature. It can also be None if not specified by the user.
        For instance, if the user asks for a morphology with an axon section volume between 1000 and 5000µm, the corresponding tuple should be: {label: 'Section Volumes', compartment: 'Axon', feat_range: FeatRangeInput(min_value=1000, max_value=5000)}.""",
    )


class KGMorphoFeatureOutput(BaseToolOutput):
    """Output schema for the knowledge graph API."""

    brain_region_id: str
    brain_region_label: str | None = None

    morphology_id: str
    morphology_name: str | None = None

    features: dict[str, str]


class KGMorphoFeatureTool(BasicTool):
    """Class defining the Knowledge Graph logic."""

    name: str = "kg-morpho-feature-tool"
    description: str = """Searches a neuroscience based knowledge graph to retrieve neuron morphology features based on a brain region of interest.
    Use this tool if and only if the user specifies explicitely certain features of morphology, and potentially the range of values expected.
    Requires a 'brain_region_id' and a dictionary with keys 'label' (and optionally 'compartment' and 'feat_range') describing the feature(s) specified by the user.
    The morphology ID is in the form of an HTTP(S) link such as 'https://bbp.epfl.ch/neurosciencegraph/data/neuronmorphologies...'.
    The output is a list of morphologies, containing:
    - The brain region ID.
    - The brain region name.
    - The morphology ID.
    - The morphology name.
    - The list of features of the morphology.
    If a given feature has multiple statistics (e.g. mean, min, max, median...), please return only its mean unless specified differently by the user."""
    metadata: dict[str, Any]
    args_schema: Type[BaseModel] = InputKGMorphoFeatures

    def _run(self) -> None:
        """Not defined yet."""
        pass

    async def _arun(
        self,
        brain_region_id: str,
        features: FeatureInput,
    ) -> list[KGMorphoFeatureOutput] | dict[str, str]:
        """Run the tool async.

        Parameters
        ----------
        brain_region_id
            ID of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        features
            Pydantic class describing the features one wants to compute

        Returns
        -------
            list of KGMorphoFeatureOutput to describe the morphology and its features, or an error dict.
        """
        try:
            logger.info(
                f"Entering KG morpho feature tool. Inputs: {brain_region_id=},"
                f" {features=}"
            )
            # Get the descendants of the brain region specified as input
            hierarchy_ids = get_descendants_id(
                brain_region_id, json_path=self.metadata["brainregion_path"]
            )
            logger.info(
                f"Found {len(list(hierarchy_ids))} children of the brain ontology."
            )

            # Get the associated ES query
            entire_query = self.create_query(
                brain_regions_ids=hierarchy_ids, features=features
            )

            # Send the ES query to the KG
            response = await self.metadata["httpx_client"].post(
                url=self.metadata["url"],
                headers={"Authorization": f"Bearer {self.metadata['token']}"},
                json=entire_query,
            )

            return self._process_output(response.json())

        except Exception as e:
            raise ToolException(str(e), self.name)

    def create_query(
        self, brain_regions_ids: set[str], features: FeatureInput
    ) -> dict[str, Any]:
        """Create ES query to query the KG with.

        Parameters
        ----------
        brain_regions_ids
            IDs of the brain region of interest (of the form http://api.brain-map.org/api/v2/data/Structure/...)
        features
            Pydantic class describing the features one wants to compute

        Returns
        -------
            Dict containing the ES query to send to the KG.
        """
        # At least one BR should match in the set of descendants
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

        # Add condition for the name of the requested feature to be present
        sub_conditions: list[dict[str, Any]] = []
        sub_conditions.append(
            {"term": {"featureSeries.label.keyword": str(features.label)}}
        )

        # Optionally add a constraint on the compartment if specified
        if features.compartment:
            sub_conditions.append(
                {
                    "term": {
                        "featureSeries.compartment.keyword": str(features.compartment)
                    }
                }
            )

        # Optionally add a constraint on the feature values if specified
        if features.feat_range:
            # Get the correct statistic for the feature
            stat = (
                "mean"
                if "mean" in STATISTICS[features.label]
                else "raw"
                if "raw" in STATISTICS[features.label]
                else "N"
            )

            # Add constraint on the statistic type
            sub_conditions.append({"term": {"featureSeries.statistic.keyword": stat}})
            feat_range = [
                features.feat_range.min_value,
                features.feat_range.max_value,
            ]
            # Add constraint on min and/or max value of the feature
            sub_condition = {"range": {"featureSeries.value": {}}}  # type: ignore
            if feat_range[0]:
                sub_condition["range"]["featureSeries.value"]["gte"] = feat_range[0]
            if feat_range[1]:
                sub_condition["range"]["featureSeries.value"]["lte"] = feat_range[1]
            if len(sub_condition["range"]["featureSeries.value"]) > 0:
                sub_conditions.append(sub_condition)

        # Nest the entire constrained query in a nested block
        feature_nested_query = {
            "nested": {
                "path": "featureSeries",
                "query": {"bool": {"must": sub_conditions}},
            }
        }
        conditions.append(feature_nested_query)  # type: ignore

        # Unwrap all of the conditions in the global query
        entire_query = {
            "size": self.metadata["search_size"],
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        *conditions,
                        {
                            "term": {
                                "@type.keyword": "https://bbp.epfl.ch/ontologies/core/bmo/NeuronMorphologyFeatureAnnotation"
                            }
                        },
                        {"term": {"deprecated": False}},
                    ]
                }
            },
        }

        return entire_query

    @staticmethod
    def _process_output(output: Any) -> list[KGMorphoFeatureOutput]:
        """Process output.

        Parameters
        ----------
        output
            Raw output of the _arun method, which comes from the KG

        Returns
        -------
            list of KGMorphoFeatureOutput to describe the morphology and its features.
        """
        formatted_output = []
        for morpho in output["hits"]["hits"]:
            morpho_source = morpho["_source"]
            feature_output = {
                f"{dic['compartment']} {dic['label']} ({dic['statistic']})": (
                    f"{dic['value']} ({dic['unit']})"
                )
                for dic in morpho_source["featureSeries"]
            }
            formatted_output.append(
                KGMorphoFeatureOutput(
                    brain_region_id=morpho_source["brainRegion"]["@id"],
                    brain_region_label=morpho_source["brainRegion"].get("label"),
                    morphology_id=morpho_source["neuronMorphology"]["@id"],
                    morphology_name=morpho_source["neuronMorphology"].get("name"),
                    features=feature_output,
                )
            )

        return formatted_output
