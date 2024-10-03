"""Tests KG Morpho Features tool."""

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.tools import ToolException

from neuroagent.tools import KGMorphoFeatureTool
from neuroagent.tools.kg_morpho_features_tool import (
    FeatRangeInput,
    FeatureInput,
    KGMorphoFeatureOutput,
)


class TestKGMorphoFeaturesTool:
    @pytest.mark.asyncio
    async def test_arun(self, httpx_mock, brain_region_json_path):
        url = "http://fake_url"
        json_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "kg_morpho_features_response.json"
        )
        with open(json_path) as f:
            kg_morpho_features_response = json.load(f)

        httpx_mock.add_response(
            url=url,
            json=kg_morpho_features_response,
        )

        tool = KGMorphoFeatureTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
                "brainregion_path": brain_region_json_path,
            }
        )

        feature_input = FeatureInput(
            label="Section Tortuosity",
        )

        response = await tool._arun(
            brain_region_id="brain_region_id_link/549", features=feature_input
        )
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], KGMorphoFeatureOutput)

    @pytest.mark.asyncio
    async def test_arun_errors(self, httpx_mock, brain_region_json_path):
        url = "http://fake_url"

        # Mock issue (resolve query without results)
        httpx_mock.add_response(
            url=url,
            json={},
        )

        tool = KGMorphoFeatureTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
                "brainregion_path": brain_region_json_path,
            }
        )

        feature_input = FeatureInput(
            label="Section Tortuosity",
        )
        with pytest.raises(ToolException) as tool_exception:
            _ = await tool._arun(
                brain_region_id="brain_region_id_link/549", features=feature_input
            )
        assert tool_exception.value.args[0] == "'hits'"

    def test_create_query(self, brain_region_json_path):
        url = "http://fake_url"

        tool = KGMorphoFeatureTool(
            metadata={
                "url": url,
                "search_size": 2,
                "httpx_client": httpx.AsyncClient(),
                "token": "fake_token",
                "brainregion_path": brain_region_json_path,
            }
        )

        feature_input = FeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
        )

        brain_regions_ids = {"brain-region-id/68"}

        entire_query = tool.create_query(
            brain_regions_ids=brain_regions_ids, features=feature_input
        )
        expected_query = {
            "size": 2,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "term": {
                                            "brainRegion.@id.keyword": (
                                                "brain-region-id/68"
                                            )
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "nested": {
                                "path": "featureSeries",
                                "query": {
                                    "bool": {
                                        "must": [
                                            {
                                                "term": {
                                                    "featureSeries.label.keyword": (
                                                        "Soma Radius"
                                                    )
                                                }
                                            },
                                            {
                                                "term": {
                                                    "featureSeries.compartment.keyword": (
                                                        "NeuronMorphology"
                                                    )
                                                }
                                            },
                                        ]
                                    }
                                },
                            }
                        },
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
        assert isinstance(entire_query, dict)
        assert entire_query == expected_query

        # Case 2 with max value
        feature_input1 = FeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
            feat_range=FeatRangeInput(max_value=5),
        )
        entire_query1 = tool.create_query(
            brain_regions_ids=brain_regions_ids, features=feature_input1
        )
        expected_query1 = {
            "size": 2,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "term": {
                                            "brainRegion.@id.keyword": (
                                                "brain-region-id/68"
                                            )
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "nested": {
                                "path": "featureSeries",
                                "query": {
                                    "bool": {
                                        "must": [
                                            {
                                                "term": {
                                                    "featureSeries.label.keyword": (
                                                        "Soma Radius"
                                                    )
                                                }
                                            },
                                            {
                                                "term": {
                                                    "featureSeries.compartment.keyword": (
                                                        "NeuronMorphology"
                                                    )
                                                }
                                            },
                                            {
                                                "term": {
                                                    "featureSeries.statistic.keyword": (
                                                        "raw"
                                                    )
                                                }
                                            },
                                            {
                                                "range": {
                                                    "featureSeries.value": {"lte": 5.0}
                                                }
                                            },
                                        ]
                                    }
                                },
                            }
                        },
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
        assert entire_query1 == expected_query1

        # Case 3 with min value
        feature_input2 = FeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
            feat_range=FeatRangeInput(min_value=2),
        )
        entire_query2 = tool.create_query(
            brain_regions_ids=brain_regions_ids, features=feature_input2
        )
        expected_query2 = {
            "size": 2,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "term": {
                                            "brainRegion.@id.keyword": (
                                                "brain-region-id/68"
                                            )
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "nested": {
                                "path": "featureSeries",
                                "query": {
                                    "bool": {
                                        "must": [
                                            {
                                                "term": {
                                                    "featureSeries.label.keyword": (
                                                        "Soma Radius"
                                                    )
                                                }
                                            },
                                            {
                                                "term": {
                                                    "featureSeries.compartment.keyword": (
                                                        "NeuronMorphology"
                                                    )
                                                }
                                            },
                                            {
                                                "term": {
                                                    "featureSeries.statistic.keyword": (
                                                        "raw"
                                                    )
                                                }
                                            },
                                            {
                                                "range": {
                                                    "featureSeries.value": {"gte": 2.0}
                                                }
                                            },
                                        ]
                                    }
                                },
                            }
                        },
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
        assert entire_query2 == expected_query2

        # Case 4 with min and max value
        feature_input3 = FeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
            feat_range=FeatRangeInput(min_value=2, max_value=5),
        )
        entire_query3 = tool.create_query(
            brain_regions_ids=brain_regions_ids, features=feature_input3
        )
        expected_query3 = {
            "size": 2,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {
                                        "term": {
                                            "brainRegion.@id.keyword": (
                                                "brain-region-id/68"
                                            )
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "nested": {
                                "path": "featureSeries",
                                "query": {
                                    "bool": {
                                        "must": [
                                            {
                                                "term": {
                                                    "featureSeries.label.keyword": (
                                                        "Soma Radius"
                                                    )
                                                }
                                            },
                                            {
                                                "term": {
                                                    "featureSeries.compartment.keyword": (
                                                        "NeuronMorphology"
                                                    )
                                                }
                                            },
                                            {
                                                "term": {
                                                    "featureSeries.statistic.keyword": (
                                                        "raw"
                                                    )
                                                }
                                            },
                                            {
                                                "range": {
                                                    "featureSeries.value": {
                                                        "gte": 2.0,
                                                        "lte": 5.0,
                                                    }
                                                }
                                            },
                                        ]
                                    }
                                },
                            }
                        },
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
        assert entire_query3 == expected_query3
