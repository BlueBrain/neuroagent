"""Tests KG Morpho Features tool."""

import json
from pathlib import Path

import httpx
import pytest

from neuroagent.tools import KGMorphoFeatureTool
from neuroagent.tools.kg_morpho_features_tool import (
    KGFeatRangeInput,
    KGFeatureInput,
    KGMorphoFeatureInput,
    KGMorphoFeatureMetadata,
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

        feature_input = KGFeatureInput(
            label="Section Tortuosity",
        )

        tool = KGMorphoFeatureTool(
            input_schema=KGMorphoFeatureInput(
                brain_region_id="brain_region_id_link/549", features=feature_input
            ),
            metadata=KGMorphoFeatureMetadata(
                knowledge_graph_url=url,
                kg_morpho_feature_search_size=2,
                token="fake_token",
                brainregion_path=brain_region_json_path,
                httpx_client=httpx.AsyncClient(),
            ),
        )
        response = await tool.arun()
        assert isinstance(response, list)
        assert len(response) == 2
        assert isinstance(response[0], dict)

    @pytest.mark.asyncio
    async def test_arun_errors(self, httpx_mock, brain_region_json_path):
        url = "http://fake_url"

        # Mock issue (resolve query without results)
        httpx_mock.add_response(
            url=url,
            json={},
        )

        feature_input = KGFeatureInput(
            label="Section Tortuosity",
        )
        tool = KGMorphoFeatureTool(
            input_schema=KGMorphoFeatureInput(
                brain_region_id="brain_region_id_link/549", features=feature_input
            ),
            metadata=KGMorphoFeatureMetadata(
                knowledge_graph_url=url,
                kg_morpho_feature_search_size=2,
                token="fake_token",
                brainregion_path=brain_region_json_path,
                httpx_client=httpx.AsyncClient(),
            ),
        )

        with pytest.raises(KeyError) as tool_exception:
            await tool.arun()
        assert tool_exception.value.args[0] == "hits"

    def test_create_query(self, brain_region_json_path):
        url = "http://fake_url"

        feature_input = KGFeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
        )

        brain_regions_ids = {"brain-region-id/68"}

        tool = KGMorphoFeatureTool(
            input_schema=KGMorphoFeatureInput(
                brain_region_id="", features=feature_input
            ),
            metadata=KGMorphoFeatureMetadata(
                knowledge_graph_url=url,
                kg_morpho_feature_search_size=2,
                token="fake_token",
                brainregion_path=brain_region_json_path,
                httpx_client=httpx.AsyncClient(),
            ),
        )

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

    def test_create_query_with_max_value(self, brain_region_json_path):
        url = "http://fake_url"

        feature_input = KGFeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
            feat_range=KGFeatRangeInput(max_value=5),
        )

        brain_regions_ids = {"brain-region-id/68"}

        tool = KGMorphoFeatureTool(
            input_schema=KGMorphoFeatureInput(
                brain_region_id="", features=feature_input
            ),
            metadata=KGMorphoFeatureMetadata(
                knowledge_graph_url=url,
                kg_morpho_feature_search_size=2,
                token="fake_token",
                brainregion_path=brain_region_json_path,
                httpx_client=httpx.AsyncClient(),
            ),
        )

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
        assert entire_query == expected_query

    def test_create_query_with_min_value(self, brain_region_json_path):
        url = "http://fake_url"

        feature_input = KGFeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
            feat_range=KGFeatRangeInput(min_value=2),
        )

        tool = KGMorphoFeatureTool(
            input_schema=KGMorphoFeatureInput(
                brain_region_id="", features=feature_input
            ),
            metadata=KGMorphoFeatureMetadata(
                knowledge_graph_url=url,
                kg_morpho_feature_search_size=2,
                token="fake_token",
                brainregion_path=brain_region_json_path,
                httpx_client=httpx.AsyncClient(),
            ),
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
        assert entire_query == expected_query

    def test_create_query_with_min_max_value(self, brain_region_json_path):
        url = "http://fake_url"

        feature_input = KGFeatureInput(
            label="Soma Radius",
            compartment="NeuronMorphology",
            feat_range=KGFeatRangeInput(min_value=2, max_value=5),
        )

        tool = KGMorphoFeatureTool(
            input_schema=KGMorphoFeatureInput(
                brain_region_id="", features=feature_input
            ),
            metadata=KGMorphoFeatureMetadata(
                knowledge_graph_url=url,
                kg_morpho_feature_search_size=2,
                token="fake_token",
                brainregion_path=brain_region_json_path,
                httpx_client=httpx.AsyncClient(),
            ),
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
        assert entire_query == expected_query
