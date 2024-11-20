"""Test utility functions."""

import json
from pathlib import Path

import pytest
from httpx import AsyncClient

from swarm_copy.schemas import KGMetadata
from swarm_copy.utils import (
    RegionMeta,
    get_descendants_id,
    get_file_from_KG,
    get_kg_data,
    is_lnmc,
)


@pytest.mark.parametrize(
    "brain_region_id,expected_descendants",
    [
        ("brain-region-id/68", {"brain-region-id/68"}),
        (
            "another-brain-region-id/985",
            {
                "another-brain-region-id/320",
                "another-brain-region-id/648",
                "another-brain-region-id/844",
                "another-brain-region-id/882",
                "another-brain-region-id/943",
                "another-brain-region-id/985",
                "another-brain-region-id/3718675619",
                "another-brain-region-id/1758306548",
            },
        ),
        (
            "another-brain-region-id/369",
            {
                "another-brain-region-id/450",
                "another-brain-region-id/369",
                "another-brain-region-id/1026",
                "another-brain-region-id/854",
                "another-brain-region-id/577",
                "another-brain-region-id/625",
                "another-brain-region-id/945",
                "another-brain-region-id/1890964946",
                "another-brain-region-id/3693772975",
            },
        ),
        (
            "another-brain-region-id/178",
            {
                "another-brain-region-id/316",
                "another-brain-region-id/178",
                "another-brain-region-id/300",
                "another-brain-region-id/1043765183",
            },
        ),
        ("brain-region-id/not-a-int", {"brain-region-id/not-a-int"}),
    ],
)
def test_get_descendants(brain_region_id, expected_descendants, brain_region_json_path):
    descendants = get_descendants_id(brain_region_id, json_path=brain_region_json_path)
    assert expected_descendants == descendants


def test_get_descendants_errors(brain_region_json_path):
    brain_region_id = "does-not-exits/1111111111"
    with pytest.raises(KeyError):
        get_descendants_id(brain_region_id, json_path=brain_region_json_path)


def test_RegionMeta_from_KG_dict():
    with open(
        Path(__file__).parent / "data" / "KG_brain_regions_hierarchy_test.json"
    ) as fh:
        KG_hierarchy = json.load(fh)

    RegionMeta_test = RegionMeta.from_KG_dict(KG_hierarchy)

    # check names.
    assert RegionMeta_test.name_[1] == "Tuberomammillary nucleus, ventral part"
    assert (
        RegionMeta_test.name_[2]
        == "Superior colliculus, motor related, intermediate gray layer"
    )
    assert RegionMeta_test.name_[3] == "Primary Motor Cortex"

    # check parents / childrens.
    assert RegionMeta_test.parent_id[1] == 2
    assert RegionMeta_test.parent_id[2] == 0
    assert RegionMeta_test.parent_id[3] == 2
    assert RegionMeta_test.children_ids[1] == []
    assert RegionMeta_test.children_ids[2] == [1, 3]
    assert RegionMeta_test.children_ids[3] == []


def test_RegionMeta_save_load(tmp_path: Path):
    # load fake file from KG
    with open(
        Path(__file__).parent / "data" / "KG_brain_regions_hierarchy_test.json"
    ) as fh:
        KG_hierarchy = json.load(fh)

    RegionMeta_test = RegionMeta.from_KG_dict(KG_hierarchy)

    # save / load file.
    json_file = tmp_path / "test.json"
    RegionMeta_test.save_config(json_file)
    RegionMeta_test.load_config(json_file)

    # check names.
    assert RegionMeta_test.name_[1] == "Tuberomammillary nucleus, ventral part"
    assert (
        RegionMeta_test.name_[2]
        == "Superior colliculus, motor related, intermediate gray layer"
    )
    assert RegionMeta_test.name_[3] == "Primary Motor Cortex"

    # check parents / childrens.
    assert RegionMeta_test.parent_id[1] == 2
    assert RegionMeta_test.parent_id[2] == 0
    assert RegionMeta_test.parent_id[3] == 2
    assert RegionMeta_test.children_ids[1] == []
    assert RegionMeta_test.children_ids[2] == [1, 3]
    assert RegionMeta_test.children_ids[3] == []
