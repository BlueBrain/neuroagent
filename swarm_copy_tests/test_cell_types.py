"""Test cell types meta functions."""
import logging
from pathlib import Path

import pytest

from swarm_copy.cell_types import CellTypesMeta, get_celltypes_descendants

CELL_TYPES_FILE = Path(__file__).parent / "data" / "kg_cell_types_hierarchy_test.json"


@pytest.mark.parametrize(
    "cell_type_id,expected_descendants",
    [
        (
            "https://bbp.epfl.ch/ontologies/core/bmo/BrainCellType",
            {
                "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/L23_PTPC",
                "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
                "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/GCL_GC",
                "https://bbp.epfl.ch/ontologies/core/bmo/BrainCellType",
            },
        ),
        (
            "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
            {
                "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
                "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
            },
        ),
        (
            "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
            {
                "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
            },
        ),
    ],
)
def test_get_celltypes_descendants(cell_type_id, expected_descendants, tmp_path):
    cell_types_meta = CellTypesMeta.from_json(CELL_TYPES_FILE)
    save_file = tmp_path / "tmp_config_cell_types_meta.json"
    cell_types_meta.save_config(save_file)

    descendants = get_celltypes_descendants(cell_type_id, json_path=save_file)
    assert expected_descendants == descendants


class TestCellTypesMeta:
    def test_from_json(self):
        ct_meta = CellTypesMeta.from_json(CELL_TYPES_FILE)
        assert isinstance(ct_meta.name_, dict)
        assert isinstance(ct_meta.descendants_ids, dict)

        expected_names = {
            "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint": "cACint",
            "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/GCL_GC": "GCL_GC",
            "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/L23_PTPC": (
                "L23_PTPC"
            ),
        }

        assert ct_meta.name_ == expected_names
        assert ct_meta.descendants_ids[
            "https://bbp.epfl.ch/ontologies/core/mtypes/HippocampusMType"
        ] == {"http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/GCL_GC"}
        assert ct_meta.descendants_ids[
            "https://bbp.epfl.ch/ontologies/core/bmo/BrainCellType"
        ] == {
            "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/L23_PTPC",
            "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
            "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/GCL_GC",
        }

    def test_from_dict(self):
        test_dict = {
            "defines": [
                {
                    "@id": "id1",
                    "label": "cell1",
                    "subClassOf": []
                },
                {
                    "@id": "id2",
                    "label": "cell2",
                    "subClassOf": ["id1"]
                },
                {
                    "@id": "id3",
                    "subClassOf": ["id2"]
                }
            ]
        }
        cell_meta = CellTypesMeta.from_dict(test_dict)
        assert isinstance(cell_meta, CellTypesMeta)
        assert cell_meta.name_ == {"id1": "cell1", "id2": "cell2", "id3": None}
        assert cell_meta.descendants_ids == {"id1": {"id2", "id3"}, "id2": {"id3"}}

    def test_from_dict_missing_label(self):
        test_dict = {
            "defines": [
                {
                    "@id": "id1",
                    "subClassOf": []
                },
                {
                    "@id": "id2",
                    "subClassOf": ["id1"]
                }
            ]
        }
        cell_meta = CellTypesMeta.from_dict(test_dict)
        assert cell_meta.name_ == {"id1": None, "id2": None}
        assert cell_meta.descendants_ids == {"id1": {"id2"}}

    def test_from_dict_missing_subClassOf(self):
        test_dict = {
            "defines": [
                {
                    "@id": "id1",
                    "label": "cell1",
                },
                {
                    "@id": "id2",
                    "label": "cell2",
                    "subClassOf": ["id1"]
                }
            ]
        }
        cell_meta = CellTypesMeta.from_dict(test_dict)
        assert cell_meta.name_ == {"id1": "cell1", "id2": "cell2"}
        assert cell_meta.descendants_ids == {"id1": {"id2"}}

    @pytest.mark.parametrize(
        "cell_type_id,expected_descendants",
        [
            (
                "https://bbp.epfl.ch/ontologies/core/bmo/BrainCellType",
                {
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/L23_PTPC",
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/GCL_GC",
                    "https://bbp.epfl.ch/ontologies/core/bmo/BrainCellType",
                },
            ),
            (
                "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
                {
                    "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
                },
            ),
            (
                [
                    "https://bbp.epfl.ch/ontologies/core/bmo/BrainCellType",
                    "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
                ],
                {
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/L23_PTPC",
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/mtypes/GCL_GC",
                    "https://bbp.epfl.ch/ontologies/core/bmo/BrainCellType",
                    "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
                },
            ),
            (
                "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
                {
                    "https://bbp.epfl.ch/ontologies/core/bmo/NeuronElectricalType",
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
                },
            ),
            (
                "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
                {
                    "http://bbp.epfl.ch/neurosciencegraph/ontologies/etypes/cACint",
                },
            ),
        ],
    )
    def test_descendants(self, cell_type_id, expected_descendants):
        ct_meta = CellTypesMeta.from_json(CELL_TYPES_FILE)
        assert ct_meta.descendants(cell_type_id) == expected_descendants

    def test_load_and_save_config(self, tmp_path):
        ct_meta = CellTypesMeta.from_json(CELL_TYPES_FILE)
        file_path = tmp_path / "ct_meta_tmp.json"
        ct_meta.save_config(file_path)
        ct_meta2 = CellTypesMeta.load_config(file_path)
        assert ct_meta.name_ == ct_meta2.name_
        assert ct_meta.descendants_ids == ct_meta2.descendants_ids
