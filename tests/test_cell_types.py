"""Test cell types meta functions."""

from pathlib import Path

import pytest
from neuroagent.cell_types import CellTypesMeta, get_celltypes_descendants

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
