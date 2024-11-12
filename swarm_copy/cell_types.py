"""Cell types metadata."""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__file__)


class CellTypesMeta:
    """Class holding the hierarchical cell types metadata.

    Typically, such information would be parsed from a `celltypes.json`
    file.
    """

    def __init__(self) -> None:
        self.name_: dict[str, str] = {}
        self.descendants_ids: dict[str, set[str]] = {}

    def descendants(self, ids: str | set[str]) -> set[str]:
        """Find all descendants of given cell type.

        The result is inclusive, i.e. the input region IDs will be
        included in the result.

        Parameters
        ----------
        ids : set or iterable of set
            A region ID or a collection of region IDs to collect
            descendants for.

        Returns
        -------
        set
            All descendant region IDs of the given regions, including the input cell type themselves.
        """
        if isinstance(ids, str):
            unique_ids = {ids}
        else:
            unique_ids = set(ids)

        descendants = unique_ids.copy()
        for id_ in unique_ids:
            try:
                descendants.update(self.descendants_ids[id_])
            except KeyError:
                logger.info(f"{id_} does not have any child in the hierarchy.")
        return descendants

    def save_config(self, json_file_path: str | Path) -> None:
        """Save the actual configuration in a json file.

        Parameters
        ----------
        json_file_path
            Path where to save the json file
        """
        descendants = {}
        for k, v in self.descendants_ids.items():
            descendants[k] = list(v)

        to_save = {
            "names": self.name_,
            "descendants_ids": descendants,
        }
        with open(json_file_path, "w") as fs:
            fs.write(json.dumps(to_save))

    @classmethod
    def load_config(cls, json_file_path: str | Path) -> "CellTypesMeta":
        """Load a configuration in a json file and return a 'CellTypesMeta' instance.

        Parameters
        ----------
        json_file_path
            Path to the json file containing the brain region hierarchy

        Returns
        -------
            RegionMeta class with pre-loaded hierarchy
        """
        with open(json_file_path, "r") as fs:
            to_load = json.load(fs)

        descendants_ids = {}
        for k, v in to_load["descendants_ids"].items():
            descendants_ids[k] = set(v)

        self = cls()

        self.name_ = to_load["names"]
        self.descendants_ids = descendants_ids
        return self

    @classmethod
    def from_dict(cls, hierarchy: dict[str, Any]) -> "CellTypesMeta":
        """Load the structure graph from a dict and create a Class instance.

        Parameters
        ----------
        hierarchy : dict[str, Any]
            Hierarchy in dictionary format.

        Returns
        -------
        RegionMeta
            The initialized instance of this class.
        """
        names = {}
        initial_json: dict[str, set[str]] = defaultdict(set)
        for i in range(len(hierarchy["defines"])):
            cell_type = hierarchy["defines"][i]
            names[cell_type["@id"]] = (
                cell_type["label"] if "label" in cell_type else None
            )
            if "subClassOf" not in cell_type.keys():
                initial_json[cell_type["@id"]] = set()
                continue
            parents = cell_type["subClassOf"]
            for parent in parents:
                initial_json[parent].add(hierarchy["defines"][i]["@id"])

        current_json = initial_json.copy()

        for i in range(10):  # maximum number of attempts
            new_json = {}
            for k, v in current_json.items():
                new_set = v.copy()
                for child in v:
                    if child in current_json.keys():
                        new_set.update(current_json[child])
                new_json[k] = new_set

            if new_json == current_json:
                break

            if i == 9:
                raise ValueError("Did not manage to create a CellTypesMeta object.")

            current_json = new_json.copy()

        self = cls()

        self.name_ = names
        self.descendants_ids = new_json

        return self

    @classmethod
    def from_json(cls, json_path: Path | str) -> "CellTypesMeta":
        """Load the structure graph from a JSON file and create a Class instance.

        Parameters
        ----------
        json_path : str or pathlib.Path

        Returns
        -------
        RegionMeta
            The initialized instance of this class.
        """
        with open(json_path) as fh:
            hierarchy = json.load(fh)

        return cls.from_dict(hierarchy)


def get_celltypes_descendants(cell_type_id: str, json_path: str | Path) -> set[str]:
    """Get all descendant of a brain region id.

    Parameters
    ----------
    cell_type_id
        Cell type ID for which to find the descendants list.
    json_path
        Path to the json file containing the Cell Types hierarchy.

    Returns
    -------
        Set of descendants of a cell type
    """
    try:
        region_meta = CellTypesMeta.load_config(json_path)
        hierarchy_ids = region_meta.descendants(cell_type_id)
    except IOError:
        logger.warning(f"The file {json_path} doesn't exist.")
        hierarchy_ids = {cell_type_id}

    return hierarchy_ids
