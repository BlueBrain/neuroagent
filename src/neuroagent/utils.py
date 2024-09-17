"""Utilies for neuroagent."""

import json
import logging
import numbers
import re
from pathlib import Path
from typing import Any, Iterator

from httpx import AsyncClient

from neuroagent.schemas import KGMetadata

logger = logging.getLogger(__name__)


class RegionMeta:
    """Class holding the hierarchical region metadata.

    Typically, such information would be parsed from a `brain_regions.json`
    file.

    Parameters
    ----------
    background_id : int, optional
        Override the default ID for the background.
    """

    def __init__(self, background_id: int = 0) -> None:
        self.background_id = background_id
        self.root_id: int | None = None

        self.name_: dict[int, str] = {self.background_id: "background"}
        self.st_level: dict[int, int | None] = {self.background_id: None}

        self.parent_id: dict[int, int] = {self.background_id: background_id}
        self.children_ids: dict[int, list[int]] = {self.background_id: []}

    def children(self, region_id: int) -> tuple[int, ...]:
        """Get all child region IDs of a given region.

        Note that by children we mean only the direct children, much like
        by parent we only mean the direct parent. The cumulative quantities
        that span all generations are called ancestors and descendants.

        Parameters
        ----------
        region_id : int
            The region ID in question.

        Returns
        -------
        int
            The region ID of a child region.
        """
        return tuple(self.children_ids[region_id])

    def descendants(self, ids: int | list[int]) -> set[int]:
        """Find all descendants of given regions.

        The result is inclusive, i.e. the input region IDs will be
        included in the result.

        Parameters
        ----------
        ids : int or iterable of int
            A region ID or a collection of region IDs to collect
            descendants for.

        Returns
        -------
        set
            All descendant region IDs of the given regions, including the input
            regions themselves.
        """
        if isinstance(ids, numbers.Integral):
            unique_ids: set[int] = {ids}
        elif isinstance(ids, set):
            unique_ids = set(ids)

        def iter_descendants(region_id: int) -> Iterator[int]:
            """Iterate over all descendants of a given region ID.

            Parameters
            ----------
            region_id
                Integer representing the id of the region

            Returns
            -------
                Iterator with descendants of the region
            """
            yield region_id
            for child in self.children(region_id):
                yield child
                yield from iter_descendants(child)

        descendants = set()
        for id_ in unique_ids:
            descendants |= set(iter_descendants(id_))

        return descendants

    def save_config(self, json_file_path: str | Path) -> None:
        """Save the actual configuration in a json file.

        Parameters
        ----------
        json_file_path
            Path where to save the json file
        """
        to_save = {
            "root_id": self.root_id,
            "names": self.name_,
            "st_level": self.st_level,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
        }
        with open(json_file_path, "w") as fs:
            fs.write(json.dumps(to_save))

    @classmethod
    def load_config(cls, json_file_path: str | Path) -> "RegionMeta":
        """Load a configuration in a json file and return a 'RegionMeta' instance.

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

        # Needed to convert json 'str' keys to int.
        for k1 in to_load.keys():
            if not isinstance(to_load[k1], int):
                to_load[k1] = {int(k): v for k, v in to_load[k1].items()}

        self = cls()

        self.root_id = to_load["root_id"]
        self.name_ = to_load["names"]
        self.st_level = to_load["st_level"]
        self.parent_id = to_load["parent_id"]
        self.children_ids = to_load["children_ids"]

        return self

    @classmethod
    def from_KG_dict(cls, KG_hierarchy: dict[str, Any]) -> "RegionMeta":
        """Construct an instance from the json of the Knowledge Graph.

        Parameters
        ----------
        KG_hierarchy : dict
            The dictionary of the region hierarchy, provided by the KG.

        Returns
        -------
        region_meta : RegionMeta
            The initialized instance of this class.
        """
        self = cls()

        for brain_region in KG_hierarchy["defines"]:
            # Filter out wrong elements of the KG.
            if "identifier" in brain_region.keys():
                region_id = int(brain_region["identifier"])

                # Check if we are at root.
                if "isPartOf" not in brain_region.keys():
                    self.root_id = int(region_id)
                    self.parent_id[region_id] = self.background_id
                else:
                    # Strip url to only keep ID.
                    self.parent_id[region_id] = int(
                        brain_region["isPartOf"][0].rsplit("/")[-1]
                    )
                self.children_ids[region_id] = []

                self.name_[region_id] = brain_region["label"]

                if "st_level" not in brain_region.keys():
                    self.st_level[region_id] = None
                else:
                    self.st_level[region_id] = brain_region["st_level"]

        # Once every parents are set, we can deduce all childrens.
        for child_id, parent_id in self.parent_id.items():
            if parent_id is not None:
                self.children_ids[int(parent_id)].append(child_id)

        return self

    @classmethod
    def load_json(cls, json_path: Path | str) -> "RegionMeta":
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
            KG_hierarchy = json.load(fh)

        return cls.from_KG_dict(KG_hierarchy)


def get_descendants_id(brain_region_id: str, json_path: str | Path) -> set[str]:
    """Get all descendant of a brain region id.

    Parameters
    ----------
    brain_region_id
        Brain region ID to find descendants for.
    json_path
        Path to the json file containing the BR hierarchy

    Returns
    -------
        Set of descendants of a brain region
    """
    # Split a brain region ID of the form "http://api.brain-map.org/api/v2/data/Structure/123" into base + id.
    id_base, _, brain_region_str = brain_region_id.rpartition("/")
    try:
        # Convert the id into an int
        brain_region_int = int(brain_region_str)

        # Get the descendant ids of this BR (as int).
        region_meta = RegionMeta.load_config(json_path)
        hierarchy = region_meta.descendants(brain_region_int)

        # Recast the descendants into the form "http://api.brain-map.org/api/v2/data/Structure/123"
        hierarchy_ids = {f"{id_base}/{h}" for h in hierarchy}
    except ValueError:
        logger.info(
            f"The brain region {brain_region_id} didn't end with an int. Returning only"
            " the parent one."
        )
        hierarchy_ids = {brain_region_id}
    except IOError:
        logger.warning(f"The file {json_path} doesn't exist.")
        hierarchy_ids = {brain_region_id}

    return hierarchy_ids


async def get_file_from_KG(
    file_url: str,
    file_name: str,
    view_url: str,
    token: str,
    httpx_client: AsyncClient,
) -> dict[str, Any]:
    """Get json file for brain region / cell types from the KG.

    Parameters
    ----------
    file_url
        URL of the view containing the potential file
    file_name
        Name of the file to download
    view_url
        URL of the sparql view where to send the request to get the file url
    token
        Token used to access the knowledge graph
    httpx_client
        AsyncClient to send requests

    Returns
    -------
        Json contained in the downloaded file
    """
    sparql_query = """
    PREFIX schema: <http://schema.org/>

    SELECT DISTINCT ?file_url
    WHERE {{
        {file_url} schema:distribution ?json_distribution .
        ?json_distribution schema:name "{file_name}" ;
                        schema:contentUrl ?file_url .
    }}
    LIMIT 1""".format(file_url=file_url, file_name=file_name)
    try:
        file_response = None

        # Get the url of the relevant file
        url_response = await httpx_client.post(
            url=view_url,
            content=sparql_query,
            headers={
                "Content-Type": "text/plain",
                "Accept": "application/sparql-results+json",
                "Authorization": f"Bearer {token}",
            },
        )

        # Download the file
        file_response = await httpx_client.get(
            url=url_response.json()["results"]["bindings"][0]["file_url"]["value"],
            headers={
                "Accept": "*/*",
                "Authorization": f"Bearer {token}",
            },
        )

        return file_response.json()

    except ValueError:
        # Issue with KG
        if url_response.status_code != 200:
            raise ValueError(
                f"Could not find the file url, status code : {url_response.status_code}"
            )
        # File not found
        elif file_response:
            raise ValueError(
                f"Could not find the file, status code : {file_response.status_code}"
            )
        else:
            # Issue when downloading the file
            raise ValueError("url_response did not return a Json.")
    except IndexError:
        # No file url found
        raise IndexError("No file url was found.")
    except KeyError:
        # Json has weird format
        raise KeyError("Incorrect json format.")


def is_lnmc(contributors: list[dict[str, Any]]) -> bool:
    """Extract contributor affiliation out of the contributors."""
    lnmc_contributors = {
        "https://www.grid.ac/institutes/grid.5333.6",
        "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/yshi",
        "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/jyi",
        "https://bbp.epfl.ch/neurosciencegraph/data/664380c8-5a22-4974-951c-68ca78c0b1f1",
        "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/perin",
        "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/rajnish",
        "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/ajaquier",
        "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/gevaert",
        "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/kanari",
    }
    for contributor in contributors:
        if "@id" in contributor and contributor["@id"] in lnmc_contributors:
            return True

    return False


async def get_kg_data(
    object_id: str,
    httpx_client: AsyncClient,
    url: str,
    token: str,
    preferred_format: str,
) -> tuple[bytes, KGMetadata]:
    """Download any knowledge graph object.

    Parameters
    ----------
    object_id
        ID of the object to which the file is attached
    httpx_client
        AsyncClient to send the request
    url
        URL of the KG view where the object is located
    token
        Token used to access the knowledge graph
    preferred_format
        Extension of the file to download

    Returns
    -------
        Tuple containing the file's content and the associated metadata

    Raises
    ------
    ValueError
        If the object ID is not found the knowledge graph.
    """
    # Extract the id from the specified input (useful for rewoo)
    extracted_id = re.findall(pattern=r"https?://\S+[a-zA-Z0-9]", string=object_id)
    if not extracted_id:
        raise ValueError(f"The provided ID ({object_id}) is not valid.")
    else:
        object_id = extracted_id[0]

    # Create ES query to retrieve the object in KG
    query = {
        "size": 1,
        "track_total_hits": True,
        "query": {
            "bool": {
                "must": [
                    {
                        "term": {
                            "@id.keyword": object_id,
                        }
                    }
                ]
            }
        },
    }

    # Retrieve the object of interest from KG
    response = await httpx_client.post(
        url=url,
        headers={"Authorization": f"Bearer {token}"},
        json=query,
    )

    if response.status_code != 200 or len(response.json()["hits"]["hits"]) == 0:
        raise ValueError(f"We did not find the object {object_id} you are asking")

    # Get the metadata of the object
    response_data = response.json()["hits"]["hits"][0]["_source"]

    # Ensure we got the expected object
    if response_data["@id"] != object_id:
        raise ValueError(f"We did not find the object {object_id} you are asking")

    metadata: dict[str, Any] = dict()
    metadata["brain_region"] = response_data["brainRegion"]["label"]
    distributions = response_data["distribution"]

    # Extract the format of the file
    has_preferred_format = [
        i
        for i, dis in enumerate(distributions)
        if dis["encodingFormat"] == f"application/{preferred_format}"
    ]

    # Set the file extension accordingly if preferred format found
    if len(has_preferred_format) > 0:
        chosen_dist = distributions[has_preferred_format[0]]
        metadata["file_extension"] = preferred_format
    else:
        chosen_dist = distributions[0]
        metadata["file_extension"] = chosen_dist["encodingFormat"].split("/")[1]
        logger.info(
            "The format you specified was not available."
            f" {metadata['file_extension']} was chosen instead."
        )

    # Check if the object has been added by the LNMC lab (useful for traces)
    if "contributors" in response_data:
        metadata["is_lnmc"] = is_lnmc(response_data["contributors"])

    # Download the file
    url = chosen_dist["contentUrl"]
    content_response = await httpx_client.get(
        url=url,
        headers={"Authorization": f"Bearer {token}"},
    )

    # Return its content and the associated metadata
    object_content = content_response.content
    return object_content, KGMetadata(**metadata)
