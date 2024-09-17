"""Utils related to object resolving in the KG."""

import asyncio
import logging
import re
from typing import Literal

from httpx import AsyncClient

logger = logging.getLogger(__name__)


SPARQL_QUERY = """
   PREFIX bmc: <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmc/>
   PREFIX bmo: <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/>
   PREFIX bmoutils: <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmoutils/>
   PREFIX commonshapes: <https://neuroshapes.org/commons/>
   PREFIX datashapes: <https://neuroshapes.org/dash/>
   PREFIX dc: <http://purl.org/dc/elements/1.1/>
   PREFIX dcat: <http://www.w3.org/ns/dcat#>
   PREFIX dcterms: <http://purl.org/dc/terms/>
   PREFIX mba: <http://api.brain-map.org/api/v2/data/Structure/>
   PREFIX nsg: <https://neuroshapes.org/>
   PREFIX nxv: <https://bluebrain.github.io/nexus/vocabulary/>
   PREFIX oa: <http://www.w3.org/ns/oa#>
   PREFIX obo: <http://purl.obolibrary.org/obo/>
   PREFIX owl: <http://www.w3.org/2002/07/owl#>
   PREFIX prov: <http://www.w3.org/ns/prov#>
   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX schema: <http://schema.org/>
   PREFIX sh: <http://www.w3.org/ns/shacl#>
   PREFIX shsh: <http://www.w3.org/ns/shacl-shacl#>
   PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
   PREFIX vann: <http://purl.org/vocab/vann/>
   PREFIX void: <http://rdfs.org/ns/void#>
   PREFIX xml: <http://www.w3.org/XML/1998/namespace/>
   PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
   PREFIX : <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/>

           CONSTRUCT {{
               ?id a ?type ;
               rdfs:label ?label ;
               skos:prefLabel ?prefLabel ;
               skos:altLabel ?altLabel ;
               skos:definition ?definition;
               rdfs:subClassOf ?subClassOf ;
               rdfs:isDefinedBy ?isDefinedBy ;
               skos:notation ?notation ;
               skos:definition ?definition ;
               nsg:atlasRelease ?atlasRelease ;
               schema:identifier ?identifier ;
               <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/delineatedBy> ?delineatedBy ;
               <https://neuroshapes.org/hasLayerLocationPhenotype> ?hasLayerLocationPhenotype ;
               <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/representedInAnnotation> ?representedInAnnotation ;
               <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/hasLeafRegionPart> ?hasLeafRegionPart ;
               schema:isPartOf ?isPartOf ;
               <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/isLayerPartOf> ?isLayerPartOf .
           }} WHERE {{
               GRAPH ?g {{
                   ?id a ?type ;
                       rdfs:label ?label ;
                   OPTIONAL {{
                   ?id rdfs:subClassOf ?subClassOf ;
                   }}
                   OPTIONAL {{
                   ?id skos:definition ?definition ;
                   }}
                   OPTIONAL {{
                   ?id skos:prefLabel ?prefLabel .
                   }}
                   OPTIONAL {{
                   ?id skos:altLabel ?altLabel .
                   }}
                   OPTIONAL {{
                   ?id rdfs:isDefinedBy ?isDefinedBy .
                   }}
                   OPTIONAL {{
                   ?id skos:notation ?notation .
                   }}
                   OPTIONAL {{
                   ?id skos:definition ?definition .
                   }}
                   OPTIONAL {{
                   ?id nsg:atlasRelease ?atlasRelease .
                   }}
                   OPTIONAL {{
                   ?id <https://neuroshapes.org/hasLayerLocationPhenotype> ?hasLayerLocationPhenotype .
                   }}
                   OPTIONAL {{
                   ?id schema:identifier ?identifier .
                   }}
                   OPTIONAL {{
                   ?id <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/delineatedBy> ?delineatedBy .
                   }}
                   OPTIONAL {{
                   ?id <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/representedInAnnotation> ?representedInAnnotation .
                   }}
                   OPTIONAL {{
                   ?id <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/hasLeafRegionPart> ?hasLeafRegionPart .
                   }}
                   OPTIONAL {{
                   ?id schema:isPartOf ?isPartOf .
                   }}
                   OPTIONAL {{
                   ?id <https://sbo-nexus-delta.shapes-registry.org/ontologies/core/bmo/isLayerPartOf> ?isLayerPartOf .
                   }}
                   OPTIONAL {{
                   ?id <https://neuroshapes.org/units> ?units .
                   }}
                   {{
                   SELECT * WHERE {{
                       {{ ?id <https://bluebrain.github.io/nexus/vocabulary/deprecated> "false"^^xsd:boolean ; a owl:Class ;
    rdfs:subClassOf* {resource} ; rdfs:label ?label  FILTER regex(?label, {keyword}, "i") }} UNION
                       {{ ?id <https://bluebrain.github.io/nexus/vocabulary/deprecated> "false"^^xsd:boolean ; a owl:Class ;
    rdfs:subClassOf* {resource} ; skos:notation ?notation  FILTER regex(?notation, {keyword}, "i") }} UNION
                       {{ ?id <https://bluebrain.github.io/nexus/vocabulary/deprecated> "false"^^xsd:boolean ; a owl:Class ;
    rdfs:subClassOf* {resource} ; skos:prefLabel ?prefLabel  FILTER regex(?prefLabel, {keyword}, "i") }} UNION
                       {{ ?id <https://bluebrain.github.io/nexus/vocabulary/deprecated> "false"^^xsd:boolean ; a owl:Class ;
    rdfs:subClassOf* {resource} ; skos:altLabel ?altLabel  FILTER regex(?altLabel, {keyword}, "i") }}
                   }} LIMIT {search_size}
                   }}
               }}
           }}
"""  # ORDER BY ?id


async def sparql_exact_resolve(
    query: str,
    resource_type: str,
    sparql_view_url: str,
    token: str,
    httpx_client: AsyncClient,
) -> list[dict[str, str]] | None:
    """Resolve query with the knowledge graph using sparql (exact match).

    Parameters
    ----------
    query
        Query to resolve (needs to be a brain region).
    resource_type
        Type of resource to match.
    sparql_view_url
        URL to the knowledge graph.
    token
        Token to access the KG.
    httpx_client
        Async Client.

    Returns
    -------
    list[dict[str, str]] | None
        List of brain region IDs and names (only one for exact match).
    """
    # For exact match query remove punctuation and add ^ + $ for regex
    sparql_query_exact = SPARQL_QUERY.format(
        keyword=f'"^{escape_punctuation(query)}$"',
        search_size=1,
        resource=resource_type,
    ).replace("\n", "")

    # Send the sparql query
    response = await httpx_client.post(
        url=sparql_view_url,
        content=sparql_query_exact,
        headers={
            "Content-Type": "text/plain",
            "Accept": "application/sparql-results+json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        # Get the BR or mtype ID
        object_id = response.json()["results"]["bindings"][0]["subject"]["value"]

        # Get the BR or mtype name
        object_name = next(
            (
                resp["object"]["value"]
                for resp in response.json()["results"]["bindings"]
                if "literal" in resp["object"]["type"]
            )
        )
        logger.info(
            f"Found object {object_name} id {object_id} from the exact"
            " match Sparql query."
        )

        # Return a single element (because exact match)
        return [{"label": object_name, "id": object_id}]

    # If nothing matched, notify parent function that exact match didn't work
    except (IndexError, KeyError):
        return None


async def sparql_fuzzy_resolve(
    query: str,
    resource_type: str,
    sparql_view_url: str,
    token: str,
    httpx_client: AsyncClient,
    search_size: int = 10,
) -> list[dict[str, str]] | None:
    """Resolve query with the knowledge graph using sparql (fuzzy match).

    Parameters
    ----------
    query
        Query to resolve (needs to be a brain region).
    resource_type
        Type of resource to match.
    sparql_view_url
        URL to the knowledge graph.
    token
        Token to access the KG.
    httpx_client
        Async Client.
    search_size
        Number of results to retrieve.


    Returns
    -------
    list[dict[str, str]] | None
        List of brain region IDs and names. None if none found.
    """
    # Prepare the fuzzy sparql query
    sparql_query_fuzzy = SPARQL_QUERY.format(
        keyword=f'"{query}"',
        search_size=search_size,
        resource=resource_type,
    ).replace("\n", "")

    # Send it
    response = await httpx_client.post(
        url=sparql_view_url,
        content=sparql_query_fuzzy,
        headers={
            "Content-Type": "text/plain",
            "Accept": "application/sparql-results+json",
            "Authorization": f"Bearer {token}",
        },
    )

    results = None
    if response.json()["results"]["bindings"]:
        # Define the regex pattern for br ids
        pattern = re.compile(r"http:\/\/api\.brain-map\.org\/api\/.*")

        # Dictionary to store unique objects
        objects: dict[str, str] = {}

        # Iterate over the response to extract the required information
        for entry in response.json()["results"]["bindings"]:
            # Test if the subject is of the form of a BR id or if we are looking for mtype
            subject = entry["subject"]["value"]
            if pattern.match(subject) or "braincelltype" in resource_type.lower():
                # If so, get the predicate value and see if it describes a label
                predicate = entry["predicate"]["value"]
                if predicate == "http://www.w3.org/2000/01/rdf-schema#label":
                    label = entry["object"]["value"]
                    # Append results if seen for the first time
                    if subject not in objects:
                        objects[subject] = label

        # Convert to the desired format
        results = [
            {"label": label, "id": subject} for subject, label in objects.items()
        ]
        # Output the result
        logger.info(f"Found {len(results)} objects from the fuzzy Sparql query.")
    return results


async def es_resolve(
    query: str,
    resource_type: str,
    es_view_url: str,
    token: str,
    httpx_client: AsyncClient,
    search_size: int = 1,
) -> list[dict[str, str]] | None:
    """Resolve query with the knowlegde graph using Elastic Search.

    Parameters
    ----------
    query
        Query to resolve (needs to be a brain region).
    resource_type
        Type of resource to match.
    es_view_url
        Optional url used to query the class view of the KG. Useful for backup 'match' query.
    token
        Token to access the KG.
    httpx_client
        Async Client.
    search_size
        Number of results to retrieve.

    Returns
    -------
    list[dict[str, str]] | None
        List of brain region IDs and names. None if none found.
    """
    # Match the label of the BR or mtype
    es_query = {
        "size": search_size,
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {"match": {"label": query}},
                                {"match": {"prefLabel": query}},
                                {"match": {"altLabel": query}},
                            ]
                        }
                    },
                    {"term": {"@type": "Class"}},
                ]
            }
        },
    }

    # If matching a BR, add extra regex match to ensure correct form of the id
    if "brainregion" in resource_type.lower():
        es_query["query"]["bool"]["must"].append(  # type: ignore
            {"regexp": {"@id": r"http:\/\/api\.brain-map\.org\/api\/.*"}}
        )

    # Send the actual query
    response = await httpx_client.post(
        url=es_view_url,
        headers={"Authorization": f"Bearer {token}"},
        json=es_query,
    )

    # If there are results
    if "hits" in response.json()["hits"]:
        logger.info(
            f"Found {len(response.json()['hits']['hits'])} objects from the"
            " elasticsearch backup query."
        )
        # Return all of the results correctly parsed
        return [
            {"label": br["_source"]["label"], "id": br["_source"]["@id"]}
            for br in response.json()["hits"]["hits"]
        ]
    logger.info("Didn't find brain region id. Try again next time !")
    return None


async def resolve_query(
    query: str,
    sparql_view_url: str,
    es_view_url: str,
    token: str,
    httpx_client: AsyncClient,
    resource_type: Literal["nsg:BrainRegion", "bmo:BrainCellType"] = "nsg:BrainRegion",
    search_size: int = 1,
) -> list[dict[str, str]]:
    """Resolve query using the knowlegde graph, with sparql and ES.

    Parameters
    ----------
    query
        Query to resolve (needs to be a brain region or an mtype).
    sparql_view_url
        URL to the knowledge graph.
    es_view_url
        Optional url used to query the class view of the KG. Useful for backup 'match' query.
    token
        Token to access the KG.
    httpx_client
        Async Client.
    search_size
        Number of results to retrieve.
    resource_type
        Type of resource to match.

    Returns
    -------
    list[dict[str, str]] | None
        List of brain region IDs and names. None if none found.
    """
    # Create one task per resolve method. They are ordered by 'importance'
    tasks = [
        asyncio.create_task(
            sparql_exact_resolve(
                query=query,
                resource_type=resource_type,
                sparql_view_url=sparql_view_url,
                token=token,
                httpx_client=httpx_client,
            )
        ),
        asyncio.create_task(
            sparql_fuzzy_resolve(
                query=query,
                resource_type=resource_type,
                sparql_view_url=sparql_view_url,
                token=token,
                httpx_client=httpx_client,
                search_size=search_size,
            )
        ),
        asyncio.create_task(
            es_resolve(
                query=query,
                resource_type=resource_type,
                es_view_url=es_view_url,
                token=token,
                httpx_client=httpx_client,
                search_size=search_size,
            )
        ),
    ]
    # Send them all async
    resolve_results = await asyncio.gather(*tasks)

    # Return the results of the first one that is not None (in descending importance order)
    if any(resolve_results):
        return next((result for result in resolve_results if result))
    else:
        raise ValueError(f"Couldn't find a brain region ID from the query {query}")


def escape_punctuation(text: str) -> str:
    """Escape punctuation for sparql query.

    Parameters
    ----------
    text
        Text to escape punctuation from

    Returns
    -------
        Escaped text
    """
    if not isinstance(text, str):
        raise TypeError("Only accepting strings.")
    punctuation = '-()"#/@;:<>{}`+=~|.!?,'
    for p in punctuation:
        if p in text:
            text = text.replace(p, f"\\\\{p}")
    return text
