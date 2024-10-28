"""Literature Search tool."""

import logging
from typing import Any, ClassVar

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from swarm_copy.tools.base_tool import BaseMetadata, BaseTool, BaseToolOutput

logger = logging.getLogger(__name__)


class LiteratureSearchInput(BaseModel):
    """Inputs of the literature search API."""

    query: str = Field(
        description=(
            "Query to match against the text of paragraphs coming from scientific"
            " articles. The matching is done using the bm25 algorithm, so the query"
            " should be based on keywords to ensure maximal efficiency."
        )
    )


class LiteratureSearchMetadata(BaseMetadata):
    """Metadata class for LiteratureSearchTool."""

    knowledge_graph_url: str
    literature_search_url: str
    token: str
    httpx_client: Any
    retriever_k: int
    reranker_k: int
    use_reranker: bool


class ParagraphMetadata(BaseToolOutput, extra="ignore"):
    """Metadata for an article."""

    article_title: str
    article_authors: list[str]
    paragraph: str
    section: str | None = None
    article_doi: str | None = None
    journal_issn: str | None = None


class LiteratureSearchTool(BaseTool):
    """Class defining the Literature Search logic."""

    name: ClassVar[str] = "literature-search-tool"
    description: ClassVar[
        str
    ] = """Searches the scientific literature. The tool should be used to gather general scientific knowledge. It is best suited for questions about neuroscience and medicine that are not about morphologies.
    It returns a list of paragraphs fron scientific papers that match the query (in the sense of the bm25 algorithm), alongside with the metadata of the articles they were extracted from, such as:
    - title
    - authors
    - paragraph_text
    - section
    - article_doi
    - journal_issn"""
    input_schema: LiteratureSearchInput
    metadata: LiteratureSearchMetadata

    def _run(self) -> list[ParagraphMetadata]:
        """Search the scientific literature and returns citations."""
        # Prepare the request's body
        req_body = {
            "query": self.input_schema.query,
            "retriever_k": self.metadata.retriever_k,
            "use_reranker": self.metadata.use_reranker,
            "reranker_k": self.metadata.reranker_k,
        }

        # Send the request
        return self._process_output(
            self.metadata.httpx_client.get(
                self.metadata.literature_search_url,
                headers={"Authorization": f"Bearer {self.metadata.token}"},
                json=req_body,
                timeout=None,
            ).json()
        )

    async def _arun(self, query: str) -> list[ParagraphMetadata] | str:
        """Async search the scientific literature and returns citations.

        Parameters
        ----------
        query
            Query to send to the literature search backend

        Returns
        -------
            List of paragraphs and their metadata
        """
        try:
            logger.info(f"Entering literature search tool. Inputs: {query=}")

            # Prepare the request's body
            req_body = {
                "query": query,
                "retriever_k": self.metadata.retriever_k,
                "use_reranker": self.metadata.use_reranker,
                "reranker_k": self.metadata.reranker_k,
            }

            # Send the request
            response = await self.metadata.httpx_client.get(
                self.metadata.literature_search_url,
                headers={"Authorization": f"Bearer {self.metadata.token}"},
                params=req_body,
                timeout=None,
            )

            return self._process_output(response.json())
        except Exception as e:
            raise ToolException(str(e), self.name)

    @staticmethod
    def _process_output(output: list[dict[str, Any]]) -> list[ParagraphMetadata]:
        """Process output."""
        paragraphs_metadata = [
            ParagraphMetadata(
                article_title=paragraph["article_title"],
                article_authors=paragraph["article_authors"],
                paragraph=paragraph["paragraph"],
                section=paragraph["section"],
                article_doi=paragraph["article_doi"],
                journal_issn=paragraph["journal_issn"],
            )
            for paragraph in output
        ]
        return paragraphs_metadata
