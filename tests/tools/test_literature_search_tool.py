"""Tests Literature Search tool."""

import httpx
import pytest

from neuroagent.tools import LiteratureSearchTool
from neuroagent.tools.literature_search_tool import (
    LiteratureSearchInput,
    LiteratureSearchMetadata,
)


class TestLiteratureSearchTool:
    @pytest.mark.asyncio
    async def test_arun(self, httpx_mock):
        url = "http://fake_url?query=covid+19&retriever_k=100&use_reranker=true&reranker_k=5"
        reranker_k = 5

        fake_response = [
            {
                "article_title": "Article title",
                "article_authors": ["Author1", "Author2"],
                "paragraph": "This is the paragraph",
                "section": "fake_section",
                "article_doi": "fake_doi",
                "journal_issn": "fake_journal_issn",
            }
            for _ in range(reranker_k)
        ]

        httpx_mock.add_response(
            url=url,
            json=fake_response,
        )

        tool = LiteratureSearchTool(
            input_schema=LiteratureSearchInput(query="covid 19"),
            metadata=LiteratureSearchMetadata(
                literature_search_url=url,
                httpx_client=httpx.AsyncClient(),
                token="fake_token",
                retriever_k=100,
                use_reranker=True,
                reranker_k=reranker_k,
            ),
        )
        response = await tool.arun()
        assert isinstance(response, list)
        assert len(response) == reranker_k
        assert isinstance(response[0], dict)
