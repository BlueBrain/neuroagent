"""Tests Literature Search tool."""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from neuroagent.tools import LiteratureSearchTool
from neuroagent.tools.literature_search_tool import ParagraphMetadata


class TestLiteratureSearchTool:
    @pytest.mark.asyncio
    async def test_arun(self):
        url = "http://fake_url"
        reranker_k = 5

        client = httpx.AsyncClient()
        client.get = AsyncMock()
        response = Mock()
        response.status_code = 200
        client.get.return_value = response
        response.json.return_value = [
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

        tool = LiteratureSearchTool(
            metadata={
                "url": url,
                "httpx_client": client,
                "token": "fake_token",
                "retriever_k": 100,
                "use_reranker": True,
                "reranker_k": 5,
            }
        )

        response = await tool._arun(query="covid 19")
        assert isinstance(response, list)
        assert len(response) == reranker_k
        assert isinstance(response[0], ParagraphMetadata)

    def test_run(self):
        url = "http://fake_url"
        reranker_k = 5

        client = httpx.Client()
        client.get = Mock()
        response = Mock()
        client.get.return_value = response
        response.json.return_value = [
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

        tool = LiteratureSearchTool(
            metadata={
                "url": url,
                "httpx_client": client,
                "token": "fake_token",
                "retriever_k": 100,
                "use_reranker": True,
                "reranker_k": 5,
            }
        )

        response = tool._run(query="covid 19")
        assert isinstance(response, list)
        assert len(response) == reranker_k
        assert isinstance(response[0], ParagraphMetadata)
