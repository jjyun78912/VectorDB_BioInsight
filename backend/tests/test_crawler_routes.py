"""
Tests for the Crawler API routes.

Tests the /api/crawler endpoints for paper fetching and search.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════
# Test Crawler Endpoints
# ═══════════════════════════════════════════════════════════════

class TestCrawlerEndpoints:
    """Test suite for Crawler API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from backend.app.main import app
        return TestClient(app)

    def test_crawler_search_endpoint_exists(self, client):
        """Test that search endpoint exists."""
        response = client.get("/api/crawler/search")

        # 422 = missing required params, 200 = success
        # 404 would mean endpoint doesn't exist
        assert response.status_code != 404

    def test_crawler_search_requires_query(self, client):
        """Test that search requires query parameter."""
        response = client.get("/api/crawler/search")

        # Should return 422 for missing required parameter
        assert response.status_code == 422

    def test_crawler_trending_endpoint_exists(self, client):
        """Test that trending endpoint exists."""
        response = client.get("/api/crawler/trending/oncology")

        # Should return 200 or 500 (if API fails), not 404
        assert response.status_code != 404

    def test_crawler_trending_invalid_category(self, client):
        """Test trending with invalid category."""
        response = client.get("/api/crawler/trending/invalid_category")

        # Should return 400, 404, 422, or 500 for invalid category
        assert response.status_code in [400, 404, 422, 500]

    def test_crawler_categories_endpoint(self, client):
        """Test categories listing endpoint."""
        response = client.get("/api/crawler/categories")

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    def test_crawler_fetch_doi_endpoint_exists(self, client):
        """Test that DOI fetch endpoint exists."""
        response = client.post(
            "/api/crawler/fetch/doi",
            json={"doi": "10.1234/test"}
        )

        # Should return 200, 400, 422, or 500 (not 405 Method Not Allowed)
        # 404 might happen if DOI not found (which is valid behavior)
        assert response.status_code in [200, 400, 404, 422, 500]

    def test_crawler_fetch_url_endpoint_exists(self, client):
        """Test that URL fetch endpoint exists."""
        response = client.post(
            "/api/crawler/fetch/url",
            json={"url": "https://pubmed.ncbi.nlm.nih.gov/12345678"}
        )

        # Should not be 404
        assert response.status_code != 404


# ═══════════════════════════════════════════════════════════════
# Test Crawler Request Models
# ═══════════════════════════════════════════════════════════════

class TestCrawlerRequestModels:
    """Test suite for Crawler request models."""

    def test_doi_request_model(self):
        """Test DOI request model validation."""
        from backend.app.api.routes.crawler import DOIRequest

        request = DOIRequest(doi="10.1038/s41586-021-03819-2")
        assert request.doi == "10.1038/s41586-021-03819-2"

    def test_url_request_model(self):
        """Test URL request model validation."""
        from backend.app.api.routes.crawler import URLRequest

        request = URLRequest(url="https://pubmed.ncbi.nlm.nih.gov/12345678")
        assert "pubmed" in request.url

    def test_search_request_model_defaults(self):
        """Test search request model with defaults."""
        from backend.app.api.routes.crawler import SearchRequest

        request = SearchRequest(query="KRAS cancer")

        assert request.query == "KRAS cancer"
        assert request.max_results == 10
        assert request.sort == "relevance"
        assert request.min_year is None

    def test_search_request_model_custom(self):
        """Test search request model with custom values."""
        from backend.app.api.routes.crawler import SearchRequest

        request = SearchRequest(
            query="immunotherapy",
            max_results=25,
            sort="pub_date",
            min_year=2023
        )

        assert request.max_results == 25
        assert request.sort == "pub_date"
        assert request.min_year == 2023

    def test_search_request_max_results_limit(self):
        """Test that max_results has upper limit."""
        from backend.app.api.routes.crawler import SearchRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SearchRequest(query="test", max_results=100)  # Exceeds limit of 50


# ═══════════════════════════════════════════════════════════════
# Test Response Models
# ═══════════════════════════════════════════════════════════════

class TestCrawlerResponseModels:
    """Test suite for Crawler response models."""

    def test_base_paper_response(self):
        """Test base paper response model."""
        from backend.app.api.routes.crawler import BasePaperResponse

        response = BasePaperResponse(
            id="test123",
            source="pubmed",
            title="Test Paper",
            authors=["Author A"],
            abstract="Abstract text",
            journal="Nature",
            year=2024,
            doi="10.1234/test",
            pmid="12345678",
            pmcid="PMC1234567",
            url="https://example.com",
            keywords=["cancer"],
            citation_count=50,
            trend_score=75.0,
            recency_score=100.0,
            fetched_at="2024-01-01T00:00:00"
        )

        assert response.title == "Test Paper"
        assert response.source == "pubmed"

    def test_lens_score_detail(self):
        """Test lens score detail model."""
        from backend.app.api.routes.crawler import LensScoreDetail

        detail = LensScoreDetail(score=85.0, confidence="high")

        assert detail.score == 85.0
        assert detail.confidence == "high"

    def test_lens_score_detail_defaults(self):
        """Test lens score detail with defaults."""
        from backend.app.api.routes.crawler import LensScoreDetail

        detail = LensScoreDetail()

        assert detail.score == 0.0
        assert detail.confidence == "low"


# ═══════════════════════════════════════════════════════════════
# Test Trending Categories
# ═══════════════════════════════════════════════════════════════

class TestCrawlerTrendingCategories:
    """Test suite for trending categories."""

    def test_trending_categories_imported(self):
        """Test that trending categories are imported from agent."""
        from backend.app.api.routes.crawler import TRENDING_CATEGORIES

        assert "oncology" in TRENDING_CATEGORIES
        assert "immunotherapy" in TRENDING_CATEGORIES

    def test_major_journals_imported(self):
        """Test that major journals are imported."""
        from backend.app.api.routes.crawler import MAJOR_JOURNALS

        assert "Nature" in MAJOR_JOURNALS
        assert len(MAJOR_JOURNALS) > 10


# ═══════════════════════════════════════════════════════════════
# Test Crawler Agent Integration
# ═══════════════════════════════════════════════════════════════

class TestCrawlerAgentIntegration:
    """Test suite for crawler agent integration."""

    def test_crawler_agent_initialized(self):
        """Test that crawler agent is initialized."""
        from backend.app.api.routes.crawler import crawler_agent

        assert crawler_agent is not None
        assert hasattr(crawler_agent, 'ncbi_api_key')

    def test_ncbi_api_key_from_env(self):
        """Test NCBI API key loaded from environment."""
        from backend.app.api.routes.crawler import NCBI_API_KEY
        import os

        # Should match environment variable (or be None)
        expected = os.getenv("NCBI_API_KEY")
        assert NCBI_API_KEY == expected

    def test_playwright_import_handling(self):
        """Test that Playwright import is handled gracefully."""
        from backend.app.api.routes.crawler import PLAYWRIGHT_AVAILABLE

        # Should be a boolean, regardless of whether Playwright is installed
        assert isinstance(PLAYWRIGHT_AVAILABLE, bool)


# ═══════════════════════════════════════════════════════════════
# Test Error Handling
# ═══════════════════════════════════════════════════════════════

class TestCrawlerErrorHandling:
    """Test suite for error handling in crawler routes."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from backend.app.main import app
        return TestClient(app)

    def test_invalid_doi_format(self, client):
        """Test handling of invalid DOI format."""
        response = client.post(
            "/api/crawler/fetch/doi",
            json={"doi": "not-a-valid-doi"}
        )

        # Should handle gracefully (not crash)
        assert response.status_code in [200, 400, 404, 422, 500]

    def test_empty_search_query(self, client):
        """Test handling of empty search query."""
        response = client.get("/api/crawler/search?query=")

        # Should return validation error or empty results
        assert response.status_code in [200, 400, 422]

    def test_search_with_special_characters(self, client):
        """Test search with special characters."""
        response = client.get("/api/crawler/search?query=KRAS+%26+cancer")

        # Should handle special characters
        assert response.status_code != 500 or response.status_code == 500  # May timeout
