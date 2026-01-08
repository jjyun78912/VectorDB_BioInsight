"""
Tests for the FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "BioInsight API"
        assert data["version"] == "1.0.0"

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_api_docs_available(self, client):
        """Test that API docs are available."""
        response = client.get("/docs")

        # Should redirect or return docs
        assert response.status_code in [200, 307]

    def test_openapi_schema(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


class TestSearchEndpoints:
    """Test suite for search-related endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    def test_search_endpoint_exists(self, client):
        """Test that search endpoint exists."""
        # Should return 422 (validation error) or 200, not 404
        response = client.get("/api/search")

        # 422 means endpoint exists but missing required params
        # 404 would mean endpoint doesn't exist
        assert response.status_code != 404


class TestCORSConfiguration:
    """Test suite for CORS configuration."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present for allowed origins."""
        response = client.options(
            "/api/search",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET"
            }
        )

        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or \
               response.status_code == 200


class TestRNAseqEndpoints:
    """Test suite for RNA-seq analysis endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    def test_rnaseq_info_endpoint(self, client):
        """Test the RNA-seq info endpoint."""
        response = client.get("/api/rnaseq/")

        assert response.status_code == 200
        data = response.json()
        assert data["module"] == "RNA-seq Analysis Pipeline"
        assert "agents" in data
        assert len(data["agents"]) == 6

    def test_rnaseq_jobs_endpoint(self, client):
        """Test the jobs list endpoint."""
        response = client.get("/api/rnaseq/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "jobs" in data

    def test_rnaseq_gene_info(self, client):
        """Test the gene info endpoint."""
        response = client.get("/api/rnaseq/genes/KRAS")

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "KRAS"
        assert "databases" in data

    def test_rnaseq_nonexistent_job(self, client):
        """Test accessing a non-existent job."""
        response = client.get("/api/rnaseq/status/nonexistent")

        assert response.status_code == 404
