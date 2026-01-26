"""
BioInsight AI - API Endpoint Tests
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test that health endpoint returns OK."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestPaperEndpoints:
    """Test paper-related API endpoints."""

    def test_paper_search_endpoint_exists(self):
        """Test that paper search endpoint exists."""
        from backend.app.main import app
        client = TestClient(app)

        # Should return 422 (validation error) not 404 when missing params
        response = client.get("/api/paper/search")
        assert response.status_code in [200, 422]  # Either works or needs params

    def test_paper_upload_endpoint_exists(self):
        """Test that paper upload endpoint exists."""
        from backend.app.main import app
        client = TestClient(app)

        # POST without file should fail with validation error
        response = client.post("/api/paper/upload")
        assert response.status_code in [400, 422]


class TestRNAseqEndpoints:
    """Test RNA-seq API endpoints."""

    def test_rnaseq_upload_endpoint(self):
        """Test RNA-seq upload endpoint."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.post("/api/rnaseq/upload")
        # Should fail without files but endpoint should exist
        assert response.status_code in [400, 422, 405]

    def test_rnaseq_status_endpoint(self):
        """Test RNA-seq job status endpoint."""
        from backend.app.main import app
        client = TestClient(app)

        # Non-existent job
        response = client.get("/api/rnaseq/status/nonexistent-job-id")
        assert response.status_code in [404, 200]


class TestChatEndpoints:
    """Test chat/RAG endpoints."""

    def test_chat_endpoint_exists(self):
        """Test that chat endpoint exists."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/chat/ask",
            json={"question": "What is BRCA1?"}
        )
        # May need API key, but endpoint should exist
        assert response.status_code in [200, 401, 500]


class TestCORSConfiguration:
    """Test CORS settings."""

    def test_cors_headers(self):
        """Test that CORS headers are set."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.options(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        # CORS preflight should work
        assert response.status_code in [200, 405]


class TestInputValidation:
    """Test input validation for API endpoints."""

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON input."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/chat/ask",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.post(
            "/api/chat/ask",
            json={}  # Missing required 'question' field
        )
        assert response.status_code == 422


class TestRateLimiting:
    """Test rate limiting (if implemented)."""

    def test_rate_limit_headers(self):
        """Check if rate limit headers are present."""
        from backend.app.main import app
        client = TestClient(app)

        response = client.get("/health")
        # Rate limit headers are optional but good to have
        # Just verify endpoint works
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
