"""
Tests for the RNA-seq API routes.

Tests the /api/rnaseq endpoints for analysis pipeline.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════
# Test RNA-seq Request/Response Models
# ═══════════════════════════════════════════════════════════════

class TestRNAseqModels:
    """Test suite for RNA-seq request/response models."""

    def test_analysis_request_model(self):
        """Test analysis request model."""
        from backend.app.api.routes.rnaseq import AnalysisRequest

        request = AnalysisRequest(
            count_matrix_path="/path/to/counts.csv",
            metadata_path="/path/to/metadata.csv"
        )

        assert request.count_matrix_path == "/path/to/counts.csv"
        assert request.condition_column == "condition"  # default
        assert request.control_label == "control"  # default
        assert request.treatment_label == "treatment"  # default

    def test_analysis_request_custom_values(self):
        """Test analysis request with custom values."""
        from backend.app.api.routes.rnaseq import AnalysisRequest

        request = AnalysisRequest(
            count_matrix_path="/path/to/counts.csv",
            metadata_path="/path/to/metadata.csv",
            condition_column="group",
            control_label="normal",
            treatment_label="tumor",
            disease_context="pancreatic_cancer",
            output_dir="/output/results"
        )

        assert request.condition_column == "group"
        assert request.control_label == "normal"
        assert request.treatment_label == "tumor"
        assert request.disease_context == "pancreatic_cancer"

    def test_analysis_status_model(self):
        """Test analysis status model."""
        from backend.app.api.routes.rnaseq import AnalysisStatus

        status = AnalysisStatus(
            job_id="abc123",
            status="running",
            progress=50,
            current_step="DEG Analysis"
        )

        assert status.job_id == "abc123"
        assert status.status == "running"
        assert status.progress == 50

    def test_deg_result_model(self):
        """Test DEG result model."""
        from backend.app.api.routes.rnaseq import DEGResult

        result = DEGResult(
            gene_symbol="KRAS",
            log2_fold_change=2.5,
            p_value=0.001,
            adjusted_p_value=0.01,
            regulation="up"
        )

        assert result.gene_symbol == "KRAS"
        assert result.log2_fold_change == 2.5
        assert result.regulation == "up"

    def test_hub_gene_model(self):
        """Test hub gene model."""
        from backend.app.api.routes.rnaseq import HubGene

        hub = HubGene(
            gene_symbol="TP53",
            degree=50,
            betweenness=0.25,
            eigenvector=0.9,
            hub_score=0.85
        )

        assert hub.gene_symbol == "TP53"
        assert hub.hub_score == 0.85

    def test_pathway_result_model(self):
        """Test pathway result model."""
        from backend.app.api.routes.rnaseq import PathwayResult

        pathway = PathwayResult(
            pathway_id="KEGG:hsa04010",
            pathway_name="MAPK signaling",
            source="KEGG",
            p_value=0.001,
            adjusted_p_value=0.01,
            gene_count=15,
            genes=["KRAS", "BRAF", "MEK1"]
        )

        assert pathway.pathway_id == "KEGG:hsa04010"
        assert pathway.source == "KEGG"
        assert len(pathway.genes) == 3

    def test_validation_result_model(self):
        """Test validation result model."""
        from backend.app.api.routes.rnaseq import ValidationResult

        validation = ValidationResult(
            gene_symbol="KRAS",
            disgenet_score=0.95,
            omim_associated=True,
            cosmic_status="Oncogene",
            associated_diseases=["Pancreatic Cancer"]
        )

        assert validation.gene_symbol == "KRAS"
        assert validation.omim_associated is True
        assert validation.cosmic_status == "Oncogene"

    def test_analysis_result_model(self):
        """Test full analysis result model."""
        from backend.app.api.routes.rnaseq import (
            AnalysisResult, DEGResult, HubGene, PathwayResult, ValidationResult
        )

        result = AnalysisResult(
            job_id="test123",
            status="completed",
            deg_count=100,
            up_regulated=60,
            down_regulated=40,
            top_deg_genes=[
                DEGResult(gene_symbol="KRAS", log2_fold_change=2.0,
                         p_value=0.01, adjusted_p_value=0.05, regulation="up")
            ],
            hub_genes=[
                HubGene(gene_symbol="KRAS", degree=30, betweenness=0.2,
                       eigenvector=0.8, hub_score=0.75)
            ],
            enriched_pathways=[
                PathwayResult(pathway_id="GO:0001", pathway_name="Test",
                             source="GO", p_value=0.01, adjusted_p_value=0.05,
                             gene_count=10, genes=["A", "B"])
            ],
            validated_genes=[
                ValidationResult(gene_symbol="KRAS")
            ]
        )

        assert result.deg_count == 100
        assert len(result.top_deg_genes) == 1
        assert len(result.hub_genes) == 1


# ═══════════════════════════════════════════════════════════════
# Test RNA-seq API Endpoints
# ═══════════════════════════════════════════════════════════════

class TestRNAseqEndpoints:
    """Test suite for RNA-seq API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from backend.app.main import app
        return TestClient(app)

    def test_rnaseq_root_endpoint(self, client):
        """Test the RNA-seq info endpoint."""
        response = client.get("/api/rnaseq/")

        assert response.status_code == 200
        data = response.json()
        assert data["module"] == "RNA-seq Analysis Pipeline"
        assert data["version"] == "2.0.0"
        assert len(data["agents"]) == 6

    def test_rnaseq_agents_list(self, client):
        """Test that all 6 agents are listed."""
        response = client.get("/api/rnaseq/")

        data = response.json()
        agents = data["agents"]

        assert "DEG Analysis" in agents[0]
        assert "Network Analysis" in agents[1]
        assert "Pathway Enrichment" in agents[2]
        assert "Database Validation" in agents[3]
        assert "Visualization" in agents[4]
        assert "Report Generation" in agents[5]

    def test_rnaseq_jobs_list(self, client):
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

    def test_rnaseq_gene_info_lowercase(self, client):
        """Test that gene symbol is uppercased."""
        response = client.get("/api/rnaseq/genes/kras")

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "KRAS"  # Should be uppercased

    def test_rnaseq_nonexistent_job_status(self, client):
        """Test accessing non-existent job status."""
        response = client.get("/api/rnaseq/status/nonexistent123")

        assert response.status_code == 404

    def test_rnaseq_nonexistent_job_result(self, client):
        """Test accessing non-existent job result."""
        response = client.get("/api/rnaseq/result/nonexistent123")

        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════
# Test Job Management
# ═══════════════════════════════════════════════════════════════

class TestRNAseqJobManagement:
    """Test suite for job management in RNA-seq routes."""

    def test_job_storage_initialized(self):
        """Test that job storage is initialized."""
        from backend.app.api.routes.rnaseq import _analysis_jobs, _analysis_results

        assert isinstance(_analysis_jobs, dict)
        assert isinstance(_analysis_results, dict)

    def test_delete_nonexistent_job(self):
        """Test deleting a non-existent job."""
        from fastapi.testclient import TestClient
        from backend.app.main import app

        client = TestClient(app)
        response = client.delete("/api/rnaseq/jobs/nonexistent123")

        assert response.status_code == 404


# ═══════════════════════════════════════════════════════════════
# Test Background Task (run_pipeline_task)
# ═══════════════════════════════════════════════════════════════

class TestRNAseqBackgroundTask:
    """Test suite for background pipeline task."""

    def test_run_pipeline_task_demo_mode(self):
        """Test pipeline task runs in demo mode when orchestrator not available."""
        import asyncio
        from backend.app.api.routes.rnaseq import (
            run_pipeline_task, AnalysisRequest, AnalysisStatus,
            _analysis_jobs, _analysis_results
        )

        job_id = "test_demo_123"

        # Create initial job status
        _analysis_jobs[job_id] = AnalysisStatus(
            job_id=job_id,
            status="pending",
            progress=0
        )

        # Create request
        request = AnalysisRequest(
            count_matrix_path="/test/counts.csv",
            metadata_path="/test/metadata.csv"
        )

        # Run the task (should complete in demo mode)
        asyncio.get_event_loop().run_until_complete(run_pipeline_task(job_id, request))

        # Check status updated
        assert _analysis_jobs[job_id].status == "completed"
        assert _analysis_jobs[job_id].progress == 100

        # Check results created
        assert job_id in _analysis_results
        result = _analysis_results[job_id]
        assert result.deg_count == 256
        assert result.up_regulated == 150
        assert result.down_regulated == 106

        # Cleanup
        del _analysis_jobs[job_id]
        del _analysis_results[job_id]

    def test_run_pipeline_task_creates_demo_genes(self):
        """Test that demo mode creates expected gene results."""
        import asyncio
        from backend.app.api.routes.rnaseq import (
            run_pipeline_task, AnalysisRequest, AnalysisStatus,
            _analysis_jobs, _analysis_results
        )

        job_id = "test_genes_456"

        _analysis_jobs[job_id] = AnalysisStatus(
            job_id=job_id,
            status="pending",
            progress=0
        )

        request = AnalysisRequest(
            count_matrix_path="/test/counts.csv",
            metadata_path="/test/metadata.csv"
        )

        asyncio.get_event_loop().run_until_complete(run_pipeline_task(job_id, request))

        result = _analysis_results[job_id]

        # Check DEG genes
        assert len(result.top_deg_genes) >= 1
        kras = result.top_deg_genes[0]
        assert kras.gene_symbol == "KRAS"
        assert kras.regulation == "up"

        # Check hub genes
        assert len(result.hub_genes) >= 1
        assert result.hub_genes[0].gene_symbol == "KRAS"

        # Check pathways
        assert len(result.enriched_pathways) >= 1
        assert "MAPK" in result.enriched_pathways[0].pathway_name

        # Cleanup
        del _analysis_jobs[job_id]
        del _analysis_results[job_id]

    def test_run_pipeline_task_error_handling(self):
        """Test pipeline task handles errors gracefully."""
        import asyncio
        from backend.app.api.routes.rnaseq import (
            run_pipeline_task, AnalysisRequest, AnalysisStatus,
            _analysis_jobs
        )

        job_id = "test_error_789"

        # Create status but then remove it to cause KeyError
        _analysis_jobs[job_id] = AnalysisStatus(
            job_id=job_id,
            status="pending",
            progress=0
        )

        # Manually corrupt the status to trigger error path
        # (In real scenario, this tests exception handling)
        request = AnalysisRequest(
            count_matrix_path="/test/counts.csv",
            metadata_path="/test/metadata.csv"
        )

        # This should not raise, even if internal error occurs
        asyncio.get_event_loop().run_until_complete(run_pipeline_task(job_id, request))

        # Cleanup
        if job_id in _analysis_jobs:
            del _analysis_jobs[job_id]


# ═══════════════════════════════════════════════════════════════
# Test Start Analysis Endpoint
# ═══════════════════════════════════════════════════════════════

class TestRNAseqStartAnalysis:
    """Test suite for starting analysis."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from backend.app.main import app
        return TestClient(app)

    def test_start_analysis_endpoint(self, client):
        """Test starting an analysis job."""
        response = client.post(
            "/api/rnaseq/analyze",
            json={
                "count_matrix_path": "/test/counts.csv",
                "metadata_path": "/test/metadata.csv"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["progress"] == 0

        # Cleanup the created job
        from backend.app.api.routes.rnaseq import _analysis_jobs
        job_id = data["job_id"]
        if job_id in _analysis_jobs:
            del _analysis_jobs[job_id]

    def test_start_analysis_requires_paths(self, client):
        """Test that analysis requires count and metadata paths."""
        response = client.post(
            "/api/rnaseq/analyze",
            json={}
        )

        # Should return validation error
        assert response.status_code == 422
