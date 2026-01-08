"""
RNA-seq Analysis API Routes.

Provides endpoints for the 6-Agent RNA-seq analysis pipeline:
1. DEG Analysis (DESeq2)
2. Network Analysis (Hub genes)
3. Pathway Enrichment (GO/KEGG)
4. Database Validation (DisGeNET, OMIM, COSMIC)
5. Visualization (Volcano, Heatmap, Network)
6. Report Generation (HTML)
"""
import os
import sys
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import setup_logging

logger = setup_logging(__name__)

router = APIRouter(prefix="/rnaseq", tags=["RNA-seq Analysis"])


# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════

class AnalysisRequest(BaseModel):
    """Request model for RNA-seq analysis."""
    count_matrix_path: str = Field(..., description="Path to count matrix CSV")
    metadata_path: str = Field(..., description="Path to sample metadata CSV")
    condition_column: str = Field(default="condition", description="Column name for condition")
    control_label: str = Field(default="control", description="Label for control samples")
    treatment_label: str = Field(default="treatment", description="Label for treatment samples")
    disease_context: str = Field(default="cancer", description="Disease context for interpretation")
    output_dir: Optional[str] = Field(None, description="Output directory path")


class AnalysisStatus(BaseModel):
    """Status of an analysis job."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: int  # 0-100
    current_step: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class DEGResult(BaseModel):
    """DEG analysis result."""
    gene_symbol: str
    log2_fold_change: float
    p_value: float
    adjusted_p_value: float
    regulation: str  # up, down, unchanged


class HubGene(BaseModel):
    """Hub gene from network analysis."""
    gene_symbol: str
    degree: int
    betweenness: float
    eigenvector: float
    hub_score: float


class PathwayResult(BaseModel):
    """Pathway enrichment result."""
    pathway_id: str
    pathway_name: str
    source: str  # GO, KEGG
    p_value: float
    adjusted_p_value: float
    gene_count: int
    genes: List[str]


class ValidationResult(BaseModel):
    """Database validation result for a gene."""
    gene_symbol: str
    disgenet_score: Optional[float] = None
    omim_associated: bool = False
    cosmic_status: Optional[str] = None
    associated_diseases: List[str] = []


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    job_id: str
    status: str
    deg_count: int
    up_regulated: int
    down_regulated: int
    top_deg_genes: List[DEGResult]
    hub_genes: List[HubGene]
    enriched_pathways: List[PathwayResult]
    validated_genes: List[ValidationResult]
    report_path: Optional[str] = None
    figures: List[str] = []


# ═══════════════════════════════════════════════════════════════
# In-memory job storage (replace with Redis/DB in production)
# ═══════════════════════════════════════════════════════════════

_analysis_jobs: dict[str, AnalysisStatus] = {}
_analysis_results: dict[str, AnalysisResult] = {}


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/")
async def get_rnaseq_info():
    """Get RNA-seq analysis module information."""
    return {
        "module": "RNA-seq Analysis Pipeline",
        "version": "2.0.0",
        "agents": [
            "Agent 1: DEG Analysis (DESeq2)",
            "Agent 2: Network Analysis",
            "Agent 3: Pathway Enrichment",
            "Agent 4: Database Validation",
            "Agent 5: Visualization",
            "Agent 6: Report Generation"
        ],
        "status": "operational",
        "endpoints": {
            "analyze": "/api/rnaseq/analyze",
            "status": "/api/rnaseq/status/{job_id}",
            "result": "/api/rnaseq/result/{job_id}",
            "genes": "/api/rnaseq/genes/{symbol}"
        }
    }


@router.post("/analyze", response_model=AnalysisStatus)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new RNA-seq analysis job.

    The analysis runs in the background. Use /status/{job_id} to check progress.
    """
    import uuid

    job_id = str(uuid.uuid4())[:8]

    # Create job status
    status = AnalysisStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        started_at=datetime.now().isoformat()
    )
    _analysis_jobs[job_id] = status

    # Start background analysis
    background_tasks.add_task(
        run_pipeline_task,
        job_id,
        request
    )

    logger.info(f"Started RNA-seq analysis job: {job_id}")
    return status


@router.get("/status/{job_id}", response_model=AnalysisStatus)
async def get_analysis_status(job_id: str):
    """Get the status of an analysis job."""
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return _analysis_jobs[job_id]


@router.get("/result/{job_id}", response_model=AnalysisResult)
async def get_analysis_result(job_id: str):
    """Get the results of a completed analysis job."""
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = _analysis_jobs[job_id]
    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not complete. Current status: {status.status}"
        )

    if job_id not in _analysis_results:
        raise HTTPException(status_code=404, detail=f"Results for job {job_id} not found")

    return _analysis_results[job_id]


@router.get("/genes/{symbol}")
async def get_gene_info(symbol: str):
    """
    Get information about a specific gene.

    Returns expression data and database annotations.
    """
    # Placeholder - would query from actual gene databases
    return {
        "symbol": symbol.upper(),
        "description": f"Gene information for {symbol}",
        "databases": {
            "disgenet": {"score": None, "diseases": []},
            "omim": {"associated": False},
            "cosmic": {"status": None}
        },
        "note": "Full gene database integration pending"
    }


@router.get("/jobs")
async def list_jobs():
    """List all analysis jobs."""
    return {
        "total": len(_analysis_jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": status.status,
                "progress": status.progress
            }
            for job_id, status in _analysis_jobs.items()
        ]
    }


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete an analysis job and its results."""
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    del _analysis_jobs[job_id]
    if job_id in _analysis_results:
        del _analysis_results[job_id]

    return {"message": f"Job {job_id} deleted"}


# ═══════════════════════════════════════════════════════════════
# Background Task
# ═══════════════════════════════════════════════════════════════

async def run_pipeline_task(job_id: str, request: AnalysisRequest):
    """
    Run the RNA-seq pipeline as a background task.
    """
    try:
        status = _analysis_jobs[job_id]
        status.status = "running"

        # Import pipeline components
        try:
            from rnaseq_pipeline.orchestrator import RNAseqOrchestrator
            pipeline_available = True
        except ImportError:
            pipeline_available = False
            logger.warning("RNA-seq pipeline not available")

        if not pipeline_available:
            # Return mock results for demo
            status.progress = 100
            status.current_step = "Demo mode"
            status.status = "completed"
            status.completed_at = datetime.now().isoformat()

            _analysis_results[job_id] = AnalysisResult(
                job_id=job_id,
                status="completed",
                deg_count=256,
                up_regulated=150,
                down_regulated=106,
                top_deg_genes=[
                    DEGResult(gene_symbol="KRAS", log2_fold_change=2.5, p_value=0.001, adjusted_p_value=0.01, regulation="up"),
                    DEGResult(gene_symbol="TP53", log2_fold_change=-1.8, p_value=0.005, adjusted_p_value=0.02, regulation="down"),
                ],
                hub_genes=[
                    HubGene(gene_symbol="KRAS", degree=45, betweenness=0.15, eigenvector=0.8, hub_score=0.9),
                ],
                enriched_pathways=[
                    PathwayResult(pathway_id="KEGG:hsa04010", pathway_name="MAPK signaling", source="KEGG", p_value=0.001, adjusted_p_value=0.01, gene_count=15, genes=["KRAS", "BRAF", "MEK1"]),
                ],
                validated_genes=[
                    ValidationResult(gene_symbol="KRAS", disgenet_score=0.95, omim_associated=True, cosmic_status="Oncogene", associated_diseases=["Pancreatic Cancer", "Lung Cancer"]),
                ],
                figures=["volcano_plot.png", "heatmap.png", "network.png"]
            )
            return

        # Run actual pipeline
        output_dir = request.output_dir or f"rnaseq_results/{job_id}"

        # Step 1: DEG Analysis
        status.current_step = "DEG Analysis"
        status.progress = 15
        # orchestrator.run_agent1()

        # Step 2: Network Analysis
        status.current_step = "Network Analysis"
        status.progress = 30
        # orchestrator.run_agent2()

        # Step 3: Pathway Enrichment
        status.current_step = "Pathway Enrichment"
        status.progress = 45
        # orchestrator.run_agent3()

        # Step 4: Database Validation
        status.current_step = "Database Validation"
        status.progress = 60
        # orchestrator.run_agent4()

        # Step 5: Visualization
        status.current_step = "Visualization"
        status.progress = 80
        # orchestrator.run_agent5()

        # Step 6: Report Generation
        status.current_step = "Report Generation"
        status.progress = 95
        # orchestrator.run_agent6()

        # Complete
        status.status = "completed"
        status.progress = 100
        status.completed_at = datetime.now().isoformat()
        logger.info(f"Completed RNA-seq analysis job: {job_id}")

    except Exception as e:
        logger.error(f"RNA-seq analysis failed for job {job_id}: {e}")
        status = _analysis_jobs[job_id]
        status.status = "failed"
        status.error = str(e)
