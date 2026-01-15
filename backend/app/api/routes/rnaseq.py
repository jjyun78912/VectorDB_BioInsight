"""
RNA-seq Analysis API Routes.

Provides endpoints for the 6-Agent RNA-seq analysis pipeline:
1. DEG Analysis (DESeq2)
2. Network Analysis (Hub genes)
3. Pathway Enrichment (GO/KEGG)
4. Database Validation (DisGeNET, OMIM, COSMIC)
5. Visualization (Volcano, Heatmap, Network)
6. Report Generation (HTML)

Features:
- File upload for count matrix and metadata
- SSE (Server-Sent Events) for real-time progress streaming
- Background task execution for long-running analyses
"""
import os
import sys
import json
import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
from threading import Thread
from queue import Queue

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import setup_logging

logger = setup_logging(__name__)

router = APIRouter(prefix="/rnaseq", tags=["RNA-seq Analysis"])

# Upload directory for RNA-seq data
UPLOAD_DIR = PROJECT_ROOT / "data" / "rnaseq_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
_job_queues: dict[str, Queue] = {}  # For SSE streaming

# Agent info for frontend display
AGENT_INFO = {
    "agent1_deg": {
        "name": "DEG Analysis",
        "description": "DESeq2로 차등 발현 유전자 분석",
        "icon": "activity",
        "order": 1
    },
    "agent2_network": {
        "name": "Network Analysis",
        "description": "유전자 네트워크 및 Hub gene 탐지",
        "icon": "share-2",
        "order": 2
    },
    "agent3_pathway": {
        "name": "Pathway Enrichment",
        "description": "GO/KEGG 경로 농축 분석",
        "icon": "git-branch",
        "order": 3
    },
    "agent4_validation": {
        "name": "DB Validation",
        "description": "DisGeNET, OMIM, COSMIC 검증",
        "icon": "database",
        "order": 4
    },
    "agent5_visualization": {
        "name": "Visualization",
        "description": "Volcano plot, Heatmap, Network 시각화",
        "icon": "bar-chart-2",
        "order": 5
    },
    "agent6_report": {
        "name": "Report Generation",
        "description": "HTML 보고서 생성",
        "icon": "file-text",
        "order": 6
    }
}


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/")
async def get_rnaseq_info():
    """Get RNA-seq analysis module information."""
    return {
        "module": "RNA-seq Analysis Pipeline",
        "version": "2.0.0",
        "agents": AGENT_INFO,
        "status": "operational",
        "endpoints": {
            "upload": "/api/rnaseq/upload",
            "analyze": "/api/rnaseq/analyze",
            "stream": "/api/rnaseq/stream/{job_id}",
            "status": "/api/rnaseq/status/{job_id}",
            "result": "/api/rnaseq/result/{job_id}",
            "report": "/api/rnaseq/report/{job_id}",
            "genes": "/api/rnaseq/genes/{symbol}"
        }
    }


@router.get("/agents")
async def get_agents_info():
    """Get detailed information about pipeline agents."""
    return {
        "agents": AGENT_INFO,
        "total": len(AGENT_INFO),
        "order": list(AGENT_INFO.keys())
    }


# ═══════════════════════════════════════════════════════════════
# File Upload Endpoints
# ═══════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    """Response for file upload."""
    job_id: str
    message: str
    files_received: List[str]
    input_dir: str


class SampleInfo(BaseModel):
    """Sample information from count matrix."""
    sample_id: str
    condition: str = "unknown"


class CountMatrixPreviewResponse(BaseModel):
    """Response for count matrix preview."""
    job_id: str
    samples: List[SampleInfo]
    gene_count: int
    detected_conditions: Dict[str, List[str]]
    suggested_treatment: str
    suggested_control: str


@router.post("/preview-samples")
async def preview_count_matrix_samples(
    count_matrix: UploadFile = File(..., description="Count matrix CSV/TSV file")
):
    """
    Preview samples from count matrix and auto-detect conditions.

    Analyzes column names to detect tumor/normal, case/control patterns.
    Returns sample list with suggested condition assignments.
    """
    import pandas as pd
    import io
    import re

    try:
        content = await count_matrix.read()

        # Detect separator
        first_line = content.decode('utf-8').split('\n')[0]
        sep = '\t' if '\t' in first_line else ','

        # Read only header (first row)
        df = pd.read_csv(io.BytesIO(content), sep=sep, nrows=5)

        # Get sample columns (exclude gene_id column)
        gene_col_patterns = ['gene_id', 'gene', 'ensembl', 'symbol', 'name', 'id']
        sample_columns = []
        gene_col = None

        for col in df.columns:
            col_lower = col.lower()
            is_gene_col = any(p in col_lower for p in gene_col_patterns)
            if is_gene_col and gene_col is None:
                gene_col = col
            else:
                sample_columns.append(col)

        # If no gene column detected, assume first column is gene_id
        if gene_col is None and len(df.columns) > 0:
            gene_col = df.columns[0]
            sample_columns = list(df.columns[1:])

        # Count total genes
        df_full = pd.read_csv(io.BytesIO(content), sep=sep, usecols=[gene_col] if gene_col else [0])
        gene_count = len(df_full)

        # Auto-detect conditions from column names
        treatment_patterns = [
            r'tumor', r'cancer', r'case', r'disease', r'patient', r'treated',
            r'_t[0-9]*$', r'_t$', r'^t[0-9]+', r'primary', r'metastatic'
        ]
        control_patterns = [
            r'normal', r'control', r'healthy', r'untreated', r'reference',
            r'_n[0-9]*$', r'_n$', r'^n[0-9]+', r'adjacent', r'baseline'
        ]

        samples = []
        detected_conditions = {"treatment": [], "control": [], "unknown": []}

        for sample in sample_columns:
            sample_lower = sample.lower()
            condition = "unknown"

            # Check treatment patterns
            for pattern in treatment_patterns:
                if re.search(pattern, sample_lower):
                    condition = "treatment"
                    break

            # Check control patterns if not treatment
            if condition == "unknown":
                for pattern in control_patterns:
                    if re.search(pattern, sample_lower):
                        condition = "control"
                        break

            samples.append(SampleInfo(sample_id=sample, condition=condition))
            detected_conditions[condition].append(sample)

        # Suggest labels based on detected patterns
        suggested_treatment = "tumor" if detected_conditions["treatment"] else "treatment"
        suggested_control = "normal" if detected_conditions["control"] else "control"

        # Generate temporary job_id for this preview
        job_id = str(uuid.uuid4())[:8]

        return CountMatrixPreviewResponse(
            job_id=job_id,
            samples=samples,
            gene_count=gene_count,
            detected_conditions=detected_conditions,
            suggested_treatment=suggested_treatment,
            suggested_control=suggested_control
        )

    except Exception as e:
        logger.error(f"Preview failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse count matrix: {str(e)}")


@router.post("/upload-with-auto-metadata", response_model=UploadResponse)
async def upload_with_auto_metadata(
    count_matrix: UploadFile = File(..., description="Count matrix CSV file"),
    sample_conditions: str = Form(..., description="JSON: {sample_id: condition}"),
    cancer_type: str = Form(default="unknown", description="Cancer type for analysis"),
    study_name: str = Form(default="", description="Study name or description"),
    treatment_label: str = Form(default="tumor", description="Treatment/case label"),
    control_label: str = Form(default="normal", description="Control label")
):
    """
    Upload count matrix and auto-generate metadata from user-selected conditions.

    sample_conditions: JSON mapping of sample_id to condition (treatment/control)
    Example: {"tumor_1": "treatment", "tumor_2": "treatment", "normal_1": "control"}
    """
    import pandas as pd
    import io

    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Parse sample conditions
        conditions_map = json.loads(sample_conditions)

        # Save count matrix
        count_content = await count_matrix.read()
        count_path = job_dir / "count_matrix.csv"

        # Detect separator and re-save as CSV if needed
        first_line = count_content.decode('utf-8').split('\n')[0]
        sep = '\t' if '\t' in first_line else ','

        df = pd.read_csv(io.BytesIO(count_content), sep=sep)
        df.to_csv(count_path, index=False)

        # Generate metadata from conditions map
        metadata_rows = []
        for sample_id, condition_type in conditions_map.items():
            # Map "treatment"/"control" to actual labels
            if condition_type == "treatment":
                condition = treatment_label
            elif condition_type == "control":
                condition = control_label
            else:
                condition = condition_type

            metadata_rows.append({
                "sample_id": sample_id,
                "condition": condition
            })

        # Save auto-generated metadata
        metadata_df = pd.DataFrame(metadata_rows)
        meta_path = job_dir / "metadata.csv"
        metadata_df.to_csv(meta_path, index=False)

        # Save config
        config = {
            "cancer_type": cancer_type,
            "study_name": study_name or f"RNA-seq Analysis {job_id}",
            "condition_column": "condition",
            "contrast": [treatment_label, control_label],
            "auto_metadata": True,
            "uploaded_at": datetime.now().isoformat()
        }
        config_path = job_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Files uploaded with auto-metadata for job {job_id}")

        return UploadResponse(
            job_id=job_id,
            message="Files uploaded with auto-generated metadata",
            files_received=[count_matrix.filename, "metadata.csv (auto-generated)"],
            input_dir=str(job_dir)
        )

    except json.JSONDecodeError as e:
        if job_dir.exists():
            shutil.rmtree(job_dir)
        raise HTTPException(status_code=400, detail=f"Invalid sample_conditions JSON: {str(e)}")
    except Exception as e:
        if job_dir.exists():
            shutil.rmtree(job_dir)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload", response_model=UploadResponse)
async def upload_rnaseq_files(
    count_matrix: UploadFile = File(..., description="Count matrix CSV file"),
    metadata: UploadFile = File(..., description="Sample metadata CSV file"),
    cancer_type: str = Form(default="unknown", description="Cancer type for analysis"),
    study_name: str = Form(default="", description="Study name or description"),
    condition_column: str = Form(default="condition", description="Condition column in metadata"),
    treatment_label: str = Form(default="tumor", description="Treatment/case label"),
    control_label: str = Form(default="normal", description="Control label")
):
    """
    Upload RNA-seq count matrix and metadata files.

    Creates a unique job directory and saves the uploaded files.
    Returns job_id to use for starting analysis.

    Expected file formats:
    - count_matrix.csv: gene_id column + sample columns with raw counts
    - metadata.csv: sample_id column + condition column
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save count matrix
        count_path = job_dir / "count_matrix.csv"
        with open(count_path, "wb") as f:
            content = await count_matrix.read()
            f.write(content)

        # Save metadata
        meta_path = job_dir / "metadata.csv"
        with open(meta_path, "wb") as f:
            content = await metadata.read()
            f.write(content)

        # Save config
        config = {
            "cancer_type": cancer_type,
            "study_name": study_name or f"RNA-seq Analysis {job_id}",
            "condition_column": condition_column,
            "contrast": [treatment_label, control_label],
            "uploaded_at": datetime.now().isoformat()
        }
        config_path = job_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Files uploaded for job {job_id}: {count_matrix.filename}, {metadata.filename}")

        return UploadResponse(
            job_id=job_id,
            message="Files uploaded successfully",
            files_received=[count_matrix.filename, metadata.filename],
            input_dir=str(job_dir)
        )

    except Exception as e:
        # Cleanup on error
        if job_dir.exists():
            shutil.rmtree(job_dir)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


class StartAnalysisRequest(BaseModel):
    """Request to start analysis for an uploaded job."""
    job_id: str = Field(..., description="Job ID from upload")
    cancer_type: Optional[str] = Field(None, description="Override cancer type")
    study_name: Optional[str] = Field(None, description="Override study name")


@router.post("/start/{job_id}", response_model=AnalysisStatus)
async def start_analysis_from_upload(
    job_id: str,
    background_tasks: BackgroundTasks,
    cancer_type: Optional[str] = None,
    study_name: Optional[str] = None
):
    """
    Start RNA-seq analysis for previously uploaded files.

    Use the job_id returned from /upload endpoint.
    Progress can be monitored via /stream/{job_id} SSE endpoint.
    """
    job_dir = UPLOAD_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found. Please upload files first.")

    # Load config
    config_path = job_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Apply overrides
    if cancer_type:
        config["cancer_type"] = cancer_type
    if study_name:
        config["study_name"] = study_name

    # Create job status
    status = AnalysisStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        started_at=datetime.now().isoformat()
    )
    _analysis_jobs[job_id] = status

    # Create message queue for SSE
    _job_queues[job_id] = Queue()

    # Start background analysis
    background_tasks.add_task(
        run_pipeline_with_streaming,
        job_id,
        job_dir,
        config
    )

    logger.info(f"Started RNA-seq analysis job: {job_id}")
    return status


@router.post("/analyze", response_model=AnalysisStatus)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new RNA-seq analysis job (legacy endpoint).

    The analysis runs in the background. Use /status/{job_id} to check progress.
    """
    job_id = str(uuid.uuid4())[:8]

    # Create job status
    status = AnalysisStatus(
        job_id=job_id,
        status="pending",
        progress=0,
        started_at=datetime.now().isoformat()
    )
    _analysis_jobs[job_id] = status

    # Create message queue for SSE
    _job_queues[job_id] = Queue()

    # Start background analysis
    background_tasks.add_task(
        run_pipeline_task,
        job_id,
        request
    )

    logger.info(f"Started RNA-seq analysis job: {job_id}")
    return status


# ═══════════════════════════════════════════════════════════════
# SSE Streaming Endpoint
# ═══════════════════════════════════════════════════════════════

async def event_generator(job_id: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for a job."""
    if job_id not in _job_queues:
        yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
        return

    queue = _job_queues[job_id]

    while True:
        try:
            # Non-blocking check with timeout
            await asyncio.sleep(0.1)

            # Check if there are messages
            if not queue.empty():
                message = queue.get_nowait()

                if message is None:  # End signal
                    yield f"data: {json.dumps({'type': 'complete', 'message': 'Pipeline finished'})}\n\n"
                    break

                yield f"data: {json.dumps(message)}\n\n"

            # Check job status
            if job_id in _analysis_jobs:
                status = _analysis_jobs[job_id]
                if status.status in ["completed", "failed"]:
                    final_message = {
                        "type": "final",
                        "status": status.status,
                        "progress": status.progress,
                        "error": status.error
                    }
                    yield f"data: {json.dumps(final_message)}\n\n"
                    break

        except Exception as e:
            logger.error(f"SSE error for job {job_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            break


@router.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    """
    Server-Sent Events endpoint for real-time pipeline progress.

    Connect to this endpoint to receive live updates as each agent runs.
    Events include:
    - agent_start: Agent is starting
    - agent_progress: Progress update within agent
    - agent_complete: Agent finished
    - agent_error: Agent failed
    - complete: Pipeline finished
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return StreamingResponse(
        event_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
# Report Serving Endpoints
# ═══════════════════════════════════════════════════════════════

@router.get("/report/{job_id}")
async def get_report(job_id: str):
    """
    Get the HTML report for a completed analysis.

    Returns the report.html file as a downloadable response.
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = _analysis_jobs[job_id]
    if status.status not in ["completed", "completed_with_errors"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not complete. Current status: {status.status}"
        )

    # Get run directory from status (stored in current_step after completion)
    run_dir = status.current_step
    if not run_dir:
        raise HTTPException(status_code=404, detail="Report path not found")

    report_path = Path(run_dir) / "report.html"
    if not report_path.exists():
        # Try accumulated directory
        report_path = Path(run_dir) / "accumulated" / "report.html"

    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename=f"rnaseq_report_{job_id}.html"
    )


@router.get("/report/{job_id}/figures/{filename}")
async def get_report_figure(job_id: str, filename: str):
    """
    Get a figure from the analysis results.

    Supports PNG, SVG, and HTML (interactive) figures.
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = _analysis_jobs[job_id]
    run_dir = status.current_step
    if not run_dir:
        raise HTTPException(status_code=404, detail="Report path not found")

    # Look for figure in multiple locations
    possible_paths = [
        Path(run_dir) / "figures" / filename,
        Path(run_dir) / "accumulated" / "figures" / filename,
        Path(run_dir) / filename,
        Path(run_dir) / "accumulated" / filename,
    ]

    figure_path = None
    for path in possible_paths:
        if path.exists():
            figure_path = path
            break

    if not figure_path:
        raise HTTPException(status_code=404, detail=f"Figure {filename} not found")

    # Determine media type
    suffix = figure_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".html": "text/html",
        ".json": "application/json"
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(path=figure_path, media_type=media_type)


@router.get("/report/{job_id}/data/{filename}")
async def get_report_data(job_id: str, filename: str):
    """
    Get a data file (CSV, JSON) from the analysis results.
    """
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = _analysis_jobs[job_id]
    run_dir = status.current_step
    if not run_dir:
        raise HTTPException(status_code=404, detail="Report path not found")

    possible_paths = [
        Path(run_dir) / filename,
        Path(run_dir) / "accumulated" / filename,
    ]

    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if not data_path:
        raise HTTPException(status_code=404, detail=f"Data file {filename} not found")

    suffix = data_path.suffix.lower()
    media_types = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".txt": "text/plain"
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(path=data_path, media_type=media_type)


# ═══════════════════════════════════════════════════════════════
# Network Graph API (for 3D visualization)
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import json

# Results directory path
RESULTS_BASE_DIR = PROJECT_ROOT / "rnaseq_test_results"


class NetworkNode(BaseModel):
    """Node for 3D network visualization."""
    id: str
    gene_symbol: Optional[str] = None
    log2FC: float = 0.0
    padj: float = 1.0
    direction: str = "unchanged"
    is_hub: bool = False
    hub_score: float = 0.0
    degree: int = 0
    betweenness: float = 0.0
    eigenvector: float = 0.0
    pathway_count: int = 0
    db_matched: bool = False
    db_sources: List[str] = []
    confidence: str = "low"
    tags: List[str] = []


class NetworkEdge(BaseModel):
    """Edge for 3D network visualization."""
    source: str
    target: str
    correlation: float
    abs_correlation: float


class NetworkGraphData(BaseModel):
    """Complete network graph data for 3D visualization."""
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    stats: dict


class AnalysisInfo(BaseModel):
    """Analysis run information."""
    id: str
    name: str
    path: str
    created_at: str
    node_count: int = 0
    edge_count: int = 0
    hub_count: int = 0


@router.get("/analyses", response_model=List[AnalysisInfo])
async def list_analyses():
    """
    List all available RNA-seq analysis results.

    Scans the results directory for analysis runs that have network data.
    """
    analyses = []

    if not RESULTS_BASE_DIR.exists():
        return analyses

    # Scan for analysis directories
    for analysis_dir in RESULTS_BASE_DIR.iterdir():
        if not analysis_dir.is_dir():
            continue

        # Look for run directories (e.g., run_20260109_115803)
        run_dirs = list(analysis_dir.glob("run_*"))
        if run_dirs:
            for run_dir in run_dirs:
                accumulated_dir = run_dir / "accumulated"
                if accumulated_dir.exists():
                    network_nodes = accumulated_dir / "network_nodes.csv"
                    network_edges = accumulated_dir / "network_edges.csv"

                    if network_nodes.exists() and network_edges.exists():
                        # Count nodes and edges
                        try:
                            nodes_df = pd.read_csv(network_nodes)
                            edges_df = pd.read_csv(network_edges)
                            hub_count = nodes_df['is_hub'].sum() if 'is_hub' in nodes_df.columns else 0

                            analyses.append(AnalysisInfo(
                                id=f"{analysis_dir.name}/{run_dir.name}",
                                name=f"{analysis_dir.name} ({run_dir.name[-6:]})",
                                path=str(accumulated_dir),
                                created_at=run_dir.name.replace("run_", ""),
                                node_count=len(nodes_df),
                                edge_count=len(edges_df),
                                hub_count=int(hub_count)
                            ))
                        except Exception as e:
                            logger.warning(f"Error reading {accumulated_dir}: {e}")

        # Also check for direct results (older format)
        if (analysis_dir / "hub_genes.csv").exists():
            try:
                hub_df = pd.read_csv(analysis_dir / "hub_genes.csv")
                analyses.append(AnalysisInfo(
                    id=analysis_dir.name,
                    name=analysis_dir.name,
                    path=str(analysis_dir),
                    created_at="unknown",
                    node_count=len(hub_df),
                    edge_count=0,
                    hub_count=len(hub_df)
                ))
            except Exception as e:
                logger.warning(f"Error reading {analysis_dir}: {e}")

    return analyses


@router.get("/network/{analysis_id:path}", response_model=NetworkGraphData)
async def get_network_graph(
    analysis_id: str,
    max_nodes: int = 500,
    max_edges: int = 2000,
    hub_only: bool = False,
    min_correlation: float = 0.7
):
    """
    Get network graph data for 3D visualization.

    Parameters:
    - analysis_id: Analysis run ID (e.g., "tcga_brca_v2/run_20260109_115803")
    - max_nodes: Maximum number of nodes to return
    - max_edges: Maximum number of edges to return
    - hub_only: If true, only return hub genes and their connections
    - min_correlation: Minimum absolute correlation for edges

    Returns graph data in a format compatible with react-force-graph-3d.
    """
    # Find analysis directory
    analysis_path = RESULTS_BASE_DIR / analysis_id

    # Check for accumulated directory
    accumulated_path = analysis_path / "accumulated"
    if accumulated_path.exists():
        analysis_path = accumulated_path

    # Load network nodes
    nodes_file = analysis_path / "network_nodes.csv"
    if not nodes_file.exists():
        raise HTTPException(status_code=404, detail=f"Network nodes not found for {analysis_id}")

    nodes_df = pd.read_csv(nodes_file)

    # Load network edges
    edges_file = analysis_path / "network_edges.csv"
    if not edges_file.exists():
        raise HTTPException(status_code=404, detail=f"Network edges not found for {analysis_id}")

    edges_df = pd.read_csv(edges_file)

    # Load integrated gene table if available (for additional info)
    integrated_file = analysis_path / "integrated_gene_table.csv"
    integrated_df = None
    if integrated_file.exists():
        integrated_df = pd.read_csv(integrated_file)

    # Load DEG results if available
    deg_file = analysis_path / "deg_significant.csv"
    deg_df = None
    if deg_file.exists():
        deg_df = pd.read_csv(deg_file)

    # Filter edges by correlation
    edges_df = edges_df[edges_df['abs_correlation'] >= min_correlation]

    # If hub_only, filter to hub genes and their neighbors
    if hub_only:
        hub_genes = set(nodes_df[nodes_df['is_hub'] == True]['gene_id'].tolist())

        # Get neighbors of hub genes
        neighbor_genes = set()
        for _, edge in edges_df.iterrows():
            if edge['gene1'] in hub_genes:
                neighbor_genes.add(edge['gene2'])
            if edge['gene2'] in hub_genes:
                neighbor_genes.add(edge['gene1'])

        # Include hub genes and their immediate neighbors
        include_genes = hub_genes | neighbor_genes
        nodes_df = nodes_df[nodes_df['gene_id'].isin(include_genes)]
        edges_df = edges_df[
            (edges_df['gene1'].isin(include_genes)) &
            (edges_df['gene2'].isin(include_genes))
        ]

    # Limit nodes (prioritize hub genes)
    if len(nodes_df) > max_nodes:
        # Sort by hub_score, keep top nodes
        nodes_df = nodes_df.sort_values('hub_score', ascending=False).head(max_nodes)

        # Filter edges to only include remaining nodes
        node_ids = set(nodes_df['gene_id'].tolist())
        edges_df = edges_df[
            (edges_df['gene1'].isin(node_ids)) &
            (edges_df['gene2'].isin(node_ids))
        ]

    # Limit edges
    if len(edges_df) > max_edges:
        edges_df = edges_df.sort_values('abs_correlation', ascending=False).head(max_edges)

    # Build nodes list
    nodes = []
    for _, row in nodes_df.iterrows():
        gene_id = row['gene_id']

        # Get additional info from integrated table
        log2fc = 0.0
        padj = 1.0
        direction = "unchanged"
        pathway_count = 0
        db_matched = False
        db_sources = []
        confidence = "low"
        tags = []

        if integrated_df is not None:
            gene_row = integrated_df[integrated_df['gene_id'] == gene_id]
            if len(gene_row) > 0:
                gene_row = gene_row.iloc[0]
                log2fc = float(gene_row.get('log2FC', 0))
                padj = float(gene_row.get('padj', 1))
                direction = str(gene_row.get('direction', 'unchanged'))
                pathway_count = int(gene_row.get('pathway_count', 0))
                db_matched = bool(gene_row.get('db_matched', False))
                if pd.notna(gene_row.get('db_sources')) and gene_row.get('db_sources'):
                    db_sources = str(gene_row['db_sources']).split(';')
                confidence = str(gene_row.get('confidence', 'low'))
                if pd.notna(gene_row.get('tags')) and gene_row.get('tags'):
                    tags = str(gene_row['tags']).split(';')
        elif deg_df is not None:
            gene_row = deg_df[deg_df['gene_id'] == gene_id]
            if len(gene_row) > 0:
                gene_row = gene_row.iloc[0]
                log2fc = float(gene_row.get('log2FC', gene_row.get('log2FoldChange', 0)))
                padj = float(gene_row.get('padj', 1))
                direction = str(gene_row.get('direction', 'up' if log2fc > 0 else 'down'))

        # Extract gene symbol from ID if possible
        gene_symbol = gene_id.split('.')[0]  # ENSG00000034971.17 -> ENSG00000034971

        nodes.append(NetworkNode(
            id=gene_id,
            gene_symbol=gene_symbol,
            log2FC=log2fc,
            padj=padj,
            direction=direction,
            is_hub=bool(row.get('is_hub', False)),
            hub_score=float(row.get('hub_score', 0)),
            degree=int(row.get('degree', 0)),
            betweenness=float(row.get('betweenness', 0)),
            eigenvector=float(row.get('eigenvector', 0)),
            pathway_count=pathway_count,
            db_matched=db_matched,
            db_sources=db_sources,
            confidence=confidence,
            tags=tags
        ))

    # Build edges list
    edges = []
    for _, row in edges_df.iterrows():
        edges.append(NetworkEdge(
            source=row['gene1'],
            target=row['gene2'],
            correlation=float(row['correlation']),
            abs_correlation=float(row['abs_correlation'])
        ))

    # Calculate stats
    hub_count = sum(1 for n in nodes if n.is_hub)
    up_count = sum(1 for n in nodes if n.direction == 'up')
    down_count = sum(1 for n in nodes if n.direction == 'down')

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "hub_count": hub_count,
        "up_regulated": up_count,
        "down_regulated": down_count,
        "db_matched_count": sum(1 for n in nodes if n.db_matched),
        "avg_correlation": float(edges_df['abs_correlation'].mean()) if len(edges_df) > 0 else 0,
        "analysis_id": analysis_id
    }

    logger.info(f"Returning network: {len(nodes)} nodes, {len(edges)} edges for {analysis_id}")

    return NetworkGraphData(nodes=nodes, edges=edges, stats=stats)


@router.get("/gene/{analysis_id:path}/{gene_id}")
async def get_gene_detail(analysis_id: str, gene_id: str):
    """
    Get detailed information about a specific gene in an analysis.

    Includes expression data, network metrics, pathway associations,
    and database validation results.
    """
    # Find analysis directory
    analysis_path = RESULTS_BASE_DIR / analysis_id
    accumulated_path = analysis_path / "accumulated"
    if accumulated_path.exists():
        analysis_path = accumulated_path

    # Load integrated gene table
    integrated_file = analysis_path / "integrated_gene_table.csv"
    if not integrated_file.exists():
        raise HTTPException(status_code=404, detail="Integrated gene table not found")

    integrated_df = pd.read_csv(integrated_file)
    gene_row = integrated_df[integrated_df['gene_id'] == gene_id]

    if len(gene_row) == 0:
        raise HTTPException(status_code=404, detail=f"Gene {gene_id} not found")

    gene_data = gene_row.iloc[0].to_dict()

    # Load network data for neighbors
    nodes_file = analysis_path / "network_nodes.csv"
    edges_file = analysis_path / "network_edges.csv"

    neighbors = []
    if edges_file.exists():
        edges_df = pd.read_csv(edges_file)

        # Find neighbors
        neighbor_edges = edges_df[
            (edges_df['gene1'] == gene_id) | (edges_df['gene2'] == gene_id)
        ]

        for _, edge in neighbor_edges.iterrows():
            neighbor_id = edge['gene2'] if edge['gene1'] == gene_id else edge['gene1']
            neighbors.append({
                "gene_id": neighbor_id,
                "correlation": float(edge['correlation']),
                "abs_correlation": float(edge['abs_correlation'])
            })

        # Sort by correlation strength
        neighbors.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # Clean up NaN values
    for key, value in gene_data.items():
        if pd.isna(value):
            gene_data[key] = None

    return {
        "gene_id": gene_id,
        "gene_symbol": gene_id.split('.')[0],
        "expression": {
            "log2FC": gene_data.get('log2FC'),
            "padj": gene_data.get('padj'),
            "direction": gene_data.get('direction')
        },
        "network": {
            "is_hub": gene_data.get('is_hub'),
            "hub_score": gene_data.get('hub_score'),
            "neighbor_count": len(neighbors),
            "top_neighbors": neighbors[:10]
        },
        "pathways": {
            "count": gene_data.get('pathway_count', 0),
            "names": []  # Would need pathway data
        },
        "validation": {
            "db_matched": gene_data.get('db_matched'),
            "db_sources": str(gene_data.get('db_sources', '')).split(';') if gene_data.get('db_sources') else [],
            "cancer_type_match": gene_data.get('cancer_type_match'),
            "tme_related": gene_data.get('tme_related')
        },
        "interpretation": {
            "score": gene_data.get('interpretation_score'),
            "confidence": gene_data.get('confidence'),
            "tags": str(gene_data.get('tags', '')).split(';') if gene_data.get('tags') else []
        }
    }


# ═══════════════════════════════════════════════════════════════
# Background Tasks
# ═══════════════════════════════════════════════════════════════

def send_sse_message(job_id: str, message: dict):
    """Send message to SSE queue."""
    if job_id in _job_queues:
        _job_queues[job_id].put(message)


def run_pipeline_with_streaming(job_id: str, input_dir: Path, config: dict):
    """
    Run the RNA-seq pipeline with real-time SSE streaming.

    This runs in a background thread and sends progress updates via SSE.
    """
    try:
        status = _analysis_jobs[job_id]
        status.status = "running"

        send_sse_message(job_id, {
            "type": "pipeline_start",
            "job_id": job_id,
            "message": "파이프라인 시작",
            "timestamp": datetime.now().isoformat()
        })

        # Import pipeline
        try:
            from rnaseq_pipeline.orchestrator import RNAseqPipeline
            pipeline_available = True
        except ImportError as e:
            pipeline_available = False
            logger.warning(f"RNA-seq pipeline not available: {e}")

        if not pipeline_available:
            # Demo mode - simulate pipeline execution
            _run_demo_pipeline(job_id)
            return

        # Setup output directory
        output_dir = PROJECT_ROOT / "rnaseq_test_results" / f"web_analysis_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create pipeline
        pipeline = RNAseqPipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config
        )

        # Run agents one by one with progress updates
        agent_progress = {
            "agent1_deg": (0, 15),
            "agent2_network": (15, 35),
            "agent3_pathway": (35, 55),
            "agent4_validation": (55, 70),
            "agent5_visualization": (70, 90),
            "agent6_report": (90, 100)
        }

        completed_agents = []
        failed_agents = []

        for agent_name in pipeline.AGENT_ORDER:
            try:
                start_progress, end_progress = agent_progress.get(agent_name, (0, 100))
                agent_info = AGENT_INFO.get(agent_name, {})

                # Send agent start message
                send_sse_message(job_id, {
                    "type": "agent_start",
                    "agent": agent_name,
                    "name": agent_info.get("name", agent_name),
                    "description": agent_info.get("description", ""),
                    "progress": start_progress,
                    "timestamp": datetime.now().isoformat()
                })

                status.current_step = agent_info.get("name", agent_name)
                status.progress = start_progress

                # Run agent
                result = pipeline.run_agent(agent_name)

                # Send agent complete message
                send_sse_message(job_id, {
                    "type": "agent_complete",
                    "agent": agent_name,
                    "name": agent_info.get("name", agent_name),
                    "progress": end_progress,
                    "result_summary": _get_agent_summary(agent_name, result),
                    "timestamp": datetime.now().isoformat()
                })

                status.progress = end_progress
                completed_agents.append(agent_name)

            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                send_sse_message(job_id, {
                    "type": "agent_error",
                    "agent": agent_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                failed_agents.append(agent_name)
                # Continue with next agent

        # Pipeline complete
        if failed_agents:
            status.status = "completed_with_errors"
        else:
            status.status = "completed"

        status.progress = 100
        status.completed_at = datetime.now().isoformat()

        # Store run directory for report access
        _analysis_jobs[job_id].current_step = str(pipeline.run_dir)

        send_sse_message(job_id, {
            "type": "pipeline_complete",
            "job_id": job_id,
            "status": status.status,
            "completed_agents": completed_agents,
            "failed_agents": failed_agents,
            "run_dir": str(pipeline.run_dir),
            "report_path": str(pipeline.run_dir / "report.html"),
            "timestamp": datetime.now().isoformat()
        })

        # Signal end of SSE stream
        send_sse_message(job_id, None)

        logger.info(f"Completed RNA-seq analysis job: {job_id}")

    except Exception as e:
        logger.error(f"RNA-seq analysis failed for job {job_id}: {e}")
        status = _analysis_jobs[job_id]
        status.status = "failed"
        status.error = str(e)

        send_sse_message(job_id, {
            "type": "pipeline_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        send_sse_message(job_id, None)


def _get_agent_summary(agent_name: str, result: dict) -> dict:
    """Extract summary from agent result."""
    summary = {}
    if agent_name == "agent1_deg":
        summary["deg_count"] = result.get("deg_count", 0)
        summary["up_count"] = result.get("up_count", 0)
        summary["down_count"] = result.get("down_count", 0)
    elif agent_name == "agent2_network":
        summary["hub_count"] = result.get("hub_count", 0)
        summary["edge_count"] = result.get("edge_count", 0)
    elif agent_name == "agent3_pathway":
        summary["pathway_count"] = result.get("pathway_count", 0)
    elif agent_name == "agent4_validation":
        summary["validated_count"] = result.get("validated_count", 0)
    elif agent_name == "agent5_visualization":
        summary["figures"] = result.get("figures", [])
    elif agent_name == "agent6_report":
        summary["report_generated"] = result.get("report_generated", False)
    return summary


def _run_demo_pipeline(job_id: str):
    """Run demo pipeline simulation."""
    import time

    status = _analysis_jobs[job_id]
    agents = [
        ("agent1_deg", "DEG Analysis", 15),
        ("agent2_network", "Network Analysis", 35),
        ("agent3_pathway", "Pathway Enrichment", 55),
        ("agent4_validation", "DB Validation", 70),
        ("agent5_visualization", "Visualization", 90),
        ("agent6_report", "Report Generation", 100)
    ]

    for agent_id, agent_name, progress in agents:
        send_sse_message(job_id, {
            "type": "agent_start",
            "agent": agent_id,
            "name": agent_name,
            "progress": progress - 15,
            "timestamp": datetime.now().isoformat()
        })

        status.current_step = agent_name
        time.sleep(1)  # Simulate work

        send_sse_message(job_id, {
            "type": "agent_complete",
            "agent": agent_id,
            "name": agent_name,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        })

        status.progress = progress

    status.status = "completed"
    status.completed_at = datetime.now().isoformat()

    send_sse_message(job_id, {
        "type": "pipeline_complete",
        "job_id": job_id,
        "status": "completed",
        "mode": "demo",
        "timestamp": datetime.now().isoformat()
    })

    send_sse_message(job_id, None)


async def run_pipeline_task(job_id: str, request: AnalysisRequest):
    """
    Run the RNA-seq pipeline as a background task (legacy).
    """
    # Create input directory from request
    input_dir = Path(request.count_matrix_path).parent

    config = {
        "cancer_type": request.disease_context,
        "condition_column": request.condition_column,
        "contrast": [request.treatment_label, request.control_label]
    }

    # Use new streaming function
    run_pipeline_with_streaming(job_id, input_dir, config)
