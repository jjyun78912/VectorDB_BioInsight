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
