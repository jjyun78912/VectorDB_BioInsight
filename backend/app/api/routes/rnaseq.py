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
# File Format Detection and Conversion Helpers
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import re


def detect_file_format(content: bytes, filename: str) -> str:
    """
    Detect file format from content and filename.

    Returns one of:
    - 'csv': Comma-separated values
    - 'tsv': Tab-separated values
    - 'geo_series_matrix': GEO series matrix format
    - 'unknown': Unknown format
    """
    try:
        text = content.decode('utf-8', errors='ignore')
        first_lines = text.split('\n')[:20]

        # Check for GEO series_matrix format (starts with !)
        if any(line.startswith('!') for line in first_lines):
            return 'geo_series_matrix'

        # Check first data line for delimiter
        first_data_line = next((line for line in first_lines if line.strip() and not line.startswith('#')), '')

        if '\t' in first_data_line:
            return 'tsv'
        elif ',' in first_data_line:
            return 'csv'
        else:
            # Check filename extension
            if filename.lower().endswith('.tsv') or filename.lower().endswith('.txt'):
                return 'tsv'
            elif filename.lower().endswith('.csv'):
                return 'csv'

    except Exception as e:
        logger.warning(f"Error detecting file format: {e}")

    return 'unknown'


def convert_tsv_to_csv(content: bytes) -> bytes:
    """Convert TSV content to CSV format."""
    try:
        text = content.decode('utf-8', errors='ignore')
        # Replace tabs with commas
        csv_text = text.replace('\t', ',')
        return csv_text.encode('utf-8')
    except Exception as e:
        logger.warning(f"TSV to CSV conversion failed: {e}")
        return content


def generate_metadata_from_count_matrix(count_content: bytes) -> tuple:
    """
    Generate metadata.csv from count_matrix column names.

    Extracts sample IDs from count_matrix columns and determines condition
    based on T/N suffix pattern (common in GEO datasets like GSE81089).

    Pattern examples:
    - L400T → tumor (T suffix)
    - L511N → normal (N suffix)
    - GSM2142443 → unknown (no clear suffix)

    Returns:
        tuple: (metadata_csv_bytes, sample_condition_mapping)
    """
    try:
        text = count_content.decode('utf-8', errors='ignore')
        first_line = text.split('\n')[0].strip()

        # Detect delimiter
        delimiter = '\t' if '\t' in first_line else ','

        # Get column names (first row)
        columns = [c.strip().strip('"') for c in first_line.split(delimiter)]

        # First column is usually gene_id, rest are sample columns
        sample_ids = columns[1:]

        if not sample_ids:
            logger.warning("No sample columns found in count matrix")
            return None, {}

        # Determine conditions from sample IDs
        conditions = []
        for sid in sample_ids:
            # Check T/N suffix pattern (most common in GEO cancer datasets)
            # Patterns: L400T, sample_1T, TCGA-XX-XXXX-01T
            sid_upper = sid.upper()

            # Check for explicit T (tumor) or N (normal) suffix
            if sid_upper.endswith('T') or '_T' in sid_upper or '-T' in sid_upper:
                # Check it's not a number like "L400T1"
                if len(sid) >= 2 and sid[-2:-1].isalpha():
                    # Second to last is also a letter, might be part of name
                    pass
                if re.search(r'[A-Za-z]T$', sid) or re.search(r'_T\d*$', sid_upper):
                    conditions.append('tumor')
                    continue

            if sid_upper.endswith('N') or '_N' in sid_upper or '-N' in sid_upper:
                if re.search(r'[A-Za-z]N$', sid) or re.search(r'_N\d*$', sid_upper):
                    conditions.append('normal')
                    continue

            # Check for tumor keywords
            if any(kw in sid_upper for kw in ['TUMOR', 'CANCER', 'CARCINOMA', 'MALIGNANT']):
                conditions.append('tumor')
                continue

            # Check for normal keywords
            if any(kw in sid_upper for kw in ['NORMAL', 'CTRL', 'CONTROL', 'HEALTHY', 'ADJACENT']):
                conditions.append('normal')
                continue

            # Default: try to guess from the last character (simple T/N)
            if sid.endswith('T') or sid.endswith('t'):
                conditions.append('tumor')
            elif sid.endswith('N') or sid.endswith('n'):
                conditions.append('normal')
            else:
                conditions.append('unknown')

        # Create CSV content
        csv_lines = ['sample_id,condition']
        for sid, cond in zip(sample_ids, conditions):
            csv_lines.append(f'{sid},{cond}')

        csv_content = '\n'.join(csv_lines).encode('utf-8')

        # Create mapping for reference
        sample_mapping = dict(zip(sample_ids, conditions))

        # Log summary
        from collections import Counter
        cond_counts = Counter(conditions)
        logger.info(f"Generated metadata from count_matrix: {len(sample_ids)} samples, conditions: {dict(cond_counts)}")

        return csv_content, sample_mapping

    except Exception as e:
        logger.error(f"Failed to generate metadata from count_matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def parse_geo_series_matrix(content: bytes, count_matrix_content: bytes = None) -> tuple:
    """
    Parse GEO series_matrix.txt format and extract metadata.

    IMPORTANT: If count_matrix_content is provided, we use the sample IDs from
    count_matrix columns instead of GSM IDs to ensure consistency.

    GEO series_matrix format:
    - Lines starting with ! are metadata headers
    - !Sample_geo_accession contains sample IDs (GSM numbers)
    - !Sample_characteristics contains sample annotations

    Returns:
        tuple: (metadata_csv_bytes, sample_condition_mapping)
    """
    try:
        # If count_matrix is provided, generate metadata from it directly
        # This ensures sample IDs match between files
        if count_matrix_content:
            cm_text = count_matrix_content.decode('utf-8', errors='ignore')
            first_line = cm_text.split('\n')[0].strip()
            delimiter = '\t' if '\t' in first_line else ','
            cm_columns = [c.strip().strip('"') for c in first_line.split(delimiter)]
            cm_sample_ids = cm_columns[1:]  # Skip gene_id column

            # Check if these are GSM IDs or actual sample names
            has_gsm_ids = any(sid.startswith('GSM') for sid in cm_sample_ids)

            if not has_gsm_ids and cm_sample_ids:
                # count_matrix has actual sample names (e.g., L400T), generate metadata from it
                logger.info("Count matrix has non-GSM sample IDs, generating metadata from column names")
                return generate_metadata_from_count_matrix(count_matrix_content)

        # Continue with GEO series_matrix parsing if count_matrix uses GSM IDs
        text = content.decode('utf-8', errors='ignore')
        lines = text.split('\n')

        sample_ids = []
        characteristics = {}
        condition_guesses = {}
        sample_labels = {}  # For patterns like "L400T" (T=tumor, N=normal)

        for line in lines:
            line = line.strip()

            # Extract sample IDs
            if line.startswith('!Sample_geo_accession'):
                parts = line.split('\t')
                sample_ids = [p.strip().strip('"') for p in parts[1:] if p.strip()]

            # Extract characteristics (may have multiple rows)
            elif line.startswith('!Sample_characteristics_ch'):
                parts = line.split('\t')
                # Extract key:value pairs like "tissue: tumor" or "tumor (t) or normal (n): L400T"
                for i, p in enumerate(parts[1:], start=0):
                    p = p.strip().strip('"')
                    if ':' in p:
                        key, val = p.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        val = val.strip()

                        # Special handling for "tumor (t) or normal (n)" pattern
                        if 'tumor' in key and 'normal' in key and i < len(sample_ids):
                            # Value like "L400T" - T suffix means tumor, N suffix means normal
                            if val.endswith('T') or val.endswith('t'):
                                sample_labels[sample_ids[i]] = 'tumor'
                            elif val.endswith('N') or val.endswith('n'):
                                sample_labels[sample_ids[i]] = 'normal'

                        if key not in characteristics:
                            characteristics[key] = [None] * len(sample_ids)
                        if i < len(characteristics[key]):
                            characteristics[key][i] = val

            # Try to detect condition from title or source
            elif line.startswith('!Sample_title') or line.startswith('!Sample_source_name'):
                parts = line.split('\t')
                for i, p in enumerate(parts[1:]):
                    p_orig = p.strip().strip('"')
                    p_lower = p_orig.lower()
                    if i < len(sample_ids):
                        sid = sample_ids[i]
                        # Check for explicit keywords
                        if any(term in p_lower for term in ['nsclc', 'tumor', 'cancer', 'tumour', 'malignant', 'carcinoma']):
                            condition_guesses[sid] = 'tumor'
                        elif any(term in p_lower for term in ['non-malignant', 'normal', 'healthy', 'control', 'adjacent']):
                            condition_guesses[sid] = 'normal'
                        # Check for "matched sample_L511N" pattern (N = normal)
                        elif 'matched' in p_lower and p_orig.endswith('N'):
                            sample_labels[sid] = 'normal'

        # If no sample IDs found, try to extract from count matrix
        if not sample_ids and count_matrix_content:
            try:
                cm_text = count_matrix_content.decode('utf-8', errors='ignore')
                first_line = cm_text.split('\n')[0]
                delimiter = '\t' if '\t' in first_line else ','
                parts = first_line.split(delimiter)
                sample_ids = [p.strip().strip('"') for p in parts[1:] if p.strip() and p.strip().startswith('GSM')]
            except:
                pass

        if not sample_ids:
            logger.warning("No sample IDs found in GEO series_matrix")
            return None, {}

        # Determine conditions - priority: sample_labels > condition_guesses > characteristics
        conditions = []
        for sid in sample_ids:
            if sid in sample_labels:
                conditions.append(sample_labels[sid])
            elif sid in condition_guesses:
                conditions.append(condition_guesses[sid])
            else:
                # Try characteristics columns
                found_cond = None
                for key in ['tissue', 'tissue_type', 'disease', 'disease_state', 'cell_type', 'condition', 'group']:
                    if key in characteristics:
                        idx = sample_ids.index(sid)
                        if idx < len(characteristics[key]) and characteristics[key][idx]:
                            found_cond = characteristics[key][idx]
                            break
                conditions.append(found_cond if found_cond else 'unknown')

        # Normalize condition values
        normalized_conditions = []
        for cond in conditions:
            if cond is None or cond == 'unknown':
                normalized_conditions.append('unknown')
            elif any(term in str(cond).lower() for term in ['tumor', 'cancer', 'tumour', 'malignant', 'carcinoma', 'primary', 'nsclc']):
                normalized_conditions.append('tumor')
            elif any(term in str(cond).lower() for term in ['normal', 'healthy', 'control', 'adjacent', 'matched', 'non-malignant']):
                normalized_conditions.append('normal')
            else:
                normalized_conditions.append(str(cond).lower().replace(' ', '_'))

        # Create CSV content
        csv_lines = ['sample_id,condition']
        for sid, cond in zip(sample_ids, normalized_conditions):
            csv_lines.append(f'{sid},{cond}')

        csv_content = '\n'.join(csv_lines).encode('utf-8')

        # Create mapping for reference
        sample_mapping = dict(zip(sample_ids, normalized_conditions))

        # Log summary
        from collections import Counter
        cond_counts = Counter(normalized_conditions)
        logger.info(f"Parsed GEO series_matrix: {len(sample_ids)} samples, conditions: {dict(cond_counts)}")

        return csv_content, sample_mapping

    except Exception as e:
        logger.error(f"Failed to parse GEO series_matrix: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}


def fix_split_header(content: bytes) -> tuple:
    """
    Fix count_matrix files where the header row is split across multiple lines.

    Some tools/exporters create files where the header is artificially split,
    with continuation lines starting with a comma. This function detects and fixes that.

    Example of broken format:
        Line 1: gene_id,sample1,sample2
        Line 2: ,sample3,sample4,sample5  <- continuation (starts with comma)
        Line 3: GENE1,1,2,3,4,5           <- actual data

    Returns:
        tuple: (fixed_content_bytes, was_fixed)
    """
    try:
        text = content.decode('utf-8', errors='ignore')
        lines = text.split('\n')

        if len(lines) < 2:
            return content, False

        # Check if line 2 starts with comma (indicates split header)
        if lines[1].startswith(','):
            logger.info("Detected split header in count_matrix - fixing...")

            # Find where the actual data starts (line that doesn't start with comma and has gene ID)
            header_parts = [lines[0].strip()]
            data_start_idx = 1

            for i in range(1, min(10, len(lines))):  # Check up to 10 lines
                if lines[i].startswith(','):
                    header_parts.append(lines[i].strip())
                    data_start_idx = i + 1
                else:
                    break

            # Combine all header parts
            combined_header = ''.join(header_parts)

            # Reconstruct file
            fixed_lines = [combined_header] + [l for l in lines[data_start_idx:] if l.strip()]
            fixed_content = '\n'.join(fixed_lines).encode('utf-8')

            header_cols = len(combined_header.split(','))
            logger.info(f"Fixed split header: combined {len(header_parts)} lines into {header_cols} columns")

            return fixed_content, True

        return content, False

    except Exception as e:
        logger.warning(f"Error checking for split header: {e}")
        return content, False


def preprocess_uploaded_files(count_content: bytes, count_filename: str,
                               meta_content: bytes, meta_filename: str) -> tuple:
    """
    Preprocess uploaded files to ensure correct format and sample ID consistency.

    Key features:
    1. Fixes split headers (header spanning multiple lines)
    2. Converts TSV to CSV if needed
    3. Parses GEO series_matrix format
    4. IMPORTANTLY: Validates sample IDs match between count_matrix and metadata
       - If mismatch detected, regenerates metadata from count_matrix column names

    Returns:
        tuple: (processed_count_bytes, processed_meta_bytes, warnings_list)
    """
    warnings = []

    # First, fix split headers (common issue with some export tools)
    count_content, header_was_fixed = fix_split_header(count_content)
    if header_was_fixed:
        warnings.append("Fixed split header in count_matrix (header was across multiple lines)")

    # Detect count matrix format
    count_format = detect_file_format(count_content, count_filename)
    logger.info(f"Count matrix format detected: {count_format} ({count_filename})")

    if count_format == 'tsv':
        count_content = convert_tsv_to_csv(count_content)
        warnings.append(f"Converted count matrix from TSV to CSV format")
    elif count_format == 'geo_series_matrix':
        warnings.append("Count matrix appears to be GEO series_matrix format - this may cause issues")

    # Detect metadata format
    meta_format = detect_file_format(meta_content, meta_filename)
    logger.info(f"Metadata format detected: {meta_format} ({meta_filename})")

    if meta_format == 'geo_series_matrix':
        # Parse GEO format and convert to standard metadata
        # This will automatically use count_matrix column names if they don't start with GSM
        parsed_meta, _ = parse_geo_series_matrix(meta_content, count_content)
        if parsed_meta:
            meta_content = parsed_meta
            warnings.append(f"Converted GEO series_matrix to standard metadata format")
        else:
            warnings.append("Failed to parse GEO series_matrix - manual metadata may be required")
    elif meta_format == 'tsv':
        meta_content = convert_tsv_to_csv(meta_content)
        warnings.append(f"Converted metadata from TSV to CSV format")

    # CRITICAL: Validate sample IDs match between count_matrix and metadata
    # This catches cases where metadata uses GSM IDs but count_matrix uses sample names
    try:
        # Get count_matrix sample IDs (column names)
        cm_text = count_content.decode('utf-8', errors='ignore')
        cm_first_line = cm_text.split('\n')[0].strip()
        cm_delimiter = '\t' if '\t' in cm_first_line else ','
        cm_columns = [c.strip().strip('"') for c in cm_first_line.split(cm_delimiter)]
        cm_sample_ids = set(cm_columns[1:])  # Skip gene_id column

        # Get metadata sample IDs
        meta_text = meta_content.decode('utf-8', errors='ignore')
        meta_lines = meta_text.strip().split('\n')
        meta_delimiter = '\t' if '\t' in meta_lines[0] else ','

        # Find sample_id column index
        meta_header = [c.strip().strip('"').lower() for c in meta_lines[0].split(meta_delimiter)]
        sample_id_col = 0  # Default to first column
        for i, col in enumerate(meta_header):
            if col in ['sample_id', 'sampleid', 'sample', 'id', 'name']:
                sample_id_col = i
                break

        meta_sample_ids = set()
        for line in meta_lines[1:]:
            if line.strip():
                parts = line.split(meta_delimiter)
                if len(parts) > sample_id_col:
                    meta_sample_ids.add(parts[sample_id_col].strip().strip('"'))

        # Check overlap
        matching = cm_sample_ids & meta_sample_ids
        only_in_count = cm_sample_ids - meta_sample_ids
        only_in_meta = meta_sample_ids - cm_sample_ids

        logger.info(f"Sample ID validation: {len(matching)} matching, {len(only_in_count)} only in count_matrix, {len(only_in_meta)} only in metadata")

        # If very low overlap, metadata is probably wrong - regenerate from count_matrix
        overlap_ratio = len(matching) / len(cm_sample_ids) if cm_sample_ids else 0

        if overlap_ratio < 0.5 and len(cm_sample_ids) > 0:
            logger.warning(f"Sample ID mismatch detected! Only {overlap_ratio*100:.1f}% overlap. Regenerating metadata from count_matrix columns.")
            warnings.append(f"Sample ID mismatch detected ({overlap_ratio*100:.1f}% overlap) - regenerating metadata from count_matrix")

            # Generate metadata from count_matrix column names
            generated_meta, _ = generate_metadata_from_count_matrix(count_content)
            if generated_meta:
                meta_content = generated_meta
                warnings.append(f"Metadata regenerated from count_matrix column names with T/N suffix detection")
            else:
                warnings.append("Failed to generate metadata from count_matrix - analysis may fail")

    except Exception as e:
        logger.warning(f"Sample ID validation failed: {e}")
        # Continue without validation

    return count_content, meta_content, warnings


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

# Agent info for frontend display - Bulk RNA-seq (6 agents)
BULK_AGENT_INFO = {
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
    "ml_prediction": {
        "name": "ML Prediction",
        "description": "암종 예측 및 SHAP 분석",
        "icon": "brain",
        "order": 6
    },
    "agent6_report": {
        "name": "Report Generation",
        "description": "HTML 보고서 생성",
        "icon": "file-text",
        "order": 7
    }
}

# Agent info for Single-cell RNA-seq (1 unified agent)
SINGLECELL_AGENT_INFO = {
    "sc_qc": {
        "name": "QC & Filtering",
        "description": "품질 관리 및 세포 필터링",
        "icon": "filter",
        "order": 1
    },
    "sc_normalize": {
        "name": "Normalization",
        "description": "정규화 및 스케일링",
        "icon": "sliders",
        "order": 2
    },
    "sc_hvg": {
        "name": "HVG Selection",
        "description": "고변이 유전자 선별",
        "icon": "trending-up",
        "order": 3
    },
    "sc_dimred": {
        "name": "Dimensionality Reduction",
        "description": "PCA, UMAP 차원 축소",
        "icon": "move",
        "order": 4
    },
    "sc_clustering": {
        "name": "Clustering",
        "description": "Leiden 클러스터링",
        "icon": "grid",
        "order": 5
    },
    "sc_annotation": {
        "name": "Cell Type Annotation",
        "description": "세포 유형 주석",
        "icon": "tag",
        "order": 6
    },
    "sc_deg": {
        "name": "Marker Genes",
        "description": "클러스터별 마커 유전자 탐지",
        "icon": "activity",
        "order": 7
    },
    "sc_visualization": {
        "name": "Visualization",
        "description": "UMAP, Violin, Dot plot 생성",
        "icon": "bar-chart-2",
        "order": 8
    },
    "sc_report": {
        "name": "Report Generation",
        "description": "Single-cell 분석 리포트 생성",
        "icon": "file-text",
        "order": 9
    }
}

# Default to bulk agent info for backward compatibility
AGENT_INFO = BULK_AGENT_INFO


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
async def get_agents_info(data_type: str = "bulk"):
    """Get detailed information about pipeline agents.

    Args:
        data_type: "bulk" for bulk RNA-seq, "singlecell" for single-cell RNA-seq
    """
    if data_type == "singlecell":
        agent_info = SINGLECELL_AGENT_INFO
    else:
        agent_info = BULK_AGENT_INFO

    return {
        "data_type": data_type,
        "agents": agent_info,
        "total": len(agent_info),
        "order": list(agent_info.keys())
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
    data_type: Optional[str] = None  # "bulk" | "singlecell" | "unknown"
    data_type_confidence: Optional[float] = None
    n_genes: Optional[int] = None
    n_samples: Optional[int] = None
    recommended_pipeline: Optional[str] = None


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


class DataTypeDetectionResult(BaseModel):
    """Result of data type detection (bulk vs single-cell)."""
    data_type: str  # "bulk" | "singlecell" | "unknown"
    confidence: float  # 0-1
    n_genes: int
    n_samples: int
    recommended_pipeline: str
    evidence: Dict[str, Any] = {}


@router.get("/detect-type/{job_id}", response_model=DataTypeDetectionResult)
async def detect_data_type_endpoint(job_id: str):
    """
    Detect data type (bulk vs single-cell) for uploaded files.

    Returns detection result with confidence score and recommended pipeline.
    """
    job_dir = UPLOAD_DIR / job_id

    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        from rnaseq_pipeline.utils.data_type_detector import detect_data_type
        result = detect_data_type(job_dir)

        return DataTypeDetectionResult(
            data_type=result.get("data_type", "unknown"),
            confidence=result.get("confidence", 0.0),
            n_genes=result.get("n_genes", 0),
            n_samples=result.get("n_samples", 0),
            recommended_pipeline=result.get("recommended_pipeline", "Unknown"),
            evidence=result.get("evidence", {})
        )
    except Exception as e:
        logger.error(f"Data type detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


async def fetch_geo_metadata(gse_id: str) -> dict:
    """Fetch sample metadata from GEO for a GSE dataset."""
    import aiohttp

    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_id}&targ=gsm&form=text&view=brief"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    return {}
                text = await response.text()

        # Parse GEO text format
        samples = {}
        current_gsm = None
        current_title = None
        current_condition = None

        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('^SAMPLE = '):
                # Save previous sample
                if current_gsm and current_title:
                    # Detect condition from title (ends with T or N)
                    title_upper = current_title.upper()
                    if title_upper.endswith('T') or '_T' in title_upper or '-T' in title_upper:
                        current_condition = 'treatment'
                    elif title_upper.endswith('N') or '_N' in title_upper or '-N' in title_upper:
                        current_condition = 'control'
                    samples[current_gsm] = {
                        'title': current_title,
                        'condition': current_condition or 'unknown'
                    }
                # Start new sample
                current_gsm = line.replace('^SAMPLE = ', '')
                current_title = None
                current_condition = None
            elif line.startswith('!Sample_title = '):
                current_title = line.replace('!Sample_title = ', '')
            elif 'tumor (t) or normal (n):' in line.lower():
                # Direct condition annotation
                value = line.split(':')[-1].strip().upper()
                if value.endswith('T'):
                    current_condition = 'treatment'
                elif value.endswith('N'):
                    current_condition = 'control'
            elif 'tissue:' in line.lower() or 'sample type:' in line.lower():
                value = line.split(':')[-1].strip().lower()
                if any(w in value for w in ['tumor', 'tumour', 'cancer', 'malignant']):
                    current_condition = 'treatment'
                elif any(w in value for w in ['normal', 'healthy', 'adjacent', 'control']):
                    current_condition = 'control'

        # Save last sample
        if current_gsm and current_title:
            title_upper = current_title.upper()
            if not current_condition:
                if title_upper.endswith('T') or '_T' in title_upper or '-T' in title_upper:
                    current_condition = 'treatment'
                elif title_upper.endswith('N') or '_N' in title_upper or '-N' in title_upper:
                    current_condition = 'control'
            samples[current_gsm] = {
                'title': current_title,
                'condition': current_condition or 'unknown'
            }

        return samples
    except Exception as e:
        logger.warning(f"Failed to fetch GEO metadata for {gse_id}: {e}")
        return {}


@router.post("/preview-samples")
async def preview_count_matrix_samples(
    count_matrix: UploadFile = File(..., description="Count matrix CSV/TSV file")
):
    """
    Preview samples from count matrix and auto-detect conditions.

    Analyzes column names to detect tumor/normal, case/control patterns.
    Also fetches GEO metadata if GSE ID is detected in filename.
    Returns sample list with suggested condition assignments.
    """
    import pandas as pd
    import io
    import re

    try:
        content = await count_matrix.read()
        filename = count_matrix.filename or ""

        # Try to extract GSE ID from filename
        gse_match = re.search(r'GSE\d+', filename, re.IGNORECASE)
        geo_metadata = {}
        if gse_match:
            gse_id = gse_match.group(0).upper()
            logger.info(f"Detected GSE ID: {gse_id}, fetching GEO metadata...")
            geo_metadata = await fetch_geo_metadata(gse_id)

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

        # Count total genes (fast: just count newlines instead of parsing)
        gene_count = content.count(b'\n') - 1  # subtract header row
        if gene_count < 0:
            gene_count = 0

        # Auto-detect conditions from column names
        # Priority: Check T/N suffix patterns first (more specific), then keywords

        # T suffix/prefix patterns (highest priority - most specific)
        t_suffix_patterns = [
            r'[_\-\.]t[0-9]*$',      # ends with _T, _T1, _T01, .T
            r'^t[0-9]+[_\-\.]',      # starts with T1_, T01-
            r'[_\-]t[_\-]',          # contains _T_ or -T-
            r'[_\-]t$',              # ends with _T or -T
        ]
        # N suffix/prefix patterns (highest priority - most specific)
        n_suffix_patterns = [
            r'[_\-\.]n[0-9]*$',      # ends with _N, _N1, _N01, .N
            r'^n[0-9]+[_\-\.]',      # starts with N1_, N01-
            r'[_\-]n[_\-]',          # contains _N_ or -N-
            r'[_\-]n$',              # ends with _N or -N
        ]

        # Treatment/Tumor keyword patterns (lower priority)
        treatment_keywords = [
            r'tumor', r'tumour', r'cancer', r'carcinoma', r'malignant',
            r'disease', r'treated', r'treatment',
            r'primary', r'metastatic', r'metastasis', r'lesion',
            r'종양', r'암',
        ]
        # Control/Normal keyword patterns (lower priority)
        control_keywords = [
            r'normal', r'control', r'ctrl', r'healthy', r'benign',
            r'untreated', r'reference', r'ref', r'baseline', r'base',
            r'adjacent', r'adj',
            r'정상', r'대조군',
        ]

        samples = []
        detected_conditions = {"treatment": [], "control": [], "unknown": []}
        geo_matched = 0

        for sample in sample_columns:
            sample_lower = sample.lower()
            condition = "unknown"

            # 0. First check GEO metadata if available (highest priority)
            if geo_metadata and sample in geo_metadata:
                condition = geo_metadata[sample].get('condition', 'unknown')
                if condition != 'unknown':
                    geo_matched += 1

            # 1. Check N suffix (control)
            if condition == "unknown":
                for pattern in n_suffix_patterns:
                    if re.search(pattern, sample_lower):
                        condition = "control"
                        break

            # 2. Check T suffix (treatment)
            if condition == "unknown":
                for pattern in t_suffix_patterns:
                    if re.search(pattern, sample_lower):
                        condition = "treatment"
                        break

            # 3. Check control keywords
            if condition == "unknown":
                for pattern in control_keywords:
                    if re.search(pattern, sample_lower):
                        condition = "control"
                        break

            # 4. Check treatment keywords (lowest priority)
            if condition == "unknown":
                for pattern in treatment_keywords:
                    if re.search(pattern, sample_lower):
                        condition = "treatment"
                        break

            samples.append(SampleInfo(sample_id=sample, condition=condition))
            detected_conditions[condition].append(sample)

        # Suggest labels based on detected patterns
        suggested_treatment = "tumor" if detected_conditions["treatment"] else "treatment"
        suggested_control = "normal" if detected_conditions["control"] else "control"

        # Log detection results
        if geo_metadata:
            logger.info(f"GEO metadata: {len(geo_metadata)} samples found, {geo_matched} matched")
        logger.info(f"Detection results: {len(detected_conditions['treatment'])} treatment, "
                   f"{len(detected_conditions['control'])} control, "
                   f"{len(detected_conditions['unknown'])} unknown")

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
        # Read file contents
        count_content = await count_matrix.read()
        meta_content = await metadata.read()

        # Preprocess files (detect format and convert if needed)
        processed_count, processed_meta, format_warnings = preprocess_uploaded_files(
            count_content, count_matrix.filename,
            meta_content, metadata.filename
        )

        # Save processed count matrix
        count_path = job_dir / "count_matrix.csv"
        with open(count_path, "wb") as f:
            f.write(processed_count)

        # Save processed metadata
        meta_path = job_dir / "metadata.csv"
        with open(meta_path, "wb") as f:
            f.write(processed_meta)

        # Save config
        config = {
            "cancer_type": cancer_type,
            "study_name": study_name or f"RNA-seq Analysis {job_id}",
            "condition_column": condition_column,
            "contrast": [treatment_label, control_label],
            "uploaded_at": datetime.now().isoformat(),
            "original_files": {
                "count_matrix": count_matrix.filename,
                "metadata": metadata.filename
            },
            "format_conversions": format_warnings
        }
        config_path = job_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Files uploaded for job {job_id}: {count_matrix.filename}, {metadata.filename}")
        if format_warnings:
            logger.info(f"Format conversions applied: {format_warnings}")

        # Detect data type (bulk vs single-cell)
        data_type_info = {}
        try:
            from rnaseq_pipeline.utils.data_type_detector import detect_data_type
            detection_result = detect_data_type(job_dir)
            data_type_info = {
                "data_type": detection_result.get("data_type", "unknown"),
                "data_type_confidence": detection_result.get("confidence", 0.0),
                "n_genes": detection_result.get("n_genes", 0),
                "n_samples": detection_result.get("n_samples", 0),
                "recommended_pipeline": detection_result.get("recommended_pipeline", "Unknown")
            }
            # Save detection result to config
            config["data_type_detection"] = detection_result
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"Data type detected: {data_type_info['data_type']} (confidence: {data_type_info['data_type_confidence']:.2f})")
        except Exception as e:
            logger.warning(f"Data type detection failed: {e}")
            data_type_info = {
                "data_type": "unknown",
                "data_type_confidence": 0.0,
                "n_genes": 0,
                "n_samples": 0,
                "recommended_pipeline": "Unknown"
            }

        return UploadResponse(
            job_id=job_id,
            message="Files uploaded successfully",
            files_received=[count_matrix.filename, metadata.filename],
            input_dir=str(job_dir),
            **data_type_info
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
    pipeline_type: Optional[str] = Field(None, description="Force pipeline type: 'bulk', 'singlecell', or 'auto'")


@router.post("/start/{job_id}", response_model=AnalysisStatus)
async def start_analysis_from_upload(
    job_id: str,
    background_tasks: BackgroundTasks,
    cancer_type: Optional[str] = None,
    study_name: Optional[str] = None,
    pipeline_type: Optional[str] = None
):
    """
    Start RNA-seq analysis for previously uploaded files.

    Use the job_id returned from /upload endpoint.
    Progress can be monitored via /stream/{job_id} SSE endpoint.

    Args:
        job_id: Job ID from upload
        cancer_type: Override cancer type for analysis
        study_name: Override study name
        pipeline_type: Force pipeline type ('bulk', 'singlecell', 'auto')
                      If not specified, auto-detects based on data characteristics
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
    if pipeline_type:
        config["pipeline_type"] = pipeline_type

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
    Automatically detects bulk vs single-cell data and routes appropriately.
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

        # Determine pipeline type from config or auto-detect
        pipeline_type = config.get("pipeline_type", "auto")

        # Create pipeline with auto-detection
        pipeline = RNAseqPipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            pipeline_type=pipeline_type
        )

        # Send detection result
        send_sse_message(job_id, {
            "type": "data_type_detected",
            "data_type": pipeline.pipeline_type,
            "detection_result": pipeline.detection_result,
            "timestamp": datetime.now().isoformat()
        })

        # Select agent info and progress based on pipeline type
        if pipeline.pipeline_type == "singlecell":
            agent_info_dict = SINGLECELL_AGENT_INFO
            # Single-cell progress mapping (9 virtual stages in 1 agent)
            agent_progress = {
                "sc_qc": (0, 10),
                "sc_normalize": (10, 20),
                "sc_hvg": (20, 30),
                "sc_dimred": (30, 40),
                "sc_clustering": (40, 55),
                "sc_annotation": (55, 65),
                "sc_deg": (65, 80),
                "sc_visualization": (80, 90),
                "sc_report": (90, 100)
            }
        else:
            agent_info_dict = BULK_AGENT_INFO
            # Bulk RNA-seq progress (6 agents + ML prediction virtual)
            agent_progress = {
                "agent1_deg": (0, 15),
                "agent2_network": (15, 30),
                "agent3_pathway": (30, 45),
                "agent4_validation": (45, 60),
                "agent5_visualization": (60, 75),
                "ml_prediction": (75, 88),  # Virtual stage within agent6
                "agent6_report": (88, 100)
            }

        completed_agents = []
        failed_agents = []

        # Get agent order from pipeline
        agent_order = pipeline.get_agent_order()

        # Handle single-cell pipeline differently (1 agent with virtual stages)
        if pipeline.pipeline_type == "singlecell":
            _run_singlecell_pipeline_with_streaming(
                job_id, pipeline, agent_info_dict, agent_progress,
                completed_agents, failed_agents, status
            )
        else:
            # Bulk RNA-seq pipeline (6 agents)
            for agent_name in agent_order:
                try:
                    # Send ML prediction virtual stage before agent6_report
                    if agent_name == "agent6_report":
                        ml_info = agent_info_dict.get("ml_prediction", {})
                        ml_start, ml_end = agent_progress.get("ml_prediction", (75, 88))

                        # ML prediction start
                        send_sse_message(job_id, {
                            "type": "agent_start",
                            "agent": "ml_prediction",
                            "name": ml_info.get("name", "ML Prediction"),
                            "progress": ml_start,
                            "timestamp": datetime.now().isoformat()
                        })
                        status.current_step = ml_info.get("name", "ML Prediction")
                        status.progress = ml_start

                    start_progress, end_progress = agent_progress.get(agent_name, (0, 100))
                    agent_info = agent_info_dict.get(agent_name, {})

                    # For agent6_report, ML prediction completes when it starts
                    if agent_name == "agent6_report":
                        ml_info = agent_info_dict.get("ml_prediction", {})
                        ml_start, ml_end = agent_progress.get("ml_prediction", (75, 88))

                        # ML prediction complete (runs within agent6_report)
                        send_sse_message(job_id, {
                            "type": "agent_complete",
                            "agent": "ml_prediction",
                            "name": ml_info.get("name", "ML Prediction"),
                            "progress": ml_end,
                            "timestamp": datetime.now().isoformat()
                        })
                        completed_agents.append("ml_prediction")

                    # Send agent start message
                    send_sse_message(job_id, {
                        "type": "agent_start",
                        "agent": agent_name,
                        "name": agent_info.get("name", agent_name),
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


def _run_singlecell_pipeline_with_streaming(
    job_id: str,
    pipeline,
    agent_info_dict: dict,
    agent_progress: dict,
    completed_agents: list,
    failed_agents: list,
    status
):
    """
    Run single-cell pipeline with virtual stage progress updates.

    The SingleCellAgent runs as one unit but we send progress updates
    for each virtual stage to provide better UI feedback.
    """
    import time

    # Virtual stages in order
    virtual_stages = [
        "sc_qc", "sc_normalize", "sc_hvg", "sc_dimred",
        "sc_clustering", "sc_annotation", "sc_deg",
        "sc_visualization", "sc_report"
    ]

    try:
        # Send start messages for first virtual stage
        first_stage = virtual_stages[0]
        stage_info = agent_info_dict.get(first_stage, {})
        start_progress, _ = agent_progress.get(first_stage, (0, 10))

        send_sse_message(job_id, {
            "type": "agent_start",
            "agent": first_stage,
            "name": stage_info.get("name", "QC & Filtering"),
            "progress": start_progress,
            "timestamp": datetime.now().isoformat()
        })
        status.current_step = stage_info.get("name", "QC & Filtering")
        status.progress = start_progress

        # Run the single-cell agent (runs all stages internally)
        result = pipeline.run_agent("singlecell")

        # Mark all virtual stages as complete
        for stage_name in virtual_stages:
            stage_info = agent_info_dict.get(stage_name, {})
            _, end_progress = agent_progress.get(stage_name, (0, 100))

            send_sse_message(job_id, {
                "type": "agent_complete",
                "agent": stage_name,
                "name": stage_info.get("name", stage_name),
                "progress": end_progress,
                "timestamp": datetime.now().isoformat()
            })
            completed_agents.append(stage_name)
            status.progress = end_progress
            time.sleep(0.1)  # Small delay for UI

        # Add final result summary
        send_sse_message(job_id, {
            "type": "singlecell_complete",
            "result_summary": _get_agent_summary("singlecell", result),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Single-cell pipeline failed: {e}")
        send_sse_message(job_id, {
            "type": "agent_error",
            "agent": "singlecell",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
        failed_agents.append("singlecell")


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
    elif agent_name == "singlecell":
        # Single-cell pipeline summary
        summary["n_cells"] = result.get("n_cells", 0)
        summary["n_genes"] = result.get("n_genes", 0)
        summary["n_clusters"] = result.get("n_clusters", 0)
        summary["n_celltypes"] = result.get("n_celltypes", 0)
        summary["n_markers"] = result.get("n_markers", 0)
    return summary


def _run_demo_pipeline(job_id: str):
    """Run demo pipeline simulation."""
    import time

    status = _analysis_jobs[job_id]
    agents = [
        ("agent1_deg", "차등 발현 분석", 15),
        ("agent2_network", "네트워크 분석", 30),
        ("agent3_pathway", "경로 분석", 45),
        ("agent4_validation", "데이터베이스 검증", 60),
        ("agent5_visualization", "시각화", 75),
        ("ml_prediction", "ML 예측", 88),
        ("agent6_report", "리포트 생성", 100)
    ]

    prev_progress = 0
    for agent_id, agent_name, progress in agents:
        send_sse_message(job_id, {
            "type": "agent_start",
            "agent": agent_id,
            "name": agent_name,
            "progress": prev_progress,
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
        prev_progress = progress

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
