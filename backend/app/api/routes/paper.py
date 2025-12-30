"""
Paper management API endpoints.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path

router = APIRouter()


class Paper(BaseModel):
    """Paper model."""
    title: str
    doi: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    chunk_count: int = 0


class PaperListResponse(BaseModel):
    """Paper list response."""
    papers: List[Paper]
    total: int


class StatsResponse(BaseModel):
    """Collection stats response."""
    collection_name: str
    disease_domain: str
    total_papers: int
    total_chunks: int
    embedding_model: str
    chunks_by_section: Dict[str, int]


@router.get("/", response_model=PaperListResponse)
async def list_papers(
    domain: str = Query("pheochromocytoma", description="Disease domain")
):
    """
    List all indexed papers.
    """
    try:
        from backend.app.core.vector_store import create_vector_store

        vector_store = create_vector_store(disease_domain=domain)
        papers = vector_store.get_all_papers()

        paper_list = []
        for p in papers:
            year_val = p.get("year")
            # Handle empty string or invalid year
            if year_val == "" or year_val is None:
                year_val = None
            elif isinstance(year_val, str):
                try:
                    year_val = int(year_val)
                except ValueError:
                    year_val = None

            paper_list.append(Paper(
                title=p.get("title", "Unknown"),
                doi=p.get("doi") or None,
                year=year_val,
                chunk_count=p.get("chunk_count", 0)
            ))

        return PaperListResponse(
            papers=paper_list,
            total=len(paper_list)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    domain: str = Query("pheochromocytoma", description="Disease domain")
):
    """
    Get collection statistics.
    """
    try:
        from backend.app.core.vector_store import create_vector_store

        vector_store = create_vector_store(disease_domain=domain)
        stats = vector_store.get_collection_stats()

        return StatsResponse(
            collection_name=stats["collection_name"],
            disease_domain=stats["disease_domain"],
            total_papers=stats["total_papers"],
            total_chunks=stats["total_chunks"],
            embedding_model=stats["embedding_model"],
            chunks_by_section=stats["chunks_by_section"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class IndexResponse(BaseModel):
    """Index response."""
    success: bool
    message: str
    paper_title: Optional[str] = None
    chunks_created: int = 0


@router.post("/upload", response_model=IndexResponse)
async def upload_paper(
    file: UploadFile = File(...),
    domain: str = Form("pheochromocytoma")
):
    """
    Upload and index a PDF paper.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        from backend.app.core.indexer import create_indexer

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            indexer = create_indexer(disease_domain=domain)
            result = indexer.index_pdf(tmp_path)

            if result.get("error"):
                return IndexResponse(
                    success=False,
                    message=result["error"]
                )

            return IndexResponse(
                success=True,
                message="Paper indexed successfully",
                paper_title=result.get("title"),
                chunks_created=result.get("chunks_added", 0) or result.get("chunks_created", 0)
            )

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{paper_title}")
async def delete_paper(
    paper_title: str,
    domain: str = Query("pheochromocytoma", description="Disease domain")
):
    """
    Delete a paper from the index.
    """
    try:
        from backend.app.core.vector_store import create_vector_store

        vector_store = create_vector_store(disease_domain=domain)
        deleted = vector_store.delete_paper(paper_title)

        return {
            "success": True,
            "message": f"Deleted {deleted} chunks from paper: {paper_title}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
