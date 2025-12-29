"""
Web Crawler API Routes.

Provides endpoints for:
- DOI/URL-based paper fetching
- Real-time PubMed search
- Trending papers
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.web_crawler_agent import WebCrawlerAgent, FetchedPaper, TRENDING_CATEGORIES

router = APIRouter()

# Initialize agent
crawler_agent = WebCrawlerAgent()


# ==================== Request/Response Models ====================

class DOIRequest(BaseModel):
    doi: str = Field(..., description="DOI to fetch (e.g., '10.1038/s41586-021-03819-2')")


class URLRequest(BaseModel):
    url: str = Field(..., description="Paper URL (DOI, PubMed, or PMC URL)")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results")
    sort: str = Field(default="relevance", description="Sort order: 'relevance' or 'pub_date'")
    min_year: Optional[int] = Field(default=None, description="Minimum publication year")


class PaperResponse(BaseModel):
    id: str
    source: str
    title: str
    authors: List[str]
    abstract: str
    journal: str
    year: int
    doi: str
    pmid: str
    pmcid: str
    url: str
    keywords: List[str]
    citation_count: int
    trend_score: float
    recency_score: float
    fetched_at: str


class SearchResponse(BaseModel):
    query: str
    total_results: int
    papers: List[PaperResponse]


class TrendingResponse(BaseModel):
    category: str
    papers: List[PaperResponse]
    cached: bool


class CategoriesResponse(BaseModel):
    categories: List[str]


class BatchDOIRequest(BaseModel):
    dois: List[str] = Field(..., description="List of DOIs to fetch")


class BatchResponse(BaseModel):
    total_requested: int
    total_found: int
    papers: List[PaperResponse]


# ==================== Helper Functions ====================

def paper_to_response(paper: FetchedPaper) -> PaperResponse:
    """Convert FetchedPaper to API response."""
    return PaperResponse(
        id=paper.id,
        source=paper.source,
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        journal=paper.journal,
        year=paper.year,
        doi=paper.doi,
        pmid=paper.pmid,
        pmcid=paper.pmcid,
        url=paper.url,
        keywords=paper.keywords,
        citation_count=paper.citation_count,
        trend_score=paper.trend_score,
        recency_score=paper.recency_score,
        fetched_at=paper.fetched_at
    )


# ==================== Endpoints ====================

@router.post("/fetch/doi", response_model=PaperResponse, summary="Fetch paper by DOI")
async def fetch_by_doi(request: DOIRequest):
    """
    Fetch paper metadata by DOI.

    Uses CrossRef and Semantic Scholar APIs to retrieve:
    - Title, authors, abstract
    - Journal, year, citations
    - Related references

    Example DOI: 10.1038/s41586-021-03819-2
    """
    try:
        paper = await crawler_agent.fetch_by_doi(request.doi)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found for DOI: {request.doi}")

        return paper_to_response(paper)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching paper: {str(e)}")


@router.post("/fetch/url", response_model=PaperResponse, summary="Fetch paper by URL")
async def fetch_by_url(request: URLRequest):
    """
    Fetch paper by URL.

    Supports:
    - DOI URLs: https://doi.org/10.1038/...
    - PubMed URLs: https://pubmed.ncbi.nlm.nih.gov/12345678/
    - PMC URLs: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234567/
    """
    try:
        paper = await crawler_agent.fetch_by_url(request.url)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found for URL: {request.url}")

        return paper_to_response(paper)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching paper: {str(e)}")


@router.post("/search", response_model=SearchResponse, summary="Search PubMed in real-time")
async def search_pubmed(request: SearchRequest):
    """
    Search PubMed in real-time.

    Returns papers matching the query with metadata including:
    - Title, abstract, authors
    - Journal, year, DOI
    - Trend and recency scores
    """
    try:
        min_date = f"{request.min_year}/01/01" if request.min_year else None

        papers = await crawler_agent.search_pubmed(
            query=request.query,
            max_results=request.max_results,
            sort=request.sort,
            min_date=min_date
        )

        return SearchResponse(
            query=request.query,
            total_results=len(papers),
            papers=[paper_to_response(p) for p in papers]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/search", response_model=SearchResponse, summary="Search PubMed (GET)")
async def search_pubmed_get(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Max results"),
    sort: str = Query(default="relevance", description="Sort: relevance or pub_date"),
    min_year: Optional[int] = Query(default=None, description="Min year")
):
    """Search PubMed via GET request."""
    request = SearchRequest(
        query=q,
        max_results=limit,
        sort=sort,
        min_year=min_year
    )
    return await search_pubmed(request)


@router.get("/trending/{category}", response_model=TrendingResponse, summary="Get trending papers")
async def get_trending(
    category: str,
    limit: int = Query(default=10, ge=1, le=20, description="Number of papers"),
    no_cache: bool = Query(default=False, description="Bypass cache")
):
    """
    Get trending papers for a category.

    Available categories:
    - oncology
    - immunotherapy
    - gene_therapy
    - neurology
    - infectious_disease
    - ai_medicine
    - genomics
    - drug_discovery
    """
    if category not in TRENDING_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Available: {list(TRENDING_CATEGORIES.keys())}"
        )

    try:
        papers = await crawler_agent.get_trending_papers(
            category=category,
            max_results=limit,
            use_cache=not no_cache
        )

        # Check if results were from cache
        cache_key = f"{category}_{limit}"
        cached = cache_key in crawler_agent._trending_cache and not no_cache

        return TrendingResponse(
            category=category,
            papers=[paper_to_response(p) for p in papers],
            cached=cached
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching trending papers: {str(e)}")


@router.get("/trending", response_model=TrendingResponse, summary="Get trending papers (default)")
async def get_trending_default(
    limit: int = Query(default=10, ge=1, le=20)
):
    """Get trending papers from default category (oncology)."""
    return await get_trending("oncology", limit=limit)


@router.get("/categories", response_model=CategoriesResponse, summary="List trending categories")
async def list_categories():
    """Get list of available trending paper categories."""
    return CategoriesResponse(categories=list(TRENDING_CATEGORIES.keys()))


@router.post("/batch/doi", response_model=BatchResponse, summary="Batch fetch by DOIs")
async def batch_fetch_dois(request: BatchDOIRequest):
    """
    Fetch multiple papers by DOI.

    Note: Rate limited, may take a few seconds for large batches.
    """
    if len(request.dois) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 DOIs per batch")

    try:
        papers = await crawler_agent.batch_fetch_dois(request.dois)

        return BatchResponse(
            total_requested=len(request.dois),
            total_found=len(papers),
            papers=[paper_to_response(p) for p in papers]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch fetch error: {str(e)}")


@router.get("/health", summary="Health check")
async def health_check():
    """Check if the crawler service is healthy."""
    return {
        "status": "healthy",
        "service": "web_crawler",
        "available_categories": list(TRENDING_CATEGORIES.keys()),
        "cache_size": len(crawler_agent._trending_cache)
    }
