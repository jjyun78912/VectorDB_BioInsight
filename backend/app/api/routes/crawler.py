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

from src.web_crawler_agent import WebCrawlerAgent, FetchedPaper, TRENDING_CATEGORIES, MAJOR_JOURNALS

# Try to import Playwright crawler (optional dependency)
try:
    from src.playwright_crawler import PlaywrightDeepCrawler, CrawlResult, PLAYWRIGHT_AVAILABLE
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightDeepCrawler = None
    CrawlResult = None

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
    min_year: Optional[int] = Query(default=None, description="Min year"),
    hybrid: bool = Query(default=True, description="Use hybrid search (latest + high-impact)")
):
    """Search PubMed via GET request. Uses hybrid search by default."""
    try:
        if hybrid:
            # Use hybrid search (latest + high-impact)
            papers = await crawler_agent.search_pubmed_hybrid(
                query=q,
                max_results=limit
            )
        else:
            # Standard search
            min_date = f"{min_year}/01/01" if min_year else None
            papers = await crawler_agent.search_pubmed(
                query=q,
                max_results=limit,
                sort=sort,
                min_date=min_date
            )

        return SearchResponse(
            query=q,
            total_results=len(papers),
            papers=[paper_to_response(p) for p in papers]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@router.get("/trending/{category}", response_model=TrendingResponse, summary="Get trending papers")
async def get_trending(
    category: str,
    limit: int = Query(default=10, ge=1, le=20, description="Number of papers"),
    no_cache: bool = Query(default=False, description="Bypass cache"),
    major_only: bool = Query(default=True, description="Filter to major journals only (Nature, Cell, Science, NEJM, Lancet, etc.)")
):
    """
    Get trending papers for a category.

    By default, filters to major journals only (Nature, Cell, Science, NEJM, Lancet, JAMA, etc.)
    Set major_only=false to include all journals.

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
            use_cache=not no_cache,
            major_journals_only=major_only
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


@router.get("/major-journals", summary="List major journals")
async def list_major_journals():
    """Get list of major journals used for filtering trending papers."""
    return {
        "journals": MAJOR_JOURNALS,
        "count": len(MAJOR_JOURNALS),
        "description": "High-impact journals including Nature, Cell, Science, NEJM, Lancet, and related publications."
    }


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


@router.get("/similar/{pmid}", response_model=SearchResponse, summary="Find similar papers via PubMed")
async def find_similar_papers_pubmed(
    pmid: str,
    limit: int = Query(default=10, ge=1, le=20, description="Max results")
):
    """
    Find similar papers using PubMed's Related Articles feature.
    Uses NCBI E-utilities elink to find related PMIDs, then fetches their metadata.
    """
    try:
        papers = await crawler_agent.find_similar_papers(pmid, max_results=limit)

        return SearchResponse(
            query=f"similar to PMID:{pmid}",
            total_results=len(papers),
            papers=[paper_to_response(p) for p in papers]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar papers: {str(e)}")


@router.get("/health", summary="Health check")
async def health_check():
    """Check if the crawler service is healthy."""
    return {
        "status": "healthy",
        "service": "web_crawler",
        "available_categories": list(TRENDING_CATEGORIES.keys()),
        "cache_size": len(crawler_agent._trending_cache),
        "playwright_available": PLAYWRIGHT_AVAILABLE
    }


# ==================== Playwright Deep Crawler Endpoints ====================

class DeepCrawlRequest(BaseModel):
    url: str = Field(..., description="URL to crawl")
    download_pdf: bool = Field(default=False, description="Attempt to download PDF")


class DeepCrawlResponse(BaseModel):
    url: str
    success: bool
    title: str = ""
    abstract: str = ""
    full_text_preview: str = ""
    full_text_length: int = 0
    references_count: int = 0
    pdf_path: Optional[str] = None
    error: Optional[str] = None


class ReferenceCrawlRequest(BaseModel):
    doi_or_url: str = Field(..., description="DOI or URL to start from")
    depth: int = Field(default=1, ge=1, le=3, description="Crawl depth (1-3)")
    max_papers: int = Field(default=10, ge=1, le=50, description="Max papers to fetch")


class ReferenceCrawlResponse(BaseModel):
    starting_point: str
    depth: int
    total_papers: int
    papers: List[PaperResponse]


@router.post("/deep/crawl", response_model=DeepCrawlResponse, summary="Deep crawl a paper page")
async def deep_crawl(request: DeepCrawlRequest):
    """
    Deep crawl a paper page using Playwright.

    Extracts:
    - Full text content
    - Title and abstract
    - References (DOIs)
    - Optionally downloads PDF

    Note: Requires Playwright to be installed.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Playwright not available. Install with: pip install playwright && playwright install chromium"
        )

    crawler = None
    try:
        crawler = PlaywrightDeepCrawler(headless=True)

        # Extract full text
        result = await crawler.extract_full_text(request.url)

        # Optionally download PDF
        pdf_path = None
        if request.download_pdf:
            pdf_path = await crawler.download_pdf(request.url)

        return DeepCrawlResponse(
            url=request.url,
            success=result.success,
            title=result.title,
            abstract=result.abstract,
            full_text_preview=result.full_text[:1000] if result.full_text else "",
            full_text_length=len(result.full_text),
            references_count=len(result.references),
            pdf_path=pdf_path,
            error=result.error
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deep crawl error: {str(e)}")
    finally:
        if crawler:
            await crawler.close()


@router.post("/deep/pdf", summary="Download PDF from publisher")
async def download_pdf(request: URLRequest):
    """
    Download PDF from a publisher page.

    Attempts to find and download the PDF from the given article page.
    Supports major publishers: Nature, Cell, Science, MDPI, Frontiers, etc.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Playwright not available. Install with: pip install playwright && playwright install chromium"
        )

    crawler = None
    try:
        crawler = PlaywrightDeepCrawler(headless=True)
        pdf_path = await crawler.download_pdf(request.url)

        if not pdf_path:
            raise HTTPException(status_code=404, detail="Could not find or download PDF")

        return {
            "success": True,
            "url": request.url,
            "pdf_path": pdf_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF download error: {str(e)}")
    finally:
        if crawler:
            await crawler.close()


@router.post("/deep/references", response_model=ReferenceCrawlResponse, summary="Crawl paper references")
async def crawl_references(request: ReferenceCrawlRequest):
    """
    Recursively crawl paper references.

    Starts from a DOI or URL and follows references up to the specified depth.
    Returns metadata for all discovered papers.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Playwright not available. Install with: pip install playwright && playwright install chromium"
        )

    crawler = None
    try:
        crawler = PlaywrightDeepCrawler(headless=True)
        papers = await crawler.crawl_references(
            request.doi_or_url,
            depth=request.depth,
            max_papers=request.max_papers
        )

        return ReferenceCrawlResponse(
            starting_point=request.doi_or_url,
            depth=request.depth,
            total_papers=len(papers),
            papers=[paper_to_response(p) for p in papers]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reference crawl error: {str(e)}")
    finally:
        if crawler:
            await crawler.close()


@router.get("/deep/status", summary="Check Playwright availability")
async def playwright_status():
    """Check if Playwright deep crawler is available."""
    return {
        "playwright_available": PLAYWRIGHT_AVAILABLE,
        "message": "Ready for deep crawling" if PLAYWRIGHT_AVAILABLE else "Install with: pip install playwright && playwright install chromium"
    }
