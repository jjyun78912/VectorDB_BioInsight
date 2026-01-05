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
import os
from datetime import datetime, timedelta

import sys
from pathlib import Path

from backend.app.core.web_crawler_agent import WebCrawlerAgent, FetchedPaper, TRENDING_CATEGORIES, TRENDING_CATEGORY_BASES, MAJOR_JOURNALS
from backend.app.core.translator import get_translator
from backend.app.core.oncology_trends import get_trend_matcher, ONCOLOGY_TRENDS

# Get NCBI API key for higher rate limits
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# Try to import Playwright crawler (optional dependency)
try:
    from backend.app.core.playwright_crawler import PlaywrightDeepCrawler, CrawlResult, PLAYWRIGHT_AVAILABLE
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    PlaywrightDeepCrawler = None
    CrawlResult = None

router = APIRouter()

# Initialize agent with NCBI API key for higher rate limits (10 req/sec vs 3 req/sec)
crawler_agent = WebCrawlerAgent(ncbi_api_key=NCBI_API_KEY)


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


class LensScoreDetail(BaseModel):
    score: float = 0.0
    confidence: str = "low"


class BasePaperResponse(BaseModel):
    """Base response model with common paper fields."""
    id: str
    source: str
    title: str
    title_ko: Optional[str] = None  # Korean translation
    authors: List[str]
    abstract: str
    abstract_ko: Optional[str] = None  # Korean translation
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
    # Advanced trend metrics
    citation_velocity: Optional[float] = None  # Citation growth rate
    publication_surge: Optional[float] = None  # Topic publication growth rate
    influential_citation_count: Optional[int] = None  # Citations from influential papers
    # Core Paper Ranking fields
    article_type: Optional[str] = None  # "Review", "Clinical Trial", etc.
    core_score: float = 0.0  # 0-100 (higher = more representative of field)
    is_core_paper: bool = False  # True if core_score >= 60
    # Lens Classification fields
    primary_lens: str = ""  # "overview", "trend", "mechanism", "clinical"
    lens_scores: Optional[Dict[str, LensScoreDetail]] = None
    lens_explanation: str = ""


class PaperResponse(BasePaperResponse):
    """Standard paper response."""
    pass


class LensGroup(BaseModel):
    """Papers grouped under a specific lens."""
    lens: str  # "overview", "trend", "mechanism", "clinical"
    label: str  # Human-readable label
    description: str  # What this lens represents
    papers: List[PaperResponse]
    count: int


class SearchResponse(BaseModel):
    query: str
    query_translated: Optional[str] = None  # English translation of Korean query
    was_translated: bool = False  # Whether query was translated
    total_results: int
    papers: List[PaperResponse]
    # Lens-grouped results
    lens_groups: Optional[Dict[str, LensGroup]] = None


class TrendingResponse(BaseModel):
    category: str
    papers: List[PaperResponse]
    cached: bool


# Enhanced Trending with Trend Context
class TrendInfo(BaseModel):
    """Information about a defined trend."""
    id: str
    name: str
    emoji: str
    color: str
    why_trending: str  # "왜 지금 이게 중요한지"
    clinical_relevance: Optional[str] = None
    matched_terms: List[str] = []  # Keywords that matched


class TrendedPaper(BasePaperResponse):
    """Paper with trend matching information."""
    trend_match: Optional[TrendInfo] = None


class TrendGroup(BaseModel):
    """Papers grouped under a specific trend."""
    trend_id: str
    trend_name: str
    emoji: str
    color: str
    why_trending: str
    papers: List[TrendedPaper]
    paper_count: int


class TrendCategory(BaseModel):
    """A category containing multiple trends."""
    category_id: str
    category_name: str
    emoji: str
    trends: Dict[str, TrendGroup]
    total_papers: int


class EnhancedTrendingResponse(BaseModel):
    """
    Enhanced trending response with:
    - Trend-first structure (trends defined, then papers mapped)
    - Why each trend matters
    - Papers grouped by sub-trends
    """
    domain: str  # "oncology"
    total_papers: int
    categories: Dict[str, TrendCategory]
    all_trends: List[Dict]  # All defined trends for reference
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

# Category display metadata for trends
_CATEGORY_DISPLAY_INFO = {
    "tumor_evolution": ("🧬 Cancer Evolution & Plasticity", "🧬"),
    "immunotherapy": ("🛡️ Immunotherapy & TME", "🛡️"),
    "precision_medicine": ("🎯 Precision Medicine", "🎯"),
    "cancer_prevention": ("🛡️ Cancer Prevention", "🛡️"),
    "treatment_resistance": ("💊 Treatment Resistance", "💊"),
    "emerging_targets": ("🎯 Emerging Targets", "🎯"),
}


def _init_trend_category(cat_id: str) -> TrendCategory:
    """
    Initialize a TrendCategory with display metadata.

    Args:
        cat_id: Category ID (e.g., "tumor_evolution")

    Returns:
        Initialized TrendCategory
    """
    cat_name, cat_emoji = _CATEGORY_DISPLAY_INFO.get(cat_id, (cat_id, "🔬"))
    return TrendCategory(
        category_id=cat_id,
        category_name=cat_name,
        emoji=cat_emoji,
        trends={},
        total_papers=0
    )


def _init_trend_group(trend_id: str, match) -> TrendGroup:
    """
    Initialize a TrendGroup from a trend match.

    Args:
        trend_id: Trend identifier
        match: TrendMatch object with trend metadata

    Returns:
        Initialized TrendGroup
    """
    return TrendGroup(
        trend_id=trend_id,
        trend_name=match.trend_name,
        emoji=match.emoji,
        color=match.color,
        why_trending=match.why_trending,
        papers=[],
        paper_count=0
    )


def _map_paper_to_trend(paper_resp: PaperResponse, matcher) -> Optional[tuple]:
    """
    Map a paper to its primary trend.

    Args:
        paper_resp: PaperResponse object to map
        matcher: TrendMatcher instance

    Returns:
        Tuple of (category_id, trend_id, TrendedPaper) if match found, None otherwise
    """
    # Find matching trends
    matches = matcher.match_paper(
        paper_resp.title,
        paper_resp.abstract,
        paper_resp.keywords
    )

    if not matches:
        return None

    primary_match = matches[0]
    cat_id = primary_match.category
    trend_id = primary_match.trend_id

    # Create TrendedPaper with trend info
    trended_paper = TrendedPaper(
        **paper_resp.model_dump(),
        trend_match=TrendInfo(
            id=primary_match.trend_id,
            name=primary_match.trend_name,
            emoji=primary_match.emoji,
            color=primary_match.color,
            why_trending=primary_match.why_trending,
            clinical_relevance=ONCOLOGY_TRENDS[trend_id].clinical_relevance if trend_id in ONCOLOGY_TRENDS else None,
            matched_terms=primary_match.matched_terms[:5]
        )
    )

    return (cat_id, trend_id, trended_paper)


def paper_to_response(paper: FetchedPaper, title_ko: str = None, abstract_ko: str = None) -> PaperResponse:
    """Convert FetchedPaper to API response."""
    # Convert lens_scores dict to LensScoreDetail objects
    lens_scores_response = None
    if paper.lens_scores:
        lens_scores_response = {
            lens: LensScoreDetail(score=data.get("score", 0), confidence=data.get("confidence", "low"))
            for lens, data in paper.lens_scores.items()
        }

    return PaperResponse(
        id=paper.id,
        source=paper.source,
        title=paper.title,
        title_ko=title_ko,
        authors=paper.authors,
        abstract=paper.abstract,
        abstract_ko=abstract_ko,
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
        fetched_at=paper.fetched_at,
        # Advanced trend metrics
        citation_velocity=paper.citation_velocity if paper.citation_velocity > 0 else None,
        publication_surge=paper.publication_surge if paper.publication_surge > 0 else None,
        influential_citation_count=paper.influential_citation_count if paper.influential_citation_count > 0 else None,
        # Core Paper Ranking
        article_type=paper.article_type or None,
        core_score=paper.core_score,
        is_core_paper=paper.is_core_paper,
        # Lens Classification
        primary_lens=paper.primary_lens,
        lens_scores=lens_scores_response,
        lens_explanation=paper.lens_explanation,
    )


# Lens metadata for grouping
LENS_METADATA = {
    "overview": {
        "label": "Overview",
        "description": "Reviews, summaries, and foundational concepts to understand the topic"
    },
    "trend": {
        "label": "Trending",
        "description": "Recent high-impact research and emerging developments"
    },
    "mechanism": {
        "label": "Mechanism",
        "description": "Molecular pathways, gene functions, and experimental findings"
    },
    "clinical": {
        "label": "Clinical",
        "description": "Clinical trials, treatments, biomarkers, and translational research"
    }
}


def group_papers_by_lens(papers: List[PaperResponse]) -> Dict[str, LensGroup]:
    """Group papers by their primary lens."""
    groups = {lens: [] for lens in LENS_METADATA.keys()}

    for paper in papers:
        if paper.primary_lens in groups:
            groups[paper.primary_lens].append(paper)

    # Convert to LensGroup objects
    result = {}
    for lens, paper_list in groups.items():
        if paper_list:  # Only include non-empty groups
            meta = LENS_METADATA[lens]
            result[lens] = LensGroup(
                lens=lens,
                label=meta["label"],
                description=meta["description"],
                papers=paper_list,
                count=len(paper_list)
            )

    return result


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
    hybrid: bool = Query(default=True, description="Use hybrid search (latest + high-impact)"),
    translate: bool = Query(default=True, description="Enable Korean translation")
):
    """
    Search PubMed via GET request. Uses hybrid search by default.

    Korean language support:
    - If query is in Korean, automatically translates to English for search
    - Results can be translated back to Korean (title_ko, abstract_ko fields)
    """
    try:
        # Check if query needs translation (Korean -> English)
        original_query = q
        query_translated = None
        was_translated = False
        translate_results = False

        if translate:
            try:
                translator = get_translator()
                query_translated, was_translated = translator.translate_search_query(q)
                if was_translated:
                    q = query_translated  # Use English query for search
                    translate_results = True  # Translate results back to Korean
            except Exception as e:
                print(f"Translation service error: {e}")
                # Continue with original query if translation fails

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

        # Translate results to Korean if original query was Korean
        paper_responses = []
        if translate_results and papers:
            try:
                translator = get_translator()
                for p in papers:
                    # Translate title and abstract to Korean
                    title_ko = translator.translate_to_korean(p.title) if p.title else None
                    # Only translate abstract if not too long (to save API calls)
                    abstract_ko = None
                    if p.abstract and len(p.abstract) < 2000:
                        abstract_ko = translator.translate_to_korean(p.abstract)
                    paper_responses.append(paper_to_response(p, title_ko=title_ko, abstract_ko=abstract_ko))
            except Exception as e:
                print(f"Result translation error: {e}")
                # Fallback to non-translated results
                paper_responses = [paper_to_response(p) for p in papers]
        else:
            paper_responses = [paper_to_response(p) for p in papers]

        # Group papers by lens
        lens_groups = group_papers_by_lens(paper_responses) if paper_responses else None

        return SearchResponse(
            query=original_query,
            query_translated=query_translated if was_translated else None,
            was_translated=was_translated,
            total_results=len(papers),
            papers=paper_responses,
            lens_groups=lens_groups
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


@router.get("/trending-enhanced/{domain}", response_model=EnhancedTrendingResponse, summary="Enhanced trending with trend context")
async def get_enhanced_trending(
    domain: str = "oncology",
    limit: int = Query(default=20, ge=5, le=50, description="Total papers to fetch"),
    major_only: bool = Query(default=True, description="Filter to major journals")
):
    """
    Enhanced trending endpoint with proper trend grouping.

    Key differences from /trending:
    1. Trends are DEFINED FIRST (not inferred from papers)
    2. Papers are MAPPED TO trends (with match scores)
    3. Each trend includes "why it matters" context
    4. Papers grouped by sub-trends within categories

    Structure:
    - domain: "oncology"
    - categories:
      - tumor_evolution:
        - trends:
          - lineage_plasticity: [papers with why_trending context]
          - ecdna: [papers with why_trending context]
      - immunotherapy:
        - trends:
          - immunotherapy_resistance: [papers]
          - tumor_microenvironment: [papers]
    """
    if domain != "oncology":
        raise HTTPException(
            status_code=400,
            detail="Currently only 'oncology' domain is supported for enhanced trending"
        )

    try:
        # Fetch trending papers using existing logic
        papers = await crawler_agent.get_trending_papers(
            category="oncology",
            max_results=limit,
            use_cache=True,
            major_journals_only=major_only
        )

        # Convert to response format with lens classification
        paper_responses = [paper_to_response(p) for p in papers]

        # Match papers to defined trends
        trend_matcher = get_trend_matcher()

        # Group papers by trend
        categories: Dict[str, TrendCategory] = {}

        for paper_resp in paper_responses:
            # Map paper to trend
            result = _map_paper_to_trend(paper_resp, trend_matcher)

            if result:
                cat_id, trend_id, trended_paper = result

                # Initialize category if needed
                if cat_id not in categories:
                    categories[cat_id] = _init_trend_category(cat_id)

                # Initialize trend if needed
                if trend_id not in categories[cat_id].trends:
                    # Get primary match for initialization
                    matches = trend_matcher.match_paper(
                        paper_resp.title,
                        paper_resp.abstract,
                        paper_resp.keywords
                    )
                    categories[cat_id].trends[trend_id] = _init_trend_group(trend_id, matches[0])

                # Add paper to trend
                categories[cat_id].trends[trend_id].papers.append(trended_paper)
                categories[cat_id].trends[trend_id].paper_count += 1
                categories[cat_id].total_papers += 1

        # Get all defined trends for reference
        all_trends = trend_matcher.get_all_trends()

        # Calculate total
        total_papers = sum(cat.total_papers for cat in categories.values())

        return EnhancedTrendingResponse(
            domain=domain,
            total_papers=total_papers,
            categories=categories,
            all_trends=all_trends,
            cached=True
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error fetching enhanced trending: {str(e)}")


@router.get("/trends/defined", summary="Get all defined oncology trends")
async def get_defined_trends():
    """
    Get all defined oncology trends with context.

    Returns the trend definitions (not papers) including:
    - Why each trend is important
    - Clinical relevance
    - Key keywords
    """
    trend_matcher = get_trend_matcher()
    return {
        "domain": "oncology",
        "trends": trend_matcher.get_all_trends(),
        "total_trends": len(ONCOLOGY_TRENDS)
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


# ==================== Daily Papers Endpoint ====================

class DailyPapersResponse(BaseModel):
    """Response for daily papers."""
    category: str
    date: str  # YYYY-MM-DD
    papers: List[PaperResponse]
    total: int
    cached: bool


# Cache for daily papers (refreshed at KST 07:00)
_daily_cache: Dict[str, Dict] = {}
_daily_cache_time: Dict[str, datetime] = {}


def _is_cache_valid(cache_key: str, cache_time_dict: Dict[str, datetime]) -> bool:
    """
    Check if cache is valid (before next KST 07:00).

    Args:
        cache_key: Key to check in cache
        cache_time_dict: Dictionary mapping cache keys to timestamps

    Returns:
        True if cache is still valid, False otherwise
    """
    if cache_key not in cache_time_dict:
        return False

    from datetime import timezone
    cached_at = cache_time_dict[cache_key]
    now = datetime.now(timezone.utc)

    # KST 07:00 = UTC 22:00 (previous day)
    kst_offset = timedelta(hours=9)
    now_kst = now + kst_offset

    # Calculate today's KST 07:00 in UTC
    today_7am_kst = now_kst.replace(hour=7, minute=0, second=0, microsecond=0)
    if now_kst.hour < 7:
        today_7am_kst -= timedelta(days=1)
    today_7am_utc = today_7am_kst - kst_offset

    # Cache is valid if it was created after today's 7AM KST
    return cached_at.replace(tzinfo=timezone.utc) >= today_7am_utc


@router.get("/daily/{category}", response_model=DailyPapersResponse, summary="Get today's papers")
async def get_daily_papers(
    category: str,
    limit: int = Query(default=10, ge=1, le=30, description="Number of papers"),
    no_cache: bool = Query(default=False, description="Bypass cache")
):
    """
    Get papers published today (or within last 24 hours).

    Uses PubMed's entrez_date (date added to PubMed) or pdat (publication date).
    Cache refreshes at KST 07:00 daily.

    Categories: oncology, immunotherapy, gene_therapy, neurology, infectious_disease, ai_medicine, genomics, drug_discovery
    """
    from datetime import timezone

    if category not in TRENDING_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Available: {list(TRENDING_CATEGORIES.keys())}"
        )

    cache_key = f"daily_{category}_{limit}"
    today = datetime.now().strftime("%Y-%m-%d")

    # Check cache
    if not no_cache and _is_cache_valid(cache_key, _daily_cache_time):
        cached = _daily_cache[cache_key]
        return DailyPapersResponse(
            category=category,
            date=cached["date"],
            papers=cached["papers"],
            total=len(cached["papers"]),
            cached=True
        )

    try:
        # Build query for today's papers using base query (without year filter)
        # Use last 7 days to ensure we get enough papers (PubMed indexing can be delayed)
        base_query = TRENDING_CATEGORY_BASES[category]

        # Get current date for filtering
        now = datetime.now()
        min_date = (now - timedelta(days=7)).strftime("%Y/%m/%d")
        max_date = now.strftime("%Y/%m/%d")

        papers = await crawler_agent.search_pubmed(
            query=base_query,
            max_results=limit * 3,  # Fetch more to filter
            sort="pub_date",  # Most recent first
            min_date=min_date,
            max_date=max_date
        )

        # Filter to major journals for quality
        filtered_papers = []
        for p in papers:
            # Check if from major journal
            is_major = any(mj.lower() in p.journal.lower() for mj in MAJOR_JOURNALS[:20])
            if is_major or len(filtered_papers) < limit // 2:
                filtered_papers.append(p)
            if len(filtered_papers) >= limit:
                break

        paper_responses = [paper_to_response(p) for p in filtered_papers[:limit]]

        # Update cache
        _daily_cache[cache_key] = {
            "date": today,
            "papers": paper_responses
        }
        _daily_cache_time[cache_key] = datetime.now(timezone.utc)

        return DailyPapersResponse(
            category=category,
            date=today,
            papers=paper_responses,
            total=len(paper_responses),
            cached=False
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching daily papers: {str(e)}")



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


# ==================== Full Text Retrieval with Chat Session ====================

class FullTextRequest(BaseModel):
    pmid: Optional[str] = Field(default=None, description="PubMed ID")
    pmcid: Optional[str] = Field(default=None, description="PMC ID (e.g., PMC1234567)")
    doi: Optional[str] = Field(default=None, description="DOI")
    title: str = Field(..., description="Paper title for session")
    url: Optional[str] = Field(default=None, description="Publisher URL for Playwright fallback")
    language: Optional[str] = Field(default="en", description="Output language for AI summary: 'en' or 'ko'")


class FullTextSummary(BaseModel):
    summary: str = ""
    key_findings: List[str] = []
    methodology: str = ""


class FullTextResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    paper_title: str
    full_text_preview: str = ""
    full_text_length: int = 0
    chunks_created: int = 0
    source: str = ""  # "pmc", "europe_pmc", "unpaywall", "publisher"
    ai_summary: Optional[FullTextSummary] = None
    error: Optional[str] = None


@router.post("/full-text", response_model=FullTextResponse, summary="Get full text and create chat session")
async def get_full_text(request: FullTextRequest):
    """
    Fetch full text of a paper and create a chat session for Q&A.

    Tries in order:
    1. PMC Open Access API (if PMCID available)
    2. Europe PMC (if PMID or PMCID available)
    3. Playwright deep crawl (if URL provided and Playwright available)

    Returns a session_id that can be used with the paper chat API.
    """
    from backend.app.core.paper_agent import create_paper_session, get_paper_agent
    from backend.app.core.text_splitter import BioPaperSplitter

    full_text = None
    source = ""

    try:
        # Try all methods via fetch_full_text (PMC, Europe PMC, Unpaywall, PubMed links)
        if request.pmid or request.pmcid or request.doi:
            full_text = await crawler_agent.fetch_full_text(
                pmid=request.pmid,
                pmcid=request.pmcid,
                doi=request.doi,
                use_playwright=PLAYWRIGHT_AVAILABLE
            )
            if full_text:
                # Determine source based on content/method used
                if request.pmcid:
                    source = "pmc"
                elif "Unpaywall" in str(full_text)[:100]:
                    source = "unpaywall"
                else:
                    source = "europe_pmc"

        # Fallback: Try Playwright directly with URL
        if not full_text and request.url and PLAYWRIGHT_AVAILABLE:
            crawler = None
            try:
                crawler = PlaywrightDeepCrawler(headless=True)
                result = await crawler.extract_full_text(request.url)
                if result.success and result.full_text:
                    full_text = result.full_text
                    source = "publisher"
            except Exception as e:
                print(f"Playwright crawl failed: {e}")
            finally:
                if crawler:
                    await crawler.close()

        if not full_text:
            return FullTextResponse(
                success=False,
                paper_title=request.title,
                error="Full text not available. Paper may not be open access."
            )

        # Create paper chat session
        session_id = create_paper_session(request.title)
        agent = get_paper_agent(session_id)

        if not agent:
            return FullTextResponse(
                success=False,
                paper_title=request.title,
                error="Failed to create chat session"
            )

        # Split full text into chunks
        splitter = BioPaperSplitter()
        chunks = splitter.split_text_simple(
            full_text,
            metadata={
                "paper_title": request.title,
                "pmid": request.pmid or "",
                "pmcid": request.pmcid or "",
                "doi": request.doi or "",
                "source": source
            }
        )

        # Add chunks to session
        chunks_added = agent.add_chunks(chunks)

        # Generate AI summary from the indexed content
        ai_summary = None
        try:
            summary_result = agent.summarize(language=request.language or "en")
            ai_summary = FullTextSummary(
                summary=summary_result.get("summary", ""),
                key_findings=summary_result.get("key_findings", []),
                methodology=summary_result.get("methodology", "")
            )
        except Exception as e:
            print(f"Summary generation failed: {e}")

        return FullTextResponse(
            success=True,
            session_id=session_id,
            paper_title=request.title,
            full_text_preview=full_text[:500] + "..." if len(full_text) > 500 else full_text,
            full_text_length=len(full_text),
            chunks_created=chunks_added,
            source=source,
            ai_summary=ai_summary
        )

    except Exception as e:
        return FullTextResponse(
            success=False,
            paper_title=request.title,
            error=str(e)
        )
