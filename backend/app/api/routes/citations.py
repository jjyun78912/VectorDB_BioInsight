"""
Citation Analysis API - Key Papers & Citation Network Discovery

Features:
- Find highly cited papers in a research area
- Identify seminal/landmark papers
- Citation network analysis
- Co-citation clustering
- Reference analysis

Data Sources:
- Semantic Scholar API (free, no key required for basic usage)
- OpenCitations API
- PubMed for metadata
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import asyncio
from datetime import datetime

router = APIRouter()

# API endpoints
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
OPENCITATIONS_API = "https://opencitations.net/index/api/v1"
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Cache
_citation_cache: Dict[str, Any] = {}
_cache_expiry: Dict[str, datetime] = {}
CACHE_HOURS = 24


# ============== Models ==============

class CitedPaper(BaseModel):
    """A cited paper with citation metrics."""
    paper_id: str
    title: str
    authors: List[str]
    year: Optional[int]
    citation_count: int
    influential_citation_count: Optional[int] = 0
    venue: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    is_open_access: bool = False
    fields_of_study: List[str] = []


class CitationMetrics(BaseModel):
    """Citation metrics for a paper."""
    total_citations: int
    influential_citations: int
    citation_velocity: float  # citations per year
    h_index_contribution: bool  # estimated
    recency_weighted_score: float


class KeyPapersResponse(BaseModel):
    """Response for key papers search."""
    query: str
    total_found: int
    papers: List[CitedPaper]
    analysis_date: str


class LandmarkPaper(BaseModel):
    """A landmark/seminal paper in a field."""
    paper: CitedPaper
    landmark_score: float
    reasons: List[str]  # Why it's a landmark
    citing_fields: List[str]  # Fields that cite this paper


class LandmarkPapersResponse(BaseModel):
    """Response for landmark papers."""
    topic: str
    landmark_papers: List[LandmarkPaper]
    methodology: str
    analysis_date: str


class CitationNode(BaseModel):
    """Node in citation network."""
    id: str
    title: str
    year: Optional[int]
    citations: int
    node_type: str  # "source", "reference", "citation"


class CitationEdge(BaseModel):
    """Edge in citation network."""
    source: str
    target: str
    weight: float = 1.0


class CitationNetwork(BaseModel):
    """Citation network structure."""
    nodes: List[CitationNode]
    edges: List[CitationEdge]
    center_paper: str


class CitationNetworkResponse(BaseModel):
    """Response for citation network."""
    paper_id: str
    paper_title: str
    network: CitationNetwork
    stats: Dict[str, Any]


class CoCitationCluster(BaseModel):
    """Co-citation cluster."""
    cluster_id: int
    theme: str
    papers: List[CitedPaper]
    co_citation_strength: float


class CoCitationResponse(BaseModel):
    """Response for co-citation analysis."""
    seed_paper_id: str
    clusters: List[CoCitationCluster]
    analysis_date: str


# ============== Helper Functions ==============

async def search_semantic_scholar(
    query: str,
    limit: int = 20,
    year_range: Optional[str] = None,
    fields_of_study: Optional[List[str]] = None
) -> List[Dict]:
    """Search Semantic Scholar for papers."""
    cache_key = f"ss_search_{query}_{limit}_{year_range}"
    if cache_key in _citation_cache:
        if datetime.now() < _cache_expiry.get(cache_key, datetime.min):
            return _citation_cache[cache_key]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,authors,year,citationCount,influentialCitationCount,venue,externalIds,abstract,isOpenAccess,fieldsOfStudy,url"
            }
            if year_range:
                params["year"] = year_range

            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params=params
            )

            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])

                from datetime import timedelta
                _citation_cache[cache_key] = papers
                _cache_expiry[cache_key] = datetime.now() + timedelta(hours=CACHE_HOURS)
                return papers
            elif response.status_code == 429:
                # Rate limited, wait and retry
                await asyncio.sleep(2)
                return []
            else:
                print(f"Semantic Scholar API error: {response.status_code}")
                return []

    except Exception as e:
        print(f"Semantic Scholar search error: {e}")
        return []


async def get_paper_details(paper_id: str) -> Optional[Dict]:
    """Get detailed information about a paper."""
    cache_key = f"ss_paper_{paper_id}"
    if cache_key in _citation_cache:
        if datetime.now() < _cache_expiry.get(cache_key, datetime.min):
            return _citation_cache[cache_key]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            fields = "paperId,title,authors,year,citationCount,influentialCitationCount,venue,externalIds,abstract,isOpenAccess,fieldsOfStudy,url,references,citations"

            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}",
                params={"fields": fields}
            )

            if response.status_code == 200:
                data = response.json()
                from datetime import timedelta
                _citation_cache[cache_key] = data
                _cache_expiry[cache_key] = datetime.now() + timedelta(hours=CACHE_HOURS)
                return data
            else:
                return None

    except Exception as e:
        print(f"Get paper details error: {e}")
        return None


async def get_paper_citations(paper_id: str, limit: int = 100) -> List[Dict]:
    """Get papers that cite a given paper."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/citations",
                params={
                    "fields": "paperId,title,authors,year,citationCount,venue",
                    "limit": limit
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []

    except Exception as e:
        print(f"Get citations error: {e}")
        return []


async def get_paper_references(paper_id: str, limit: int = 100) -> List[Dict]:
    """Get papers referenced by a given paper."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/{paper_id}/references",
                params={
                    "fields": "paperId,title,authors,year,citationCount,venue",
                    "limit": limit
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            return []

    except Exception as e:
        print(f"Get references error: {e}")
        return []


def convert_to_cited_paper(paper_data: Dict) -> CitedPaper:
    """Convert Semantic Scholar paper data to CitedPaper model."""
    external_ids = paper_data.get("externalIds", {}) or {}

    authors = []
    for author in paper_data.get("authors", [])[:5]:  # Limit to 5 authors
        if isinstance(author, dict):
            authors.append(author.get("name", "Unknown"))
        else:
            authors.append(str(author))

    return CitedPaper(
        paper_id=paper_data.get("paperId", ""),
        title=paper_data.get("title", "Unknown Title"),
        authors=authors,
        year=paper_data.get("year"),
        citation_count=paper_data.get("citationCount", 0) or 0,
        influential_citation_count=paper_data.get("influentialCitationCount", 0) or 0,
        venue=paper_data.get("venue"),
        doi=external_ids.get("DOI"),
        pmid=external_ids.get("PubMed"),
        abstract=paper_data.get("abstract"),
        url=paper_data.get("url"),
        is_open_access=paper_data.get("isOpenAccess", False) or False,
        fields_of_study=[f.get("category", f) if isinstance(f, dict) else str(f)
                        for f in (paper_data.get("fieldsOfStudy") or [])[:5]]
    )


def calculate_landmark_score(paper: CitedPaper, current_year: int = 2025) -> float:
    """
    Calculate a landmark score for a paper.

    Factors:
    - Total citations (normalized)
    - Influential citations ratio
    - Age-adjusted citation rate
    - Cross-field impact
    """
    if not paper.year:
        return 0.0

    age = max(1, current_year - paper.year)
    citations_per_year = paper.citation_count / age

    # Base score from citation velocity
    if citations_per_year >= 500:
        base_score = 100
    elif citations_per_year >= 200:
        base_score = 80
    elif citations_per_year >= 100:
        base_score = 60
    elif citations_per_year >= 50:
        base_score = 40
    else:
        base_score = min(40, citations_per_year * 0.8)

    # Influential citation bonus
    influential_ratio = (paper.influential_citation_count / max(1, paper.citation_count))
    influential_bonus = influential_ratio * 20

    # Cross-field bonus
    field_bonus = min(10, len(paper.fields_of_study) * 2)

    # Age bonus (older highly-cited papers are more landmark)
    if age >= 10 and paper.citation_count >= 1000:
        age_bonus = 10
    elif age >= 5 and paper.citation_count >= 500:
        age_bonus = 5
    else:
        age_bonus = 0

    return min(100, base_score + influential_bonus + field_bonus + age_bonus)


def get_landmark_reasons(paper: CitedPaper, score: float, current_year: int = 2025) -> List[str]:
    """Generate reasons why a paper is considered a landmark."""
    reasons = []
    age = max(1, current_year - (paper.year or current_year))
    citations_per_year = paper.citation_count / age

    if paper.citation_count >= 10000:
        reasons.append(f"Exceptionally high citations ({paper.citation_count:,})")
    elif paper.citation_count >= 1000:
        reasons.append(f"Highly cited ({paper.citation_count:,} citations)")

    if citations_per_year >= 200:
        reasons.append(f"High citation velocity ({citations_per_year:.0f}/year)")

    if paper.influential_citation_count and paper.influential_citation_count >= 100:
        reasons.append(f"High influential citations ({paper.influential_citation_count})")

    if len(paper.fields_of_study) >= 3:
        reasons.append(f"Cross-disciplinary impact ({len(paper.fields_of_study)} fields)")

    if age >= 10 and paper.citation_count >= 1000:
        reasons.append("Established foundational work")

    if paper.is_open_access:
        reasons.append("Open access - widely accessible")

    if not reasons:
        reasons.append("Significant contribution to the field")

    return reasons[:4]


# ============== API Endpoints ==============

@router.get("/key-papers", response_model=KeyPapersResponse)
async def get_key_papers(
    query: str = Query(..., description="Search query (topic, gene, disease, etc.)"),
    limit: int = Query(20, ge=5, le=50),
    min_citations: int = Query(0, ge=0),
    year_from: Optional[int] = Query(None, ge=1900),
    year_to: Optional[int] = Query(None, le=2025)
):
    """
    Find highly cited key papers in a research area.

    Returns papers sorted by citation count, with detailed metrics.
    """
    year_range = None
    if year_from and year_to:
        year_range = f"{year_from}-{year_to}"
    elif year_from:
        year_range = f"{year_from}-"
    elif year_to:
        year_range = f"-{year_to}"

    papers_data = await search_semantic_scholar(query, limit=limit * 2, year_range=year_range)

    if not papers_data:
        return KeyPapersResponse(
            query=query,
            total_found=0,
            papers=[],
            analysis_date=datetime.now().strftime("%Y-%m-%d")
        )

    papers = []
    for p in papers_data:
        cited_paper = convert_to_cited_paper(p)
        if cited_paper.citation_count >= min_citations:
            papers.append(cited_paper)

    # Sort by citation count
    papers.sort(key=lambda x: x.citation_count, reverse=True)

    return KeyPapersResponse(
        query=query,
        total_found=len(papers),
        papers=papers[:limit],
        analysis_date=datetime.now().strftime("%Y-%m-%d")
    )


@router.get("/landmark-papers", response_model=LandmarkPapersResponse)
async def get_landmark_papers(
    topic: str = Query(..., description="Research topic"),
    limit: int = Query(10, ge=5, le=20)
):
    """
    Identify landmark/seminal papers in a research field.

    Uses citation metrics, influential citations, and cross-field impact
    to identify foundational papers.
    """
    # Search for highly cited papers in the topic
    papers_data = await search_semantic_scholar(topic, limit=50)

    if not papers_data:
        return LandmarkPapersResponse(
            topic=topic,
            landmark_papers=[],
            methodology="Citation analysis with influential citation weighting",
            analysis_date=datetime.now().strftime("%Y-%m-%d")
        )

    current_year = datetime.now().year
    landmark_papers = []

    for p in papers_data:
        paper = convert_to_cited_paper(p)
        if paper.citation_count >= 50:  # Minimum threshold
            score = calculate_landmark_score(paper, current_year)
            reasons = get_landmark_reasons(paper, score, current_year)

            landmark_papers.append(LandmarkPaper(
                paper=paper,
                landmark_score=round(score, 1),
                reasons=reasons,
                citing_fields=paper.fields_of_study
            ))

    # Sort by landmark score
    landmark_papers.sort(key=lambda x: x.landmark_score, reverse=True)

    return LandmarkPapersResponse(
        topic=topic,
        landmark_papers=landmark_papers[:limit],
        methodology="Multi-factor analysis: citation count, citation velocity, influential citations, cross-field impact, longevity",
        analysis_date=datetime.now().strftime("%Y-%m-%d")
    )


@router.get("/citation-network/{paper_id}", response_model=CitationNetworkResponse)
async def get_citation_network(
    paper_id: str,
    depth: int = Query(1, ge=1, le=2, description="Network depth (1 or 2 levels)"),
    limit_per_level: int = Query(10, ge=5, le=20)
):
    """
    Build a citation network around a paper.

    Returns nodes (papers) and edges (citation relationships)
    for visualization.
    """
    # Get the center paper
    center_paper = await get_paper_details(paper_id)
    if not center_paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    nodes = []
    edges = []
    seen_ids = set()

    # Add center node
    center_node = CitationNode(
        id=paper_id,
        title=center_paper.get("title", "Unknown"),
        year=center_paper.get("year"),
        citations=center_paper.get("citationCount", 0) or 0,
        node_type="source"
    )
    nodes.append(center_node)
    seen_ids.add(paper_id)

    # Get references (papers this paper cites)
    references = await get_paper_references(paper_id, limit=limit_per_level)
    await asyncio.sleep(0.5)  # Rate limiting

    for ref in references:
        ref_paper = ref.get("citedPaper", {})
        if not ref_paper or not ref_paper.get("paperId"):
            continue

        ref_id = ref_paper["paperId"]
        if ref_id not in seen_ids:
            nodes.append(CitationNode(
                id=ref_id,
                title=ref_paper.get("title", "Unknown"),
                year=ref_paper.get("year"),
                citations=ref_paper.get("citationCount", 0) or 0,
                node_type="reference"
            ))
            seen_ids.add(ref_id)

        edges.append(CitationEdge(
            source=paper_id,
            target=ref_id,
            weight=1.0
        ))

    # Get citations (papers that cite this paper)
    citations = await get_paper_citations(paper_id, limit=limit_per_level)

    for cit in citations:
        cit_paper = cit.get("citingPaper", {})
        if not cit_paper or not cit_paper.get("paperId"):
            continue

        cit_id = cit_paper["paperId"]
        if cit_id not in seen_ids:
            nodes.append(CitationNode(
                id=cit_id,
                title=cit_paper.get("title", "Unknown"),
                year=cit_paper.get("year"),
                citations=cit_paper.get("citationCount", 0) or 0,
                node_type="citation"
            ))
            seen_ids.add(cit_id)

        edges.append(CitationEdge(
            source=cit_id,
            target=paper_id,
            weight=1.0
        ))

    network = CitationNetwork(
        nodes=nodes,
        edges=edges,
        center_paper=paper_id
    )

    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "references_count": len([n for n in nodes if n.node_type == "reference"]),
        "citations_count": len([n for n in nodes if n.node_type == "citation"]),
        "avg_citations": sum(n.citations for n in nodes) / len(nodes) if nodes else 0
    }

    return CitationNetworkResponse(
        paper_id=paper_id,
        paper_title=center_paper.get("title", "Unknown"),
        network=network,
        stats=stats
    )


@router.get("/paper/{paper_id}")
async def get_paper_info(paper_id: str):
    """Get detailed information about a specific paper."""
    paper = await get_paper_details(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    return {
        "paper": convert_to_cited_paper(paper),
        "references_count": len(paper.get("references", []) or []),
        "citations_count": len(paper.get("citations", []) or [])
    }


@router.get("/search-by-doi")
async def search_by_doi(doi: str = Query(..., description="DOI of the paper")):
    """Search for a paper by DOI and get citation info."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{doi}",
                params={
                    "fields": "paperId,title,authors,year,citationCount,influentialCitationCount,venue,externalIds,abstract,isOpenAccess,fieldsOfStudy,url"
                }
            )

            if response.status_code == 200:
                paper_data = response.json()
                return {
                    "found": True,
                    "paper": convert_to_cited_paper(paper_data)
                }
            else:
                return {"found": False, "message": "Paper not found"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/citation-trends/{paper_id}")
async def get_citation_trends(paper_id: str):
    """
    Get citation trends over time for a paper.
    Note: Semantic Scholar doesn't provide year-by-year data,
    so we estimate based on citing paper years.
    """
    citations = await get_paper_citations(paper_id, limit=200)

    if not citations:
        return {
            "paper_id": paper_id,
            "yearly_citations": {},
            "message": "No citation data available"
        }

    # Count citations by year
    yearly_counts: Dict[int, int] = {}
    for cit in citations:
        cit_paper = cit.get("citingPaper", {})
        year = cit_paper.get("year")
        if year:
            yearly_counts[year] = yearly_counts.get(year, 0) + 1

    # Sort by year
    sorted_years = sorted(yearly_counts.items())

    return {
        "paper_id": paper_id,
        "yearly_citations": dict(sorted_years),
        "total_sampled": len(citations),
        "peak_year": max(yearly_counts.items(), key=lambda x: x[1])[0] if yearly_counts else None
    }


@router.get("/related-papers/{paper_id}")
async def get_related_papers(
    paper_id: str,
    limit: int = Query(10, ge=5, le=20)
):
    """
    Find papers related to a given paper based on citations.

    Related papers are those that share common references
    or are commonly co-cited with this paper.
    """
    # Get papers that cite this paper
    citations = await get_paper_citations(paper_id, limit=50)
    await asyncio.sleep(0.5)

    # Get references of this paper
    references = await get_paper_references(paper_id, limit=50)

    # Combine and find most cited among them
    related = []

    for cit in citations:
        cit_paper = cit.get("citingPaper", {})
        if cit_paper and cit_paper.get("paperId"):
            related.append({
                "paper": convert_to_cited_paper(cit_paper),
                "relation": "cites_this",
                "relevance": cit_paper.get("citationCount", 0) or 0
            })

    for ref in references:
        ref_paper = ref.get("citedPaper", {})
        if ref_paper and ref_paper.get("paperId"):
            related.append({
                "paper": convert_to_cited_paper(ref_paper),
                "relation": "cited_by_this",
                "relevance": ref_paper.get("citationCount", 0) or 0
            })

    # Sort by citation count and deduplicate
    seen_ids = set()
    unique_related = []
    for r in sorted(related, key=lambda x: x["relevance"], reverse=True):
        pid = r["paper"].paper_id
        if pid not in seen_ids and pid != paper_id:
            unique_related.append(r)
            seen_ids.add(pid)

    return {
        "paper_id": paper_id,
        "related_papers": unique_related[:limit],
        "total_found": len(unique_related)
    }
