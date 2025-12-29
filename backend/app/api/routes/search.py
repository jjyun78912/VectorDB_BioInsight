"""
Search API endpoints - Enhanced for paper discovery and similarity.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

router = APIRouter()


class SearchResult(BaseModel):
    """Search result model."""
    content: str
    relevance_score: float
    paper_title: str
    section: str
    pmid: Optional[str] = None
    year: Optional[str] = None
    doi: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    domain: str
    results: List[SearchResult]
    total: int


class PaperSearchResult(BaseModel):
    """Paper-level search result."""
    pmid: str
    title: str
    abstract: str
    year: Optional[str] = None
    journal: Optional[str] = None
    authors: List[str] = []
    keywords: List[str] = []
    relevance_score: float
    has_fulltext: bool = False


class PaperSearchResponse(BaseModel):
    """Paper search response."""
    query: str
    domain: str
    papers: List[PaperSearchResult]
    total: int


class SimilarPaper(BaseModel):
    """Similar paper model."""
    pmid: str
    title: str
    abstract: str = ""
    similarity_score: float
    common_keywords: List[str] = []
    year: Optional[str] = None
    doi: Optional[str] = None
    keywords: str = ""


class PaperCoordinate(BaseModel):
    """Paper coordinate for visualization."""
    pmid: str
    title: str
    x: float
    y: float
    similarity_score: float


class VisualizationData(BaseModel):
    """Visualization data for galaxy view."""
    source: Dict
    papers: List[PaperCoordinate]


class SimilarPapersResponse(BaseModel):
    """Similar papers response."""
    source_paper: str
    source_pmid: str
    similar_papers: List[SimilarPaper]
    total: int
    visualization: Optional[VisualizationData] = None


# Available disease domains
DISEASE_DOMAINS = [
    {"key": "pancreatic_cancer", "name": "Pancreatic Cancer", "kr_name": "췌장암"},
    {"key": "blood_cancer", "name": "Blood Cancer", "kr_name": "혈액암"},
    {"key": "glioblastoma", "name": "Glioblastoma", "kr_name": "교모세포종"},
    {"key": "alzheimer", "name": "Alzheimer's Disease", "kr_name": "알츠하이머"},
    {"key": "pcos", "name": "Polycystic Ovary Syndrome", "kr_name": "다낭성난소증후군"},
]


@router.get("/domains")
async def list_domains():
    """List available disease domains."""
    return {"domains": DISEASE_DOMAINS}


@router.get("/", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="Search query"),
    domain: str = Query("pancreatic_cancer", description="Disease domain"),
    section: Optional[str] = Query(None, description="Filter by section"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results")
):
    """
    Search the vector database for relevant content chunks.
    """
    try:
        from src.vector_store import create_vector_store

        vector_store = create_vector_store(disease_domain=domain)

        if vector_store.count == 0:
            return SearchResponse(
                query=query,
                domain=domain,
                results=[],
                total=0
            )

        if section:
            results = vector_store.search_by_section(query, section, top_k=top_k)
        else:
            results = vector_store.search(query, top_k=top_k)

        search_results = [
            SearchResult(
                content=r.content[:500],
                relevance_score=r.relevance_score,
                paper_title=r.metadata.get("paper_title", "Unknown"),
                section=r.metadata.get("section", "Unknown"),
                pmid=r.metadata.get("pmid"),
                year=r.metadata.get("year"),
                doi=r.metadata.get("doi")
            )
            for r in results
        ]

        return SearchResponse(
            query=query,
            domain=domain,
            results=search_results,
            total=len(search_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers", response_model=PaperSearchResponse)
async def search_papers(
    query: str = Query(..., description="Search query (keyword or topic)"),
    domain: str = Query("pancreatic_cancer", description="Disease domain"),
    top_k: int = Query(10, ge=1, le=50, description="Number of papers")
):
    """
    Search for papers by keyword - returns paper-level results.
    Aggregates chunk results to paper level.
    """
    try:
        from src.vector_store import create_vector_store
        from src.config import PAPERS_DIR

        vector_store = create_vector_store(disease_domain=domain)

        if vector_store.count == 0:
            return PaperSearchResponse(
                query=query,
                domain=domain,
                papers=[],
                total=0
            )

        # Search with more results to aggregate by paper
        results = vector_store.search(query, top_k=top_k * 3)

        # Aggregate by paper
        paper_scores: Dict[str, Dict] = {}

        for r in results:
            pmid = r.metadata.get("pmid", "")
            title = r.metadata.get("paper_title", "Unknown")

            if pmid not in paper_scores:
                paper_scores[pmid] = {
                    "pmid": pmid,
                    "title": title,
                    "max_score": r.relevance_score,
                    "year": r.metadata.get("year"),
                    "doi": r.metadata.get("doi"),
                    "journal": r.metadata.get("journal", ""),
                    "authors": [],
                    "keywords": [],
                    "abstract": "",
                    "has_fulltext": False
                }
            else:
                paper_scores[pmid]["max_score"] = max(
                    paper_scores[pmid]["max_score"],
                    r.relevance_score
                )

        # Load full paper info from JSON files
        papers_dir = PAPERS_DIR / domain

        for pmid, paper_data in paper_scores.items():
            json_file = papers_dir / f"{pmid}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        full_data = json.load(f)
                        paper_data["abstract"] = full_data.get("abstract", "")[:500]
                        paper_data["authors"] = full_data.get("authors", [])[:5]
                        paper_data["keywords"] = full_data.get("keywords", [])[:10]
                        paper_data["journal"] = full_data.get("journal", "")
                        paper_data["has_fulltext"] = bool(full_data.get("full_text"))
                except:
                    pass

        # Sort by score and take top_k
        sorted_papers = sorted(
            paper_scores.values(),
            key=lambda x: x["max_score"],
            reverse=True
        )[:top_k]

        paper_results = [
            PaperSearchResult(
                pmid=p["pmid"],
                title=p["title"],
                abstract=p["abstract"],
                year=p["year"],
                journal=p["journal"],
                authors=p["authors"],
                keywords=p["keywords"],
                relevance_score=p["max_score"],
                has_fulltext=p["has_fulltext"]
            )
            for p in sorted_papers
        ]

        return PaperSearchResponse(
            query=query,
            domain=domain,
            papers=paper_results,
            total=len(paper_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{pmid}", response_model=SimilarPapersResponse)
async def find_similar_papers(
    pmid: str,
    domain: str = Query("pancreatic_cancer", description="Disease domain"),
    top_k: int = Query(5, ge=1, le=20, description="Number of similar papers"),
    include_visualization: bool = Query(True, description="Include 2D coordinates for visualization")
):
    """
    Find papers similar to a given paper.
    Uses embedding similarity from vector store with t-SNE visualization coordinates.
    """
    try:
        from src.vector_store import create_vector_store
        from src.config import PAPERS_DIR

        # First try vector store (faster, uses indexed embeddings)
        vector_store = create_vector_store(disease_domain=domain)

        if vector_store.count > 0:
            result = vector_store.find_similar_papers(
                source_pmid=pmid,
                top_k=top_k,
                include_coordinates=include_visualization
            )

            if "error" not in result:
                # Get source paper title from JSON
                papers_dir = PAPERS_DIR / domain
                source_file = papers_dir / f"{pmid}.json"
                source_title = pmid
                source_keywords = set()

                if source_file.exists():
                    try:
                        with open(source_file, 'r', encoding='utf-8') as f:
                            source_paper = json.load(f)
                            source_title = source_paper.get("title", pmid)
                            source_keywords = set(source_paper.get("keywords", []))
                    except:
                        pass

                # Enrich similar papers with abstracts from JSON files
                enriched_papers = []
                for paper in result.get("similar_papers", []):
                    paper_file = papers_dir / f"{paper['pmid']}.json"
                    abstract = ""
                    common_keywords = []

                    if paper_file.exists():
                        try:
                            with open(paper_file, 'r', encoding='utf-8') as f:
                                paper_data = json.load(f)
                                abstract = paper_data.get("abstract", "")[:300]
                                other_keywords = set(paper_data.get("keywords", []))
                                common_keywords = list(source_keywords & other_keywords)[:5]
                        except:
                            pass

                    enriched_papers.append(SimilarPaper(
                        pmid=paper["pmid"],
                        title=paper["title"],
                        abstract=abstract,
                        similarity_score=paper["similarity_score"],
                        common_keywords=common_keywords,
                        year=paper.get("year"),
                        doi=paper.get("doi"),
                        keywords=paper.get("keywords", "")
                    ))

                # Build visualization data
                visualization = None
                if include_visualization and "visualization" in result:
                    vis_data = result["visualization"]
                    visualization = VisualizationData(
                        source=vis_data["source"],
                        papers=[PaperCoordinate(**p) for p in vis_data["papers"]]
                    )

                return SimilarPapersResponse(
                    source_paper=source_title,
                    source_pmid=pmid,
                    similar_papers=enriched_papers,
                    total=len(enriched_papers),
                    visualization=visualization
                )

        # Fallback: JSON file-based comparison (for papers not in vector store)
        return await _fallback_similar_papers(pmid, domain, top_k)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _fallback_similar_papers(pmid: str, domain: str, top_k: int) -> SimilarPapersResponse:
    """Fallback method using JSON files and live embedding calculation."""
    from src.config import PAPERS_DIR
    from src.embeddings import get_embedder
    import numpy as np

    papers_dir = PAPERS_DIR / domain
    source_file = papers_dir / f"{pmid}.json"

    if not source_file.exists():
        raise HTTPException(status_code=404, detail=f"Paper {pmid} not found")

    with open(source_file, 'r', encoding='utf-8') as f:
        source_paper = json.load(f)

    source_abstract = source_paper.get("abstract", "")
    source_keywords = set(source_paper.get("keywords", []))

    if not source_abstract:
        raise HTTPException(status_code=400, detail="Source paper has no abstract")

    # Get embeddings
    embedder = get_embedder()
    source_embedding = np.array(embedder.embed_text(source_abstract))

    # Compare with all other papers
    similar_papers = []

    for json_file in papers_dir.glob("*.json"):
        if json_file.name.startswith("_") or json_file.stem == pmid:
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                other_paper = json.load(f)

            other_abstract = other_paper.get("abstract", "")
            if not other_abstract:
                continue

            # Calculate similarity
            other_embedding = np.array(embedder.embed_text(other_abstract))
            similarity = np.dot(source_embedding, other_embedding) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(other_embedding)
            )

            # Find common keywords
            other_keywords = set(other_paper.get("keywords", []))
            common = list(source_keywords & other_keywords)

            similar_papers.append({
                "pmid": other_paper.get("pmid", json_file.stem),
                "title": other_paper.get("title", "Unknown"),
                "abstract": other_abstract[:300],
                "similarity_score": float(similarity) * 100,
                "common_keywords": common[:5],
                "year": other_paper.get("year")
            })

        except Exception:
            continue

    # Sort by similarity
    similar_papers.sort(key=lambda x: x["similarity_score"], reverse=True)
    top_similar = similar_papers[:top_k]

    return SimilarPapersResponse(
        source_paper=source_paper.get("title", pmid),
        source_pmid=pmid,
        similar_papers=[SimilarPaper(**p) for p in top_similar],
        total=len(top_similar),
        visualization=None
    )


@router.get("/papers/all")
async def list_all_papers(
    domain: str = Query("pancreatic_cancer", description="Disease domain"),
    limit: int = Query(50, ge=1, le=200, description="Max papers to return")
):
    """
    List all papers in a domain with basic info.
    """
    try:
        from src.config import PAPERS_DIR

        papers_dir = PAPERS_DIR / domain

        if not papers_dir.exists():
            return {"papers": [], "total": 0, "domain": domain}

        papers = []

        for json_file in papers_dir.glob("*.json"):
            if json_file.name.startswith("_"):
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)

                papers.append({
                    "pmid": paper.get("pmid", json_file.stem),
                    "title": paper.get("title", "Unknown"),
                    "year": paper.get("year"),
                    "journal": paper.get("journal", ""),
                    "authors": paper.get("authors", [])[:3],
                    "keywords": paper.get("keywords", [])[:5],
                    "has_abstract": bool(paper.get("abstract")),
                    "has_fulltext": bool(paper.get("full_text"))
                })

            except Exception:
                continue

        # Sort by year (newest first)
        papers.sort(key=lambda x: x.get("year") or "0000", reverse=True)

        return {
            "papers": papers[:limit],
            "total": len(papers),
            "domain": domain
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper/{pmid}")
async def get_paper_detail(
    pmid: str,
    domain: str = Query("pancreatic_cancer", description="Disease domain")
):
    """
    Get detailed information about a specific paper.
    """
    try:
        from src.config import PAPERS_DIR

        papers_dir = PAPERS_DIR / domain
        paper_file = papers_dir / f"{pmid}.json"

        if not paper_file.exists():
            raise HTTPException(status_code=404, detail=f"Paper {pmid} not found in {domain}")

        with open(paper_file, 'r', encoding='utf-8') as f:
            paper = json.load(f)

        return {
            "pmid": paper.get("pmid"),
            "pmcid": paper.get("pmcid"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "authors": paper.get("authors", []),
            "journal": paper.get("journal"),
            "year": paper.get("year"),
            "doi": paper.get("doi"),
            "keywords": paper.get("keywords", []),
            "has_fulltext": bool(paper.get("full_text")),
            "fulltext_preview": paper.get("full_text", "")[:1000] if paper.get("full_text") else None,
            "disease_domain": domain
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
