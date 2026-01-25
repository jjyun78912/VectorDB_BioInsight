"""
Paper Recommender for RNA-seq Analysis Reports.

Recommends relevant papers based on:
1. Cancer type
2. Top hub genes
3. Key pathways
4. Citation-based quality filtering (Classic vs Breakthrough papers)

Uses PubMed E-utilities API for real-time search and
Semantic Scholar API for citation data.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Semantic Scholar API endpoint
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"

# High-impact journals for fallback quality estimation
HIGH_IMPACT_JOURNALS = {
    "Nature", "Science", "Cell", "Nature Medicine", "Cancer Cell",
    "Cancer Discovery", "Nature Genetics", "The Lancet Oncology",
    "Journal of Clinical Oncology", "Nature Reviews Cancer",
    "Cancer Research", "Clinical Cancer Research", "Gastroenterology",
    "Hepatology", "Gut", "JAMA Oncology", "Annals of Oncology"
}

# Field-specific citation thresholds for quality filtering
FIELD_CITATION_BASELINES = {
    "BRCA": {"classic_threshold": 80, "breakthrough_velocity": 25},
    "breast_cancer": {"classic_threshold": 80, "breakthrough_velocity": 25},
    "LUAD": {"classic_threshold": 70, "breakthrough_velocity": 20},
    "LUSC": {"classic_threshold": 70, "breakthrough_velocity": 20},
    "lung_cancer": {"classic_threshold": 70, "breakthrough_velocity": 20},
    "STAD": {"classic_threshold": 50, "breakthrough_velocity": 15},
    "stomach_cancer": {"classic_threshold": 50, "breakthrough_velocity": 15},
    "PAAD": {"classic_threshold": 50, "breakthrough_velocity": 15},
    "pancreatic_cancer": {"classic_threshold": 50, "breakthrough_velocity": 15},
    "COAD": {"classic_threshold": 60, "breakthrough_velocity": 18},
    "colorectal_cancer": {"classic_threshold": 60, "breakthrough_velocity": 18},
    "LIHC": {"classic_threshold": 55, "breakthrough_velocity": 16},
    "liver_cancer": {"classic_threshold": 55, "breakthrough_velocity": 16},
    "KIRC": {"classic_threshold": 50, "breakthrough_velocity": 15},
    "kidney_cancer": {"classic_threshold": 50, "breakthrough_velocity": 15},
    "general_oncology": {"classic_threshold": 50, "breakthrough_velocity": 15},
}

# PubMed API endpoints
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Cancer type to search term mapping
CANCER_SEARCH_TERMS = {
    "BRCA": "breast cancer",
    "LUAD": "lung adenocarcinoma",
    "LUSC": "lung squamous cell carcinoma",
    "COAD": "colorectal cancer",
    "STAD": "stomach cancer gastric cancer",
    "LIHC": "liver cancer hepatocellular carcinoma",
    "KIRC": "kidney renal clear cell carcinoma",
    "HNSC": "head and neck squamous cell carcinoma",
    "THCA": "thyroid cancer",
    "PRAD": "prostate cancer",
    "BLCA": "bladder cancer",
    "OV": "ovarian cancer",
    "UCEC": "uterine endometrial cancer",
    "PAAD": "pancreatic cancer",
    "GBM": "glioblastoma",
    "LGG": "low grade glioma",
    "SKCM": "melanoma skin cancer",
    "breast_cancer": "breast cancer",
    "lung_cancer": "lung cancer",
    "colorectal_cancer": "colorectal cancer",
    "pancreatic_cancer": "pancreatic cancer",
    "liver_cancer": "liver hepatocellular carcinoma",
    "kidney_cancer": "kidney renal cell carcinoma",
    "stomach_cancer": "gastric cancer stomach cancer",
    "thyroid_cancer": "thyroid cancer",
    "prostate_cancer": "prostate cancer",
    "ovarian_cancer": "ovarian cancer",
    "bladder_cancer": "bladder cancer",
    "melanoma": "melanoma",
    "glioblastoma": "glioblastoma",
}


@dataclass
class RecommendedPaper:
    """Recommended paper data structure."""
    pmid: str
    title: str
    authors: str
    journal: str
    year: str
    abstract: str
    doi: str = ""
    relevance_reason: str = ""
    pubmed_url: str = ""

    def __post_init__(self):
        if not self.pubmed_url and self.pmid:
            self.pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnhancedRecommendedPaper:
    """Enhanced paper with citation-aware quality metrics."""
    # Basic metadata (from PubMed)
    pmid: str
    title: str
    authors: str
    journal: str
    year: str
    abstract: str
    doi: str = ""
    relevance_reason: str = ""
    pubmed_url: str = ""

    # Citation metrics (from Semantic Scholar)
    citation_count: int = 0
    influential_citation_count: int = 0
    citation_velocity: float = 0.0  # citations per year

    # Quality classification
    paper_type: str = "unknown"  # "classic", "breakthrough", "unknown"
    quality_score: float = 0.0   # 0-100 combined score
    passes_quality_gate: bool = False

    # Score components
    citation_score: float = 0.0      # 0-40 based on absolute citations
    velocity_score: float = 0.0      # 0-30 based on citation velocity
    recency_score: float = 0.0       # 0-30 based on publication year

    # Metadata
    citation_data_source: str = "none"  # "semantic_scholar", "cache", "estimated"

    def __post_init__(self):
        if not self.pubmed_url and self.pmid:
            self.pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"

        # Calculate citation velocity
        if self.year and self.citation_count > 0:
            try:
                age = max(1, datetime.now().year - int(self.year))
                self.citation_velocity = self.citation_count / age
            except (ValueError, TypeError):
                self.citation_velocity = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_basic_paper(cls, paper: RecommendedPaper) -> 'EnhancedRecommendedPaper':
        """Create EnhancedRecommendedPaper from basic RecommendedPaper."""
        return cls(
            pmid=paper.pmid,
            title=paper.title,
            authors=paper.authors,
            journal=paper.journal,
            year=paper.year,
            abstract=paper.abstract,
            doi=paper.doi,
            relevance_reason=paper.relevance_reason,
            pubmed_url=paper.pubmed_url
        )


class PaperRecommender:
    """
    Recommends relevant papers for RNA-seq analysis results.

    Uses PubMed API to find papers related to:
    - Specific cancer type
    - Top differentially expressed genes
    - Enriched pathways

    Enhanced with Semantic Scholar citation data for quality filtering.
    """

    # Default cache directory
    DEFAULT_CACHE_DIR = Path.home() / ".bioinsight" / "citation_cache"

    def __init__(
        self,
        cancer_type: str = "",
        email: str = "bioinsight@research.ai",
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        self.cancer_type = cancer_type
        self.email = email
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=30)
        self._last_request_time = 0
        self._last_ss_request_time = 0  # Semantic Scholar rate limiting
        self._rate_limit_delay = 0.4  # 400ms between PubMed requests
        self._ss_rate_limit_delay = 5.0  # 5 seconds between Semantic Scholar requests (avoid 429)

        # Citation cache setup
        self.cache_dir = cache_dir or self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "paper_recommender_cache.json"
        self._citation_cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load citation cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_cache(self):
        """Save citation cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._citation_cache, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save citation cache: {e}")

    def _get_cached_citation(self, pmid: str) -> Optional[Dict]:
        """Get citation data from cache if valid."""
        if pmid not in self._citation_cache:
            return None

        cached = self._citation_cache[pmid]
        cache_time = cached.get('timestamp', 0)
        ttl_days = cached.get('ttl_days', 7)

        # Check if cache is still valid
        if time.time() - cache_time < ttl_days * 24 * 60 * 60:
            return cached

        return None

    def _cache_citation(self, pmid: str, data: Dict):
        """Cache citation data."""
        self._citation_cache[pmid] = {
            **data,
            'timestamp': time.time(),
            'ttl_days': 7
        }
        self._save_cache()

    async def _rate_limit(self):
        """Ensure we don't exceed PubMed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    async def _ss_rate_limit(self):
        """Ensure we don't exceed Semantic Scholar rate limits."""
        elapsed = time.time() - self._last_ss_request_time
        if elapsed < self._ss_rate_limit_delay:
            await asyncio.sleep(self._ss_rate_limit_delay - elapsed)
        self._last_ss_request_time = time.time()

    async def _fetch_semantic_scholar(
        self,
        session: aiohttp.ClientSession,
        pmid: str
    ) -> Optional[Dict]:
        """Fetch citation data from Semantic Scholar API."""
        # Check cache first
        cached = self._get_cached_citation(pmid)
        if cached:
            logger.debug(f"Cache hit for PMID {pmid}")
            return cached

        await self._ss_rate_limit()

        url = f"{SEMANTIC_SCHOLAR_API}/PMID:{pmid}"
        params = {"fields": "citationCount,influentialCitationCount,year,venue"}

        try:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    citation_data = {
                        'citation_count': data.get('citationCount', 0),
                        'influential_citation_count': data.get('influentialCitationCount', 0),
                        'year': data.get('year'),
                        'venue': data.get('venue', ''),
                        'source': 'semantic_scholar'
                    }
                    self._cache_citation(pmid, citation_data)
                    return citation_data
                elif response.status == 404:
                    # Paper not found - cache this to avoid repeated lookups
                    self._cache_citation(pmid, {
                        'citation_count': 0,
                        'influential_citation_count': 0,
                        'source': 'not_found'
                    })
                    return None
                else:
                    logger.warning(f"Semantic Scholar API error for PMID {pmid}: {response.status}")
                    return None
        except asyncio.TimeoutError:
            logger.warning(f"Semantic Scholar timeout for PMID {pmid}")
            return None
        except Exception as e:
            logger.error(f"Semantic Scholar error for PMID {pmid}: {e}")
            return None

    async def _fetch_citations_batch(
        self,
        session: aiohttp.ClientSession,
        pmids: List[str],
        max_api_calls: int = 15
    ) -> Dict[str, Dict]:
        """
        Fetch citation data for a batch of PMIDs with rate limiting.

        Args:
            session: aiohttp session
            pmids: List of PMIDs to fetch
            max_api_calls: Maximum API calls to make (rest will use cache or skip)

        Returns:
            Dict mapping PMID to citation data
        """
        results = {}
        api_calls = 0

        for pmid in pmids:
            # Check cache first
            cached = self._get_cached_citation(pmid)
            if cached:
                results[pmid] = cached
                continue

            # Limit API calls
            if api_calls >= max_api_calls:
                logger.debug(f"Reached max API calls ({max_api_calls}), skipping remaining")
                break

            # Fetch from API
            data = await self._fetch_semantic_scholar(session, pmid)
            if data:
                results[pmid] = data
                api_calls += 1

        logger.info(f"Citation fetch: {len(results)} results ({api_calls} API calls)")
        return results

    def _build_search_query(
        self,
        genes: List[str],
        pathways: List[str] = None,
        focus: str = "genes"
    ) -> str:
        """
        Build optimized PubMed search query.

        Args:
            genes: List of gene symbols
            pathways: List of pathway names (optional)
            focus: "genes", "pathways", or "overview"

        Returns:
            PubMed search query string
        """
        # Get cancer search term
        cancer_term = CANCER_SEARCH_TERMS.get(
            self.cancer_type,
            self.cancer_type.replace("_", " ")
        )

        if focus == "genes" and genes:
            # Focus on top genes - use broader search
            top_genes = genes[:5]
            gene_query = " OR ".join([f"{g}" for g in top_genes])
            # Broaden search: cancer + gene + expression-related
            query = f"({cancer_term}) AND ({gene_query}) AND (expression OR RNA-seq OR transcriptome)"

        elif focus == "pathways" and pathways:
            # Focus on pathways
            top_pathways = pathways[:3]
            pathway_terms = " OR ".join([f'"{p}"' for p in top_pathways])
            query = f"({cancer_term}) AND ({pathway_terms}) AND (cancer OR tumor)"

        else:
            # General overview - focus on RNA-seq and key cancer terms
            query = f"({cancer_term}) AND (RNA-seq OR transcriptome OR gene expression) AND (biomarker OR therapeutic OR prognosis OR survival)"

        return query

    async def _search_pmids(
        self,
        session: aiohttp.ClientSession,
        query: str,
        max_results: int = 10,
        sort: str = "relevance"
    ) -> List[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        # Add date filter for recent papers (last 5 years)
        min_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y/%m/%d")
        params["mindate"] = min_date
        params["datetype"] = "pdat"

        await self._rate_limit()

        try:
            async with session.get(PUBMED_SEARCH_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("esearchresult", {}).get("idlist", [])
                else:
                    logger.warning(f"PubMed search failed: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            return []

    async def _fetch_paper_details(
        self,
        session: aiohttp.ClientSession,
        pmids: List[str]
    ) -> List[RecommendedPaper]:
        """Fetch paper details from PubMed."""
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        await self._rate_limit()

        try:
            async with session.get(PUBMED_FETCH_URL, params=params) as response:
                if response.status != 200:
                    return []

                xml_text = await response.text()
                return self._parse_pubmed_xml(xml_text)
        except Exception as e:
            logger.error(f"PubMed fetch error: {e}")
            return []

    def _parse_pubmed_xml(self, xml_text: str) -> List[RecommendedPaper]:
        """Parse PubMed XML response."""
        papers = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    # PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""

                    # Title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title"

                    # Authors
                    authors = []
                    for author in article.findall(".//Author")[:3]:
                        lastname = author.find("LastName")
                        if lastname is not None:
                            authors.append(lastname.text)
                    authors_str = ", ".join(authors)
                    if len(article.findall(".//Author")) > 3:
                        authors_str += " et al."

                    # Journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""

                    # Year
                    year_elem = article.find(".//PubDate/Year")
                    year = year_elem.text if year_elem is not None else ""

                    # Abstract
                    abstract_parts = []
                    for abstract_text in article.findall(".//AbstractText"):
                        if abstract_text.text:
                            abstract_parts.append(abstract_text.text)
                    abstract = " ".join(abstract_parts)[:500]  # Limit length

                    # DOI
                    doi = ""
                    for article_id in article.findall(".//ArticleId"):
                        if article_id.get("IdType") == "doi":
                            doi = article_id.text
                            break

                    papers.append(RecommendedPaper(
                        pmid=pmid,
                        title=title,
                        authors=authors_str,
                        journal=journal,
                        year=year,
                        abstract=abstract,
                        doi=doi
                    ))

                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return papers

    def _get_field_config(self) -> Dict[str, int]:
        """Get citation threshold configuration for this cancer type."""
        return FIELD_CITATION_BASELINES.get(
            self.cancer_type,
            FIELD_CITATION_BASELINES["general_oncology"]
        )

    def _calculate_quality_score(self, paper: EnhancedRecommendedPaper) -> float:
        """
        Calculate comprehensive quality score (0-100).

        Score breakdown:
        - Citation score (0-40): Based on absolute citation count
        - Velocity score (0-30): Based on citation velocity
        - Recency score (0-30): Based on publication year
        """
        current_year = datetime.now().year

        # 1. Citation score (0-40) - log scale
        citation_count = paper.citation_count
        if citation_count <= 0:
            citation_score = 0
        elif citation_count < 10:
            citation_score = citation_count * 2  # 0-20
        elif citation_count < 100:
            citation_score = 20 + (citation_count - 10) * 0.11  # 20-30
        elif citation_count < 1000:
            citation_score = 30 + (citation_count - 100) * 0.011  # 30-40
        else:
            citation_score = 40

        # 2. Velocity score (0-30) - citations per year
        velocity = paper.citation_velocity
        if velocity <= 0:
            velocity_score = 0
        elif velocity < 5:
            velocity_score = velocity * 2  # 0-10
        elif velocity < 20:
            velocity_score = 10 + (velocity - 5) * 0.67  # 10-20
        elif velocity < 50:
            velocity_score = 20 + (velocity - 20) * 0.33  # 20-30
        else:
            velocity_score = 30

        # 3. Recency score (0-30) - recent papers get bonus
        try:
            pub_year = int(paper.year) if paper.year else 0
            years_old = current_year - pub_year
        except (ValueError, TypeError):
            years_old = 5  # Default to neutral

        if years_old <= 1:
            recency_score = 30
        elif years_old <= 2:
            recency_score = 27
        elif years_old <= 3:
            recency_score = 24
        elif years_old <= 5:
            recency_score = 20
        elif years_old <= 10:
            recency_score = 15
        else:
            recency_score = max(5, 15 - (years_old - 10))

        # Update paper's component scores
        paper.citation_score = round(citation_score, 1)
        paper.velocity_score = round(velocity_score, 1)
        paper.recency_score = round(recency_score, 1)

        return round(citation_score + velocity_score + recency_score, 1)

    def _classify_paper_type(self, paper: EnhancedRecommendedPaper) -> str:
        """
        Classify paper as classic, breakthrough, likely variants, or unknown.

        Classic paper: High absolute citations (100+), 3+ years old
        Likely Classic: Good citations (50+), 3+ years old
        Breakthrough paper: Recent (1-2 years), high citation velocity (20+)
        Likely Breakthrough: Recent (1-3 years), moderate velocity (10+)
        """
        current_year = datetime.now().year

        try:
            pub_year = int(paper.year) if paper.year else 0
            age = current_year - pub_year
        except (ValueError, TypeError):
            return "unknown"

        # Classic paper criteria (strict)
        if (age >= 3 and
            paper.citation_count >= 100 and
            paper.citation_velocity >= 10):
            return "classic"

        # Likely classic (relaxed)
        if (age >= 3 and
            paper.citation_count >= 50 and
            paper.citation_velocity >= 5):
            return "likely_classic"

        # Breakthrough paper criteria (strict)
        if age <= 2:
            # High velocity or high influential citations
            if (paper.citation_velocity >= 20 or
                paper.influential_citation_count >= 5):
                return "breakthrough"

        # Likely breakthrough (relaxed) - 1-3 years, moderate velocity
        if (1 <= age <= 3 and
            (paper.citation_velocity >= 10 or
             paper.influential_citation_count >= 3)):
            return "likely_breakthrough"

        return "unknown"

    def _passes_quality_gate(self, paper: EnhancedRecommendedPaper) -> bool:
        """
        Check if paper meets quality criteria for its type.

        Classic papers: Must meet classic_threshold for citations
        Breakthrough papers: Must meet breakthrough_velocity for citation velocity
        Likely variants: Use relaxed thresholds
        """
        config = self._get_field_config()

        if paper.paper_type == "classic":
            return paper.citation_count >= config["classic_threshold"]
        elif paper.paper_type == "likely_classic":
            # Relaxed threshold for likely classics (50% of classic threshold)
            return paper.citation_count >= config["classic_threshold"] * 0.5
        elif paper.paper_type == "breakthrough":
            return paper.citation_velocity >= config["breakthrough_velocity"]
        elif paper.paper_type == "likely_breakthrough":
            # Relaxed threshold for likely breakthroughs (50% of breakthrough velocity)
            return paper.citation_velocity >= config["breakthrough_velocity"] * 0.5

        # Unknown type: use lower bar for consideration
        return paper.citation_count >= 20 or paper.citation_velocity >= 10

    def _estimate_quality_without_citations(
        self,
        paper: EnhancedRecommendedPaper
    ) -> Tuple[float, str]:
        """
        Estimate quality when citation data is unavailable.

        Uses journal prestige and publication year as proxies.

        Returns:
            Tuple of (estimated_score, paper_type)
        """
        score = 50.0  # Base score
        paper_type = "unknown"

        current_year = datetime.now().year

        # Journal prestige bonus
        if paper.journal and any(j.lower() in paper.journal.lower() for j in HIGH_IMPACT_JOURNALS):
            score += 30

        # Age factor
        try:
            pub_year = int(paper.year) if paper.year else 0
            age = current_year - pub_year
        except (ValueError, TypeError):
            age = 3  # Default

        if age >= 5 and score >= 70:
            paper_type = "likely_classic"
            score += 10
        elif age <= 2 and score >= 70:
            paper_type = "likely_breakthrough"
            score += 10
        elif age <= 2:
            # Recent paper from decent journal
            score += 5

        paper.citation_data_source = "estimated"
        return min(100, score), paper_type

    def _enrich_with_citations(
        self,
        papers: List[RecommendedPaper],
        citation_data: Dict[str, Dict]
    ) -> List[EnhancedRecommendedPaper]:
        """
        Enrich basic papers with citation data.

        Args:
            papers: List of basic RecommendedPaper objects
            citation_data: Dict mapping PMID to citation info

        Returns:
            List of EnhancedRecommendedPaper objects
        """
        enriched = []

        for paper in papers:
            enhanced = EnhancedRecommendedPaper.from_basic_paper(paper)

            if paper.pmid in citation_data:
                data = citation_data[paper.pmid]
                enhanced.citation_count = data.get('citation_count', 0)
                enhanced.influential_citation_count = data.get('influential_citation_count', 0)
                enhanced.citation_data_source = data.get('source', 'semantic_scholar')

                # Recalculate velocity with actual data
                try:
                    age = max(1, datetime.now().year - int(paper.year))
                    enhanced.citation_velocity = enhanced.citation_count / age
                except (ValueError, TypeError):
                    enhanced.citation_velocity = 0.0
            else:
                # Estimate quality without citation data
                estimated_score, estimated_type = self._estimate_quality_without_citations(enhanced)
                enhanced.quality_score = estimated_score
                if enhanced.paper_type == "unknown":
                    enhanced.paper_type = estimated_type

            enriched.append(enhanced)

        return enriched

    async def recommend_papers_enhanced(
        self,
        hub_genes: List[str],
        pathways: List[str] = None,
        max_papers: int = 6,
        balance_classic_breakthrough: bool = True,
        quality_filter: bool = True
    ) -> List[EnhancedRecommendedPaper]:
        """
        Get citation-aware paper recommendations.

        Args:
            hub_genes: List of top hub genes from analysis
            pathways: List of enriched pathways (optional)
            max_papers: Maximum number of papers to recommend
            balance_classic_breakthrough: If True, aim for 50% classic / 50% breakthrough
            quality_filter: If True, only return papers passing quality gate

        Returns:
            List of EnhancedRecommendedPaper objects
        """
        all_papers = []
        seen_pmids = set()

        # Expand search to get more candidates for filtering
        search_multiplier = 6 if quality_filter else 3

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Strategy 1: Search by top genes
            gene_count = max(3, int(max_papers * search_multiplier * 0.5))
            gene_query = self._build_search_query(hub_genes, focus="genes")
            gene_pmids = await self._search_pmids(session, gene_query, gene_count)

            if gene_pmids:
                gene_papers = await self._fetch_paper_details(session, gene_pmids)
                for paper in gene_papers:
                    if paper.pmid not in seen_pmids:
                        paper.relevance_reason = f"Hub genes 관련: {', '.join(hub_genes[:3])}"
                        all_papers.append(paper)
                        seen_pmids.add(paper.pmid)

            # Strategy 2: Search by pathways
            if pathways:
                pathway_count = max(2, int(max_papers * search_multiplier * 0.25))
                pathway_query = self._build_search_query([], pathways, focus="pathways")
                pathway_pmids = await self._search_pmids(session, pathway_query, pathway_count)

                if pathway_pmids:
                    pathway_papers = await self._fetch_paper_details(session, pathway_pmids)
                    for paper in pathway_papers:
                        if paper.pmid not in seen_pmids:
                            paper.relevance_reason = f"Pathway 관련: {pathways[0][:30]}"
                            all_papers.append(paper)
                            seen_pmids.add(paper.pmid)

            # Strategy 3: General overview (for diversity)
            overview_count = max(2, int(max_papers * search_multiplier * 0.25))
            overview_query = self._build_search_query([], focus="overview")
            overview_pmids = await self._search_pmids(session, overview_query, overview_count, sort="pub_date")

            if overview_pmids:
                overview_papers = await self._fetch_paper_details(session, overview_pmids)
                for paper in overview_papers:
                    if paper.pmid not in seen_pmids:
                        paper.relevance_reason = f"{self.cancer_type} 최신 연구"
                        all_papers.append(paper)
                        seen_pmids.add(paper.pmid)

            # Fetch citation data for all candidates
            pmids = [p.pmid for p in all_papers]
            citation_data = await self._fetch_citations_batch(session, pmids, max_api_calls=15)

        # Enrich papers with citation data
        enriched = self._enrich_with_citations(all_papers, citation_data)

        # Calculate quality scores and classify
        for paper in enriched:
            if paper.citation_data_source != "estimated":
                paper.quality_score = self._calculate_quality_score(paper)
                paper.paper_type = self._classify_paper_type(paper)
            paper.passes_quality_gate = self._passes_quality_gate(paper)

        # Apply quality filter if requested
        if quality_filter:
            # Keep papers that pass quality gate OR have good estimated scores
            enriched = [p for p in enriched if p.passes_quality_gate or p.quality_score >= 60]

        logger.info(f"After quality filter: {len(enriched)} papers remain")

        # Balance classic vs breakthrough if requested
        if balance_classic_breakthrough and len(enriched) >= 2:
            classics = sorted(
                [p for p in enriched if p.paper_type in ("classic", "likely_classic")],
                key=lambda p: p.quality_score,
                reverse=True
            )
            breakthroughs = sorted(
                [p for p in enriched if p.paper_type in ("breakthrough", "likely_breakthrough")],
                key=lambda p: p.quality_score,
                reverse=True
            )
            unknowns = sorted(
                [p for p in enriched if p.paper_type == "unknown"],
                key=lambda p: p.quality_score,
                reverse=True
            )

            # Target 50:50 split
            half = max_papers // 2
            result = []

            # Add classics
            result.extend(classics[:half])

            # Add breakthroughs
            result.extend(breakthroughs[:half])

            # Fill remaining slots with best available
            remaining = max_papers - len(result)
            if remaining > 0:
                available = [p for p in enriched if p not in result]
                available.sort(key=lambda p: p.quality_score, reverse=True)
                result.extend(available[:remaining])

            logger.info(f"Balanced result: {len([p for p in result if 'classic' in p.paper_type])} classic, "
                       f"{len([p for p in result if 'breakthrough' in p.paper_type])} breakthrough")

            return result[:max_papers]

        # Return top quality papers
        enriched.sort(key=lambda p: p.quality_score, reverse=True)
        return enriched[:max_papers]

    async def recommend_papers(
        self,
        hub_genes: List[str],
        pathways: List[str] = None,
        max_papers: int = 5
    ) -> List[RecommendedPaper]:
        """
        Get paper recommendations based on analysis results.

        Args:
            hub_genes: List of top hub genes from analysis
            pathways: List of enriched pathways (optional)
            max_papers: Maximum number of papers to recommend

        Returns:
            List of RecommendedPaper objects
        """
        all_papers = []
        seen_pmids = set()

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Strategy 1: Search by top genes (60% of results)
            gene_count = max(3, int(max_papers * 0.6))
            gene_query = self._build_search_query(hub_genes, focus="genes")
            gene_pmids = await self._search_pmids(session, gene_query, gene_count * 2)

            if gene_pmids:
                gene_papers = await self._fetch_paper_details(session, gene_pmids[:gene_count])
                for paper in gene_papers:
                    if paper.pmid not in seen_pmids:
                        paper.relevance_reason = f"Hub genes 관련: {', '.join(hub_genes[:3])}"
                        all_papers.append(paper)
                        seen_pmids.add(paper.pmid)

            # Strategy 2: Search by pathways (if provided, 20% of results)
            if pathways:
                pathway_count = max(1, int(max_papers * 0.2))
                pathway_query = self._build_search_query([], pathways, focus="pathways")
                pathway_pmids = await self._search_pmids(session, pathway_query, pathway_count * 2)

                if pathway_pmids:
                    pathway_papers = await self._fetch_paper_details(session, pathway_pmids[:pathway_count])
                    for paper in pathway_papers:
                        if paper.pmid not in seen_pmids:
                            paper.relevance_reason = f"Pathway 관련: {pathways[0][:30]}"
                            all_papers.append(paper)
                            seen_pmids.add(paper.pmid)

            # Strategy 3: General cancer + RNA-seq overview (remaining)
            remaining = max_papers - len(all_papers)
            if remaining > 0:
                overview_query = self._build_search_query([], focus="overview")
                overview_pmids = await self._search_pmids(
                    session, overview_query, remaining * 2, sort="pub_date"
                )

                if overview_pmids:
                    overview_papers = await self._fetch_paper_details(session, overview_pmids[:remaining])
                    for paper in overview_papers:
                        if paper.pmid not in seen_pmids:
                            paper.relevance_reason = f"{self.cancer_type} 최신 연구"
                            all_papers.append(paper)
                            seen_pmids.add(paper.pmid)

        # Return top papers
        return all_papers[:max_papers]

    def save_recommendations(
        self,
        papers: List[RecommendedPaper],
        output_path: Path
    ) -> None:
        """Save recommendations to JSON file."""
        output_path = Path(output_path)

        data = {
            "cancer_type": self.cancer_type,
            "generated_at": datetime.now().isoformat(),
            "paper_count": len(papers),
            "papers": [p.to_dict() for p in papers]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(papers)} paper recommendations to {output_path}")


def get_paper_recommender(cancer_type: str) -> PaperRecommender:
    """Factory function to create PaperRecommender."""
    return PaperRecommender(cancer_type=cancer_type)


async def recommend_papers_for_analysis(
    cancer_type: str,
    hub_genes: List[str],
    pathways: List[str] = None,
    output_dir: Path = None,
    max_papers: int = 5,
    quality_filter: bool = False,
    balance_classic_breakthrough: bool = False
) -> List[Dict[str, Any]]:
    """
    Convenience function to get paper recommendations for RNA-seq analysis.

    Args:
        cancer_type: TCGA code or cancer name
        hub_genes: List of hub genes from analysis
        pathways: List of enriched pathways
        output_dir: Optional output directory to save results
        max_papers: Maximum papers to recommend
        quality_filter: If True, use citation-based quality filtering
        balance_classic_breakthrough: If True, balance classic and breakthrough papers

    Returns:
        List of paper dictionaries
    """
    recommender = PaperRecommender(cancer_type=cancer_type)
    papers = await recommender.recommend_papers(hub_genes, pathways, max_papers)

    if output_dir:
        output_path = Path(output_dir) / "recommended_papers.json"
        recommender.save_recommendations(papers, output_path)

    return [p.to_dict() for p in papers]


async def recommend_papers_enhanced_for_analysis(
    cancer_type: str,
    hub_genes: List[str],
    pathways: List[str] = None,
    output_dir: Path = None,
    max_papers: int = 6,
    quality_filter: bool = True,
    balance_classic_breakthrough: bool = True
) -> Dict[str, Any]:
    """
    Enhanced paper recommendations with citation-based quality filtering.

    Args:
        cancer_type: TCGA code or cancer name
        hub_genes: List of hub genes from analysis
        pathways: List of enriched pathways
        output_dir: Optional output directory to save results
        max_papers: Maximum papers to recommend
        quality_filter: If True, filter to top 10% papers
        balance_classic_breakthrough: If True, aim for 50% classic / 50% breakthrough

    Returns:
        Dict with papers organized by type and metadata
    """
    recommender = PaperRecommender(cancer_type=cancer_type)
    papers = await recommender.recommend_papers_enhanced(
        hub_genes, pathways, max_papers,
        balance_classic_breakthrough=balance_classic_breakthrough,
        quality_filter=quality_filter
    )

    # Organize by type
    classics = [p for p in papers if p.paper_type in ("classic", "likely_classic")]
    breakthroughs = [p for p in papers if p.paper_type in ("breakthrough", "likely_breakthrough")]
    others = [p for p in papers if p.paper_type == "unknown"]

    result = {
        "cancer_type": cancer_type,
        "search_genes": hub_genes[:5],
        "generated_at": datetime.now().isoformat(),
        "paper_count": len(papers),
        "classic_count": len(classics),
        "breakthrough_count": len(breakthroughs),
        "classic_papers": [p.to_dict() for p in classics],
        "breakthrough_papers": [p.to_dict() for p in breakthroughs],
        "other_papers": [p.to_dict() for p in others],
        "papers": [p.to_dict() for p in papers]  # backward compatibility
    }

    if output_dir:
        output_path = Path(output_dir) / "recommended_papers.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(papers)} enhanced paper recommendations to {output_path}")

    return result


# Synchronous wrapper for non-async contexts
def recommend_papers_sync(
    cancer_type: str,
    hub_genes: List[str],
    pathways: List[str] = None,
    output_dir: Path = None,
    max_papers: int = 5
) -> List[Dict[str, Any]]:
    """Synchronous wrapper for recommend_papers_for_analysis."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create new loop
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        recommend_papers_for_analysis(
            cancer_type, hub_genes, pathways, output_dir, max_papers
        )
    )


def recommend_papers_enhanced_sync(
    cancer_type: str,
    hub_genes: List[str],
    pathways: List[str] = None,
    output_dir: Path = None,
    max_papers: int = 6,
    quality_filter: bool = True,
    balance_classic_breakthrough: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for enhanced paper recommendations.

    Returns dict with classic_papers, breakthrough_papers, and papers lists.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        recommend_papers_enhanced_for_analysis(
            cancer_type, hub_genes, pathways, output_dir, max_papers,
            quality_filter, balance_classic_breakthrough
        )
    )
