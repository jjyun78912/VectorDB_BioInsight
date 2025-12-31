"""
Web Crawler Agent for BioInsight.

Provides real-time paper fetching capabilities:
1. DOI/URL-based paper import
2. Real-time PubMed search
3. Trending papers from PubMed
4. Semantic Scholar integration

Usage:
    from backend.app.core.web_crawler_agent import WebCrawlerAgent

    agent = WebCrawlerAgent()

    # Fetch by DOI
    paper = await agent.fetch_by_doi("10.1038/s41586-021-03819-2")

    # Search PubMed in real-time
    papers = await agent.search_pubmed("CRISPR cancer therapy", max_results=10)

    # Get trending papers
    trending = await agent.get_trending_papers(category="oncology")
"""

import asyncio
import aiohttp
import re
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import quote, urljoin
import xml.etree.ElementTree as ET

from .config import PAPERS_DIR, CHROMA_DIR
from .text_splitter import BioPaperSplitter, TextChunk


# API Endpoints
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_TRENDING_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi"
CROSSREF_API = "https://api.crossref.org/works"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"


# Major journals for high-impact papers
MAJOR_JOURNALS = [
    # Top tier general science
    "Nature",
    "Science",
    "Cell",
    "New England Journal of Medicine",
    "The Lancet",
    "JAMA",
    "BMJ",
    # Nature family
    "Nature Medicine",
    "Nature Genetics",
    "Nature Biotechnology",
    "Nature Communications",
    "Nature Cancer",
    "Nature Immunology",
    "Nature Cell Biology",
    "Nature Methods",
    "Nature Reviews Cancer",
    "Nature Reviews Drug Discovery",
    "Nature Reviews Genetics",
    # Cell family
    "Cell Metabolism",
    "Cell Reports",
    "Cell Stem Cell",
    "Molecular Cell",
    "Cancer Cell",
    "Immunity",
    # Other high-impact
    "PNAS",
    "Science Translational Medicine",
    "Journal of Clinical Oncology",
    "Journal of Clinical Investigation",
    "Lancet Oncology",
    "JAMA Oncology",
    "Annals of Oncology",
    "Blood",
    "Gastroenterology",
    "Gut",
    "Hepatology",
    "Circulation",
    "European Heart Journal",
    "Diabetes",
    "eLife",
    "EMBO Journal",
    "Genome Research",
    "Genome Biology",
    "Nucleic Acids Research",
    "Cancer Research",
    "Clinical Cancer Research",
    "Cancer Discovery",
]

# Build journal filter for PubMed
MAJOR_JOURNAL_FILTER = " OR ".join([f'"{j}"[Journal]' for j in MAJOR_JOURNALS])

# Trending topic categories for PubMed (with major journal filter option)
TRENDING_CATEGORIES = {
    "oncology": "(cancer[Title] OR tumor[Title] OR oncology[Title]) AND (2024[pdat] OR 2025[pdat])",
    "immunotherapy": "(immunotherapy[Title] OR CAR-T[Title] OR checkpoint inhibitor[Title]) AND (2024[pdat] OR 2025[pdat])",
    "gene_therapy": "(gene therapy[Title] OR CRISPR[Title] OR gene editing[Title]) AND (2024[pdat] OR 2025[pdat])",
    "neurology": "(Alzheimer[Title] OR Parkinson[Title] OR neurodegenerative[Title]) AND (2024[pdat] OR 2025[pdat])",
    "infectious_disease": "(COVID-19[Title] OR SARS-CoV-2[Title] OR pandemic[Title]) AND (2024[pdat] OR 2025[pdat])",
    "ai_medicine": "(artificial intelligence[Title] OR machine learning[Title] OR deep learning[Title]) AND medicine AND (2024[pdat] OR 2025[pdat])",
    "genomics": "(single-cell[Title] OR RNA-seq[Title] OR genomics[Title]) AND (2024[pdat] OR 2025[pdat])",
    "drug_discovery": "(drug discovery[Title] OR pharmaceutical[Title] OR clinical trial[Title]) AND (2024[pdat] OR 2025[pdat])",
}


@dataclass
class FetchedPaper:
    """Normalized paper data from any source."""
    id: str  # Unique identifier (DOI, PMID, etc.)
    source: str  # "pubmed", "crossref", "semantic_scholar", "doi"
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    journal: str = ""
    year: int = 0
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""
    url: str = ""
    keywords: List[str] = field(default_factory=list)
    citation_count: int = 0
    references: List[str] = field(default_factory=list)
    full_text: str = ""
    fetched_at: str = ""

    # Calculated fields
    trend_score: float = 0.0  # 0-100
    recency_score: float = 0.0  # Based on publication date

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().isoformat()
        self._calculate_scores()

    def _calculate_scores(self):
        """Calculate trend and recency scores."""
        # Recency score (papers from last 2 years get higher scores)
        current_year = datetime.now().year
        if self.year:
            years_old = current_year - self.year
            self.recency_score = max(0, 100 - (years_old * 20))

        # Trend score (combination of citations and recency)
        if self.citation_count > 0:
            # Normalize citations (log scale)
            import math
            citation_score = min(100, math.log10(self.citation_count + 1) * 30)
            self.trend_score = (citation_score * 0.6) + (self.recency_score * 0.4)
        else:
            self.trend_score = self.recency_score * 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WebCrawlerAgent:
    """
    Intelligent web crawler agent for biomedical papers.

    Capabilities:
    - Fetch papers by DOI/URL
    - Real-time PubMed search
    - Get trending papers
    - Multi-source aggregation (PubMed, CrossRef, Semantic Scholar)
    """

    def __init__(self, ncbi_api_key: str = None, timeout: int = 30):
        """
        Initialize the crawler agent.

        Args:
            ncbi_api_key: Optional NCBI API key for higher rate limits
            timeout: Request timeout in seconds
        """
        self.ncbi_api_key = ncbi_api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.text_splitter = BioPaperSplitter()

        # Rate limiting (requests per second)
        self.rate_limits = {
            "pubmed": 3 if ncbi_api_key else 1,
            "crossref": 5,
            "semantic_scholar": 5,
        }
        self.last_request_time = {}

        # Cache for trending papers
        self._trending_cache = {}
        self._cache_expiry = timedelta(hours=1)

    async def _rate_limit(self, api: str):
        """Apply rate limiting for API calls."""
        now = time.time()
        last = self.last_request_time.get(api, 0)
        wait_time = 1.0 / self.rate_limits.get(api, 1)

        if now - last < wait_time:
            await asyncio.sleep(wait_time - (now - last))

        self.last_request_time[api] = time.time()

    async def _fetch_json(self, session: aiohttp.ClientSession, url: str, params: dict = None) -> dict:
        """Fetch JSON from URL."""
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def _fetch_text(self, session: aiohttp.ClientSession, url: str, params: dict = None) -> str:
        """Fetch text/XML from URL."""
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.text()

    # ==================== DOI/URL Fetching ====================

    async def fetch_by_doi(self, doi: str) -> Optional[FetchedPaper]:
        """
        Fetch paper metadata by DOI using CrossRef and Semantic Scholar.

        Args:
            doi: DOI string (e.g., "10.1038/s41586-021-03819-2")

        Returns:
            FetchedPaper object or None
        """
        # Clean DOI
        doi = self._clean_doi(doi)
        if not doi:
            return None

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            paper = None

            # Try CrossRef first (most comprehensive for DOIs)
            try:
                await self._rate_limit("crossref")
                paper = await self._fetch_from_crossref(session, doi)
            except Exception as e:
                print(f"CrossRef fetch failed: {e}")

            # Enrich with Semantic Scholar (citations, references)
            if paper:
                try:
                    await self._rate_limit("semantic_scholar")
                    paper = await self._enrich_with_semantic_scholar(session, paper)
                except Exception as e:
                    print(f"Semantic Scholar enrichment failed: {e}")

            # Try Semantic Scholar as primary if CrossRef failed
            if not paper:
                try:
                    await self._rate_limit("semantic_scholar")
                    paper = await self._fetch_from_semantic_scholar(session, doi)
                except Exception as e:
                    print(f"Semantic Scholar fetch failed: {e}")

            return paper

    def _clean_doi(self, doi: str) -> str:
        """Extract and clean DOI from various formats."""
        # Handle URLs
        if "doi.org/" in doi:
            doi = doi.split("doi.org/")[-1]

        # Remove URL encoding
        doi = doi.strip().strip("/")

        # Validate DOI format (10.xxxx/xxxxx)
        if re.match(r"^10\.\d{4,}/", doi):
            return doi

        return ""

    async def _fetch_from_crossref(self, session: aiohttp.ClientSession, doi: str) -> Optional[FetchedPaper]:
        """Fetch paper from CrossRef API."""
        url = f"{CROSSREF_API}/{quote(doi, safe='')}"

        data = await self._fetch_json(session, url)
        work = data.get("message", {})

        if not work:
            return None

        # Extract authors
        authors = []
        for author in work.get("author", []):
            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
            if name:
                authors.append(name)

        # Extract year
        date_parts = work.get("published-print", {}).get("date-parts", [[]])
        if not date_parts[0]:
            date_parts = work.get("published-online", {}).get("date-parts", [[]])
        year = date_parts[0][0] if date_parts and date_parts[0] else 0

        return FetchedPaper(
            id=doi,
            source="crossref",
            title=work.get("title", [""])[0] if work.get("title") else "",
            authors=authors,
            abstract=work.get("abstract", "").replace("<jats:p>", "").replace("</jats:p>", ""),
            journal=work.get("container-title", [""])[0] if work.get("container-title") else "",
            year=year,
            doi=doi,
            url=work.get("URL", ""),
            citation_count=work.get("is-referenced-by-count", 0),
        )

    async def _fetch_from_semantic_scholar(self, session: aiohttp.ClientSession, doi: str) -> Optional[FetchedPaper]:
        """Fetch paper from Semantic Scholar API."""
        url = f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{doi}"
        params = {
            "fields": "title,authors,abstract,year,venue,citationCount,referenceCount,externalIds,url"
        }

        data = await self._fetch_json(session, url, params)

        if not data or "paperId" not in data:
            return None

        return FetchedPaper(
            id=data.get("paperId", ""),
            source="semantic_scholar",
            title=data.get("title", ""),
            authors=[a.get("name", "") for a in data.get("authors", [])],
            abstract=data.get("abstract", "") or "",
            journal=data.get("venue", "") or "",
            year=data.get("year", 0) or 0,
            doi=data.get("externalIds", {}).get("DOI", ""),
            pmid=data.get("externalIds", {}).get("PubMed", ""),
            url=data.get("url", ""),
            citation_count=data.get("citationCount", 0),
        )

    async def _enrich_with_semantic_scholar(self, session: aiohttp.ClientSession, paper: FetchedPaper) -> FetchedPaper:
        """Enrich paper with Semantic Scholar data."""
        if not paper.doi:
            return paper

        url = f"{SEMANTIC_SCHOLAR_API}/paper/DOI:{paper.doi}"
        params = {"fields": "citationCount,referenceCount,references.title"}

        try:
            data = await self._fetch_json(session, url, params)

            paper.citation_count = data.get("citationCount", paper.citation_count)

            # Get reference titles
            refs = data.get("references", [])
            paper.references = [r.get("title", "") for r in refs[:10] if r.get("title")]

            paper._calculate_scores()
        except:
            pass

        return paper

    # ==================== PubMed Search ====================

    async def search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        sort: str = "relevance",
        min_date: str = None,
        max_date: str = None
    ) -> List[FetchedPaper]:
        """
        Search PubMed in real-time.

        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order ("relevance", "pub_date")
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)

        Returns:
            List of FetchedPaper objects
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Step 1: Search for PMIDs
            await self._rate_limit("pubmed")
            pmids = await self._search_pubmed_ids(session, query, max_results, sort, min_date, max_date)

            if not pmids:
                return []

            # Step 2: Fetch paper details
            await self._rate_limit("pubmed")
            papers = await self._fetch_pubmed_details(session, pmids)

            return papers

    async def search_pubmed_hybrid(
        self,
        query: str,
        max_results: int = 10
    ) -> List[FetchedPaper]:
        """
        Search PubMed with HYBRID approach (latest + high-impact).

        Combines:
        - 50% latest papers (sorted by pub_date, Title-focused search)
        - 50% high-impact papers (sorted by relevance)

        Args:
            query: Search query
            max_results: Total number of results to return

        Returns:
            List of FetchedPaper objects, mixed latest + high-impact
        """
        half_results = max(3, max_results // 2)

        # Add [Title] qualifier for more relevant results in latest search
        title_query = f"{query}[Title]"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # 1. Get LATEST papers (Title-focused, sorted by date)
            await self._rate_limit("pubmed")
            latest_pmids = await self._search_pubmed_ids(
                session, title_query, half_results + 3, "pub_date", "2024/01/01", None
            )

            # If not enough Title results, also search Title/Abstract
            if len(latest_pmids) < half_results:
                await self._rate_limit("pubmed")
                tiab_query = f"{query}[Title/Abstract]"
                extra_pmids = await self._search_pubmed_ids(
                    session, tiab_query, half_results + 3, "pub_date", "2024/01/01", None
                )
                for pmid in extra_pmids:
                    if pmid not in latest_pmids:
                        latest_pmids.append(pmid)

            # 2. Get HIGH-IMPACT papers (sorted by relevance)
            await self._rate_limit("pubmed")
            impact_pmids = await self._search_pubmed_ids(
                session, query, half_results + 3, "relevance", "2020/01/01", None
            )

            # Combine and deduplicate
            seen = set()
            combined_pmids = []

            # Add latest first (priority)
            for pmid in latest_pmids:
                if pmid not in seen:
                    seen.add(pmid)
                    combined_pmids.append(pmid)

            # Add high-impact
            for pmid in impact_pmids:
                if pmid not in seen:
                    seen.add(pmid)
                    combined_pmids.append(pmid)

            if not combined_pmids:
                return []

            # Fetch details
            await self._rate_limit("pubmed")
            papers = await self._fetch_pubmed_details(session, combined_pmids[:max_results + 4])

            # Filter: ensure query terms appear in title (case-insensitive)
            query_terms = query.lower().split()
            filtered_papers = []
            for paper in papers:
                title_lower = paper.title.lower()
                # At least one query term should be in title
                if any(term in title_lower for term in query_terms):
                    filtered_papers.append(paper)
                elif len(filtered_papers) < max_results:
                    # Include some that match in abstract if we don't have enough
                    if paper.abstract and any(term in paper.abstract.lower() for term in query_terms):
                        paper.trend_score -= 10  # Lower priority
                        filtered_papers.append(paper)

            # Boost latest papers
            latest_set = set(latest_pmids)
            for paper in filtered_papers:
                if paper.pmid in latest_set:
                    paper.trend_score += 20

            # Sort by (year, trend_score)
            filtered_papers.sort(key=lambda p: (p.year, p.trend_score), reverse=True)

            return filtered_papers[:max_results]

    async def _search_pubmed_ids(
        self,
        session: aiohttp.ClientSession,
        query: str,
        max_results: int,
        sort: str,
        min_date: str,
        max_date: str
    ) -> List[str]:
        """Search PubMed and return PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "sort": sort,
            "retmode": "json",
        }

        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"
        if max_date:
            params["maxdate"] = max_date
            params["datetype"] = "pdat"

        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key

        data = await self._fetch_json(session, PUBMED_SEARCH_URL, params)
        return data.get("esearchresult", {}).get("idlist", [])

    async def _fetch_pubmed_details(self, session: aiohttp.ClientSession, pmids: List[str]) -> List[FetchedPaper]:
        """Fetch detailed paper information from PubMed."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        }

        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key

        xml_text = await self._fetch_text(session, PUBMED_FETCH_URL, params)
        return self._parse_pubmed_xml(xml_text)

    def _parse_pubmed_xml(self, xml_text: str) -> List[FetchedPaper]:
        """Parse PubMed XML response."""
        papers = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
        except ET.ParseError as e:
            print(f"XML parse error: {e}")

        return papers

    def _parse_pubmed_article(self, article) -> Optional[FetchedPaper]:
        """Parse a single PubMed article."""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        pmid_elem = medline.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        if not pmid:
            return None

        article_elem = medline.find(".//Article")
        if article_elem is None:
            return None

        # Title
        title_elem = article_elem.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()) if title_elem is not None else ""

        # Abstract
        abstract_parts = []
        for abstract_text in article_elem.findall(".//AbstractText"):
            label = abstract_text.get("Label", "")
            text = "".join(abstract_text.itertext()) or ""
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)
        abstract = "\n".join(abstract_parts)

        # Authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last_name = author.find("LastName")
            fore_name = author.find("ForeName")
            if last_name is not None and fore_name is not None:
                authors.append(f"{fore_name.text} {last_name.text}")

        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Year
        year = 0
        year_elem = article_elem.find(".//PubDate/Year")
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text[:4])
            except ValueError:
                pass

        # DOI and PMC ID
        doi = ""
        pmcid = ""
        for id_elem in article.findall(".//ArticleId"):
            id_type = id_elem.get("IdType")
            if id_type == "doi":
                doi = id_elem.text or ""
            elif id_type == "pmc":
                pmcid = id_elem.text or ""

        # Keywords
        keywords = []
        for kw in medline.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)

        return FetchedPaper(
            id=pmid,
            source="pubmed",
            title=title,
            authors=authors,
            abstract=abstract,
            journal=journal,
            year=year,
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            keywords=keywords,
        )

    # ==================== Trending Papers ====================

    async def get_trending_papers(
        self,
        category: str = "oncology",
        max_results: int = 10,
        use_cache: bool = True,
        major_journals_only: bool = True
    ) -> List[FetchedPaper]:
        """
        Get trending papers from PubMed.

        Uses a HYBRID approach:
        - 50% Latest papers (sorted by date, 2025 priority)
        - 50% High-impact papers (sorted by relevance/citations)

        Args:
            category: Topic category (see TRENDING_CATEGORIES)
            max_results: Number of papers to return
            use_cache: Whether to use cached results
            major_journals_only: Filter to major journals only (Nature, Cell, Science, NEJM, Lancet, etc.)

        Returns:
            List of trending FetchedPaper objects, mixed latest + high-impact
        """
        cache_key = f"{category}_{max_results}_{'major' if major_journals_only else 'all'}"

        # Check cache
        if use_cache and cache_key in self._trending_cache:
            cached_data, cached_time = self._trending_cache[cache_key]
            if datetime.now() - cached_time < self._cache_expiry:
                return cached_data

        query = TRENDING_CATEGORIES.get(category, TRENDING_CATEGORIES["oncology"])

        # Add major journal filter if requested
        if major_journals_only:
            query = f"({query}) AND ({MAJOR_JOURNAL_FILTER})"

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Strategy: Fetch BOTH latest and high-impact papers
            half_results = max(3, max_results // 2)

            # 1. Get LATEST papers (sorted by date) - 2025 priority
            await self._rate_limit("pubmed")
            latest_papers = await self.search_pubmed(
                query=query,
                max_results=half_results + 2,
                sort="pub_date",  # Sort by publication date (newest first)
                min_date="2024/06/01"  # Recent 6 months to get 2025 papers
            )

            # 2. Get HIGH-IMPACT papers (sorted by relevance)
            await self._rate_limit("pubmed")
            impact_papers = await self.search_pubmed(
                query=query,
                max_results=half_results + 2,
                sort="relevance",  # Sort by relevance (high-cited)
                min_date="2024/01/01"
            )

            # Combine and deduplicate by PMID
            seen_pmids = set()
            all_papers = []

            # Add latest papers first (priority)
            for paper in latest_papers:
                if paper.pmid and paper.pmid not in seen_pmids:
                    seen_pmids.add(paper.pmid)
                    paper.trend_score += 20  # Boost for being latest
                    all_papers.append(paper)

            # Add high-impact papers
            for paper in impact_papers:
                if paper.pmid and paper.pmid not in seen_pmids:
                    seen_pmids.add(paper.pmid)
                    all_papers.append(paper)

            # Try to enrich with citation data from Semantic Scholar
            enriched_papers = []
            for paper in all_papers[:max_results + 2]:
                if paper.doi:
                    try:
                        await self._rate_limit("semantic_scholar")
                        paper = await self._enrich_with_semantic_scholar(session, paper)
                    except:
                        pass
                enriched_papers.append(paper)

            # Sort by combined score (trend_score already includes recency)
            enriched_papers.sort(key=lambda p: (p.year, p.trend_score), reverse=True)
            result = enriched_papers[:max_results]

            # Cache results
            self._trending_cache[cache_key] = (result, datetime.now())

            return result

    async def get_all_trending_categories(self, papers_per_category: int = 5) -> Dict[str, List[FetchedPaper]]:
        """
        Get trending papers for all categories.

        Args:
            papers_per_category: Number of papers per category

        Returns:
            Dict mapping category names to paper lists
        """
        results = {}

        for category in TRENDING_CATEGORIES.keys():
            try:
                papers = await self.get_trending_papers(
                    category=category,
                    max_results=papers_per_category
                )
                results[category] = papers
            except Exception as e:
                print(f"Error fetching {category}: {e}")
                results[category] = []

        return results

    # ==================== URL-based Fetching ====================

    async def fetch_by_url(self, url: str) -> Optional[FetchedPaper]:
        """
        Fetch paper from various URL formats.

        Supports:
        - DOI URLs (doi.org/...)
        - PubMed URLs (pubmed.ncbi.nlm.nih.gov/...)
        - PMC URLs (ncbi.nlm.nih.gov/pmc/articles/...)

        Args:
            url: Paper URL

        Returns:
            FetchedPaper or None
        """
        url = url.strip()

        # DOI URL
        if "doi.org/" in url:
            doi = url.split("doi.org/")[-1].strip("/")
            return await self.fetch_by_doi(doi)

        # PubMed URL
        if "pubmed.ncbi.nlm.nih.gov/" in url:
            pmid = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
            if pmid:
                papers = await self.search_pubmed(f"{pmid.group(1)}[uid]", max_results=1)
                return papers[0] if papers else None

        # PMC URL
        if "/pmc/articles/PMC" in url:
            pmcid = re.search(r"PMC(\d+)", url)
            if pmcid:
                papers = await self.search_pubmed(f"PMC{pmcid.group(1)}[pmcid]", max_results=1)
                return papers[0] if papers else None

        return None

    # ==================== Batch Operations ====================

    async def batch_fetch_dois(self, dois: List[str]) -> List[FetchedPaper]:
        """
        Fetch multiple papers by DOI.

        Args:
            dois: List of DOIs

        Returns:
            List of FetchedPaper objects
        """
        results = []

        for doi in dois:
            try:
                paper = await self.fetch_by_doi(doi)
                if paper:
                    results.append(paper)
            except Exception as e:
                print(f"Error fetching DOI {doi}: {e}")

        return results

    async def find_similar_papers(self, pmid: str, max_results: int = 10) -> List[FetchedPaper]:
        """
        Find similar papers using PubMed's Related Articles feature (elink).

        Args:
            pmid: Source paper PMID
            max_results: Maximum number of similar papers to return

        Returns:
            List of similar FetchedPaper objects
        """
        ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

        async with aiohttp.ClientSession() as session:
            await self._rate_limit("pubmed")

            # Step 1: Get related PMIDs using elink
            params = {
                "dbfrom": "pubmed",
                "db": "pubmed",
                "id": pmid,
                "cmd": "neighbor_score",  # Get related articles with scores
                "retmode": "json",
            }
            # Only add api_key if it exists
            if self.ncbi_api_key:
                params["api_key"] = self.ncbi_api_key

            try:
                data = await self._fetch_json(session, ELINK_URL, params)

                # Parse related PMIDs from response
                related_pmids = []

                if "linksets" in data:
                    for linkset in data["linksets"]:
                        if "linksetdbs" in linkset:
                            for linksetdb in linkset["linksetdbs"]:
                                if linksetdb.get("linkname") == "pubmed_pubmed":
                                    links = linksetdb.get("links", [])
                                    # links are dicts with {"id": "...", "score": ...}
                                    # Extract PMIDs as strings
                                    related_pmids = [
                                        str(link["id"]) if isinstance(link, dict) else str(link)
                                        for link in links[:max_results]
                                    ]
                                    break

                if not related_pmids:
                    # Fallback: try simple related query
                    return await self._fallback_similar_search(session, pmid, max_results)

                # Step 2: Fetch details for related papers
                papers = await self._fetch_pubmed_details(session, related_pmids)

                # Calculate similarity score based on position (higher = more similar)
                for i, paper in enumerate(papers):
                    # Score from 95 (most similar) down to 50
                    paper.trend_score = max(50, 95 - (i * 5))

                return papers

            except Exception as e:
                print(f"Error finding similar papers: {e}")
                # Fallback to keyword search
                return await self._fallback_similar_search(session, pmid, max_results)

    async def _fallback_similar_search(
        self,
        session: aiohttp.ClientSession,
        pmid: str,
        max_results: int
    ) -> List[FetchedPaper]:
        """Fallback: search by title keywords if elink fails."""
        try:
            # First get the source paper
            source_papers = await self._fetch_pubmed_details(session, [pmid])
            if not source_papers:
                return []

            source = source_papers[0]

            # Extract keywords from title
            title_words = source.title.split()
            # Take important words (skip common words)
            skip_words = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to", "with", "by"}
            keywords = [w for w in title_words if w.lower() not in skip_words and len(w) > 2][:5]

            if not keywords:
                return []

            # Search for similar papers
            query = " ".join(keywords)
            papers = await self.search_pubmed(query, max_results=max_results + 1)

            # Remove source paper from results
            papers = [p for p in papers if p.pmid != pmid][:max_results]

            return papers

        except Exception as e:
            print(f"Fallback similar search failed: {e}")
            return []

    # ==================== Full Text Retrieval ====================

    async def fetch_full_text(self, pmid: str = None, pmcid: str = None, doi: str = None, use_playwright: bool = True) -> Optional[str]:
        """
        Fetch full text content from various sources.

        Tries multiple methods in order:
        1. PMC OA API for open access papers (if pmcid available)
        2. Europe PMC for full text XML
        3. Unpaywall API (finds legal free versions by DOI)
        4. PubMed full text links → Publisher site via Playwright

        Args:
            pmid: PubMed ID
            pmcid: PMC ID (e.g., "PMC1234567")
            doi: DOI (for fallback lookup)
            use_playwright: Whether to try Playwright for publisher sites

        Returns:
            Full text string or None if not available
        """
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            # Method 1: Try PMC OA API (for open access papers)
            if pmcid:
                full_text = await self._fetch_from_pmc_oa(session, pmcid)
                if full_text:
                    return full_text

            # Method 2: Try Europe PMC
            if pmid or pmcid:
                full_text = await self._fetch_from_europe_pmc(session, pmid, pmcid)
                if full_text:
                    return full_text

            # Method 3: Try Unpaywall API (finds legal free versions)
            if doi:
                full_text = await self._fetch_from_unpaywall(session, doi)
                if full_text:
                    return full_text

            # Method 4: If we only have DOI, try to get PMCID first
            if doi and not pmcid:
                paper = await self.fetch_by_doi(doi)
                if paper and paper.pmcid:
                    return await self.fetch_full_text(pmcid=paper.pmcid, use_playwright=False)

            # Method 5: Try Playwright to crawl publisher site via PubMed links
            if use_playwright and pmid:
                full_text = await self._fetch_via_pubmed_links(pmid)
                if full_text:
                    return full_text

            return None

    async def _fetch_from_pmc_oa(self, session: aiohttp.ClientSession, pmcid: str) -> Optional[str]:
        """Fetch full text from PMC Open Access subset."""
        try:
            # Clean PMCID format
            if not pmcid.startswith("PMC"):
                pmcid = f"PMC{pmcid}"

            # Get OA file location
            params = {"id": pmcid, "format": "tgz"}
            await self._rate_limit("pubmed")

            async with session.get(PMC_OA_URL, params=params) as response:
                if response.status != 200:
                    return None

                xml_text = await response.text()
                root = ET.fromstring(xml_text)

                # Check if full text available
                record = root.find(".//record")
                if record is None:
                    return None

                # Get full text XML URL
                link = record.find(".//link[@format='xml']")
                if link is None:
                    link = record.find(".//link[@format='pdf']")

                if link is None:
                    return None

                xml_url = link.get("href")
                if not xml_url:
                    return None

                # Fetch the full text XML
                async with session.get(xml_url) as xml_response:
                    if xml_response.status == 200:
                        content_type = xml_response.content_type
                        if 'xml' in content_type:
                            xml_content = await xml_response.text()
                            return self._parse_pmc_xml(xml_content)

            return None

        except Exception as e:
            print(f"PMC OA fetch error: {e}")
            return None

    async def _fetch_from_europe_pmc(self, session: aiohttp.ClientSession, pmid: str = None, pmcid: str = None) -> Optional[str]:
        """Fetch full text from Europe PMC."""
        try:
            # Build query
            if pmcid:
                source = "PMC"
                ext_id = pmcid.replace("PMC", "")
            elif pmid:
                source = "MED"
                ext_id = pmid
            else:
                return None

            url = f"{EUROPE_PMC_API}/{source}/{ext_id}/fullTextXML"

            async with session.get(url) as response:
                if response.status != 200:
                    return None

                xml_text = await response.text()
                return self._parse_pmc_xml(xml_text)

        except Exception as e:
            print(f"Europe PMC fetch error: {e}")
            return None

    def _parse_pmc_xml(self, xml_text: str) -> str:
        """Parse PMC/JATS XML and extract text content."""
        try:
            root = ET.fromstring(xml_text)

            sections = []

            # Extract title
            title = root.find(".//article-title")
            if title is not None:
                title_text = "".join(title.itertext()).strip()
                if title_text:
                    sections.append(f"# {title_text}\n")

            # Extract abstract
            abstract = root.find(".//abstract")
            if abstract is not None:
                abstract_text = "".join(abstract.itertext()).strip()
                if abstract_text:
                    sections.append(f"## Abstract\n{abstract_text}\n")

            # Extract body sections
            body = root.find(".//body")
            if body is not None:
                for sec in body.findall(".//sec"):
                    section_title = sec.find("title")
                    title_text = "".join(section_title.itertext()).strip() if section_title is not None else ""

                    # Get paragraphs
                    paragraphs = []
                    for p in sec.findall(".//p"):
                        p_text = "".join(p.itertext()).strip()
                        if p_text:
                            paragraphs.append(p_text)

                    if paragraphs:
                        if title_text:
                            sections.append(f"## {title_text}\n" + "\n\n".join(paragraphs))
                        else:
                            sections.append("\n\n".join(paragraphs))

            # Also get any remaining paragraphs in body not in sections
            if body is not None:
                for p in body.findall("./p"):
                    p_text = "".join(p.itertext()).strip()
                    if p_text:
                        sections.append(p_text)

            full_text = "\n\n".join(sections)

            # Clean up whitespace
            import re
            full_text = re.sub(r'\n{3,}', '\n\n', full_text)
            full_text = re.sub(r' {2,}', ' ', full_text)

            return full_text.strip() if len(full_text) > 100 else None

        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return None

    async def _fetch_from_unpaywall(self, session: aiohttp.ClientSession, doi: str) -> Optional[str]:
        """
        Fetch full text using Unpaywall API.
        Unpaywall finds legal free versions of papers.
        """
        try:
            # Unpaywall API requires an email
            url = f"https://api.unpaywall.org/v2/{doi}?email=bioinsight@example.com"

            async with session.get(url) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                # Check for open access version
                best_oa = data.get("best_oa_location")
                if not best_oa:
                    return None

                # Get the PDF or HTML URL
                oa_url = best_oa.get("url_for_pdf") or best_oa.get("url")
                if not oa_url:
                    return None

                # If it's a PMC URL, extract via Europe PMC
                if "ncbi.nlm.nih.gov/pmc" in oa_url:
                    pmcid_match = re.search(r"PMC(\d+)", oa_url)
                    if pmcid_match:
                        return await self._fetch_from_europe_pmc(session, None, f"PMC{pmcid_match.group(1)}")

                # If it's a direct PDF/HTML, we need Playwright to extract
                # Return the URL for now, the caller can use Playwright
                print(f"Unpaywall found OA version: {oa_url}")

                # Try to fetch if it's XML/HTML
                if not oa_url.endswith('.pdf'):
                    try:
                        async with session.get(oa_url) as oa_response:
                            if oa_response.status == 200:
                                content_type = oa_response.headers.get('content-type', '')
                                if 'xml' in content_type:
                                    xml_text = await oa_response.text()
                                    return self._parse_pmc_xml(xml_text)
                    except:
                        pass

                return None

        except Exception as e:
            print(f"Unpaywall fetch error: {e}")
            return None

    async def _fetch_via_pubmed_links(self, pmid: str) -> Optional[str]:
        """
        Fetch full text by following PubMed's 'Full text links' to publisher site.
        Uses Playwright to navigate and extract content.
        """
        try:
            # Check if Playwright is available
            from .playwright_crawler import PlaywrightDeepCrawler, PLAYWRIGHT_AVAILABLE
            if not PLAYWRIGHT_AVAILABLE:
                return None

            crawler = PlaywrightDeepCrawler(headless=True)
            try:
                # First, get the full text link from PubMed page
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                publisher_url = await self._get_pubmed_fulltext_link(crawler, pubmed_url)

                if not publisher_url:
                    return None

                print(f"Found publisher link: {publisher_url}")

                # Extract full text from publisher site
                result = await crawler.extract_full_text(publisher_url)

                if result.success and result.full_text:
                    # Clean up the text
                    text = result.full_text.strip()
                    if len(text) > 500:  # Minimum viable full text
                        return text

                return None

            finally:
                await crawler.close()

        except ImportError:
            return None
        except Exception as e:
            print(f"PubMed links fetch error: {e}")
            return None

    async def _get_pubmed_fulltext_link(self, crawler, pubmed_url: str) -> Optional[str]:
        """Extract full text link from PubMed page."""
        try:
            await crawler._ensure_browser()
            page = await crawler._context.new_page()

            try:
                await page.goto(pubmed_url, wait_until="domcontentloaded", timeout=30000)

                # Wait for the full text links section
                await page.wait_for_selector("#full-text-links-list, .full-text-links", timeout=10000)

                # Find the first full text link (usually the best one)
                # PubMed shows links like "Free PMC article", "Free article", etc.
                link_selectors = [
                    "#full-text-links-list a.link-item",
                    ".full-text-links a",
                    "a[data-ga-action='full_text']",
                    ".linkout-category-links a",
                ]

                for selector in link_selectors:
                    link = await page.query_selector(selector)
                    if link:
                        href = await link.get_attribute("href")
                        if href:
                            # Make absolute URL
                            if href.startswith("/"):
                                href = f"https://pubmed.ncbi.nlm.nih.gov{href}"
                            return href

                return None

            finally:
                await page.close()

        except Exception as e:
            print(f"Error getting PubMed full text link: {e}")
            return None

    # ==================== Indexing Integration ====================

    def paper_to_chunks(self, paper: FetchedPaper) -> List[TextChunk]:
        """
        Convert a fetched paper to text chunks for vector indexing.

        Args:
            paper: FetchedPaper object

        Returns:
            List of TextChunk objects
        """
        chunks = []

        metadata_base = {
            "paper_id": paper.id,
            "paper_title": paper.title,
            "authors": ", ".join(paper.authors[:3]) + ("..." if len(paper.authors) > 3 else ""),
            "journal": paper.journal,
            "year": paper.year,
            "doi": paper.doi,
            "pmid": paper.pmid,
            "source": paper.source,
            "url": paper.url,
            "trend_score": paper.trend_score,
        }

        # Add title as a chunk
        if paper.title:
            chunks.append(TextChunk(
                content=f"Title: {paper.title}",
                metadata={**metadata_base, "section": "Title"}
            ))

        # Add abstract as a chunk
        if paper.abstract:
            chunks.append(TextChunk(
                content=paper.abstract,
                metadata={**metadata_base, "section": "Abstract"}
            ))

        # Split full text if available
        if paper.full_text:
            text_chunks = self.text_splitter.split_text_simple(
                paper.full_text,
                metadata={**metadata_base, "section": "Full Text"}
            )
            chunks.extend(text_chunks)

        return chunks


# ==================== Convenience Functions ====================

async def fetch_paper_by_doi(doi: str) -> Optional[FetchedPaper]:
    """Convenience function to fetch a paper by DOI."""
    agent = WebCrawlerAgent()
    return await agent.fetch_by_doi(doi)


async def search_papers(query: str, max_results: int = 10) -> List[FetchedPaper]:
    """Convenience function to search PubMed."""
    agent = WebCrawlerAgent()
    return await agent.search_pubmed(query, max_results=max_results)


async def get_trending(category: str = "oncology", count: int = 10) -> List[FetchedPaper]:
    """Convenience function to get trending papers."""
    agent = WebCrawlerAgent()
    return await agent.get_trending_papers(category=category, max_results=count)


# CLI for testing
if __name__ == "__main__":
    import sys

    async def main():
        agent = WebCrawlerAgent()

        if len(sys.argv) > 1:
            cmd = sys.argv[1]

            if cmd == "doi" and len(sys.argv) > 2:
                paper = await agent.fetch_by_doi(sys.argv[2])
                if paper:
                    print(json.dumps(paper.to_dict(), indent=2, default=str))
                else:
                    print("Paper not found")

            elif cmd == "search" and len(sys.argv) > 2:
                query = " ".join(sys.argv[2:])
                papers = await agent.search_pubmed(query, max_results=5)
                for p in papers:
                    print(f"[{p.pmid}] {p.title[:80]}... ({p.year})")

            elif cmd == "trending":
                category = sys.argv[2] if len(sys.argv) > 2 else "oncology"
                papers = await agent.get_trending_papers(category=category)
                print(f"\nTrending in {category}:")
                for i, p in enumerate(papers, 1):
                    print(f"{i}. [{p.trend_score:.1f}] {p.title[:70]}... ({p.year})")

            elif cmd == "categories":
                print("Available trending categories:")
                for cat in TRENDING_CATEGORIES.keys():
                    print(f"  - {cat}")

            else:
                print("Usage:")
                print("  python web_crawler_agent.py doi <DOI>")
                print("  python web_crawler_agent.py search <query>")
                print("  python web_crawler_agent.py trending [category]")
                print("  python web_crawler_agent.py categories")
        else:
            # Demo
            print("=== Web Crawler Agent Demo ===\n")

            print("1. Fetching paper by DOI...")
            paper = await agent.fetch_by_doi("10.1038/s41586-023-06747-5")
            if paper:
                print(f"   Title: {paper.title[:60]}...")
                print(f"   Citations: {paper.citation_count}")

            print("\n2. Searching PubMed for 'CRISPR cancer'...")
            papers = await agent.search_pubmed("CRISPR cancer therapy", max_results=3)
            for p in papers:
                print(f"   - {p.title[:60]}...")

            print("\n3. Getting trending oncology papers...")
            trending = await agent.get_trending_papers("oncology", max_results=3)
            for p in trending:
                print(f"   [{p.trend_score:.0f}] {p.title[:50]}...")

    asyncio.run(main())
