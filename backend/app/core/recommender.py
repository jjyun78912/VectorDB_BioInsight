"""
Paper Recommendation Service.

Features:
- Keyword-based paper search
- Similar paper recommendations based on indexed PDF
- Scoring based on: similarity, recency, impact (citations)
- Links to PubMed, DOI, and full-text sources
"""
import os
import re
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import time

from .config import GOOGLE_API_KEY, GEMINI_MODEL
from .vector_store import create_vector_store
from .pdf_parser import BioPaperParser
from .embeddings import get_embedder


# API Configuration
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# PubMed E-utilities base URL
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1"


@dataclass
class RecommendedPaper:
    """A recommended paper with metadata and scores."""
    title: str
    authors: list[str] = field(default_factory=list)
    year: int = 0
    abstract: str = ""
    doi: str = ""
    pmid: str = ""
    journal: str = ""
    citation_count: int = 0

    # Scores (0-100)
    similarity_score: float = 0.0
    recency_score: float = 0.0
    impact_score: float = 0.0
    total_score: float = 0.0

    # Links
    pubmed_url: str = ""
    doi_url: str = ""
    pmc_url: str = ""

    # Category info
    is_emerging_field: bool = False  # 연구가 더딘 분야 (과거 논문도 OK)

    def format(self, index: int) -> str:
        """Format paper for display."""
        output = []
        output.append(f"\n[{index}] 📄 {self.title}")
        output.append(f"    Authors: {', '.join(self.authors[:3])}{'...' if len(self.authors) > 3 else ''}")
        output.append(f"    Journal: {self.journal} ({self.year})")
        output.append(f"    Citations: {self.citation_count}")

        # Score bar
        score_bar = "█" * int(self.total_score / 5) + "░" * (20 - int(self.total_score / 5))
        output.append(f"    Score: [{score_bar}] {self.total_score:.1f}")
        output.append(f"      - Similarity: {self.similarity_score:.1f} | Recency: {self.recency_score:.1f} | Impact: {self.impact_score:.1f}")

        # Links
        links = []
        if self.pubmed_url:
            links.append(f"PubMed: {self.pubmed_url}")
        if self.doi_url:
            links.append(f"DOI: {self.doi_url}")
        if self.pmc_url:
            links.append(f"PMC: {self.pmc_url}")
        if links:
            output.append(f"    🔗 Links:")
            for link in links:
                output.append(f"       {link}")

        return "\n".join(output)


class PaperRecommender:
    """
    Service for recommending related papers.

    Uses PubMed and Semantic Scholar APIs to find relevant papers
    based on keywords or similarity to indexed papers.
    """

    def __init__(
        self,
        disease_domain: str | None = None,
        ncbi_api_key: str | None = None,
    ):
        """
        Initialize recommender.

        Args:
            disease_domain: Disease domain for vector store access
            ncbi_api_key: NCBI API key for PubMed access
        """
        self.disease_domain = disease_domain
        self.ncbi_api_key = ncbi_api_key or NCBI_API_KEY

        # Initialize embedder for similarity calculation
        self.embedder = get_embedder()

        # Initialize vector store if domain specified
        self.vector_store = None
        if disease_domain:
            self.vector_store = create_vector_store(disease_domain=disease_domain)

    def recommend_by_keyword(
        self,
        keywords: str,
        max_results: int = 10,
        min_year: int | None = None,
        sort_by: str = "relevance"  # relevance, date, citations
    ) -> list[RecommendedPaper]:
        """
        Recommend papers based on keyword search.

        Args:
            keywords: Search keywords
            max_results: Maximum number of results
            min_year: Minimum publication year
            sort_by: Sorting criteria

        Returns:
            List of recommended papers
        """
        # Search PubMed
        papers = self._search_pubmed(keywords, max_results * 2, min_year)

        # Enrich with citation data from Semantic Scholar
        papers = self._enrich_with_citations(papers)

        # Calculate scores
        for paper in papers:
            paper.similarity_score = self._calculate_keyword_similarity(keywords, paper)
            paper.recency_score = self._calculate_recency_score(paper.year)
            paper.impact_score = self._calculate_impact_score(paper.citation_count)
            paper.total_score = self._calculate_total_score(paper)

        # Sort and return top results
        papers = sorted(papers, key=lambda p: p.total_score, reverse=True)
        return papers[:max_results]

    def recommend_by_paper(
        self,
        paper_title: str | None = None,
        pdf_path: str | None = None,
        max_results: int = 10
    ) -> list[RecommendedPaper]:
        """
        Recommend papers similar to an indexed paper or PDF.

        Args:
            paper_title: Title of indexed paper
            pdf_path: Path to PDF file
            max_results: Maximum number of results

        Returns:
            List of recommended papers
        """
        # Get content to base recommendations on
        if pdf_path:
            parser = BioPaperParser()
            metadata, sections = parser.parse_pdf(pdf_path)
            # Combine abstract and title for base content
            base_content = metadata.abstract + " " + metadata.title
            # Add content from first few sections
            for section in sections[:3]:
                base_content += " " + section.content[:500]
            base_keywords = self._extract_keywords(base_content)
        elif paper_title and self.vector_store:
            # Get paper content from vector store
            results = self.vector_store.collection.get(
                where={"paper_title": paper_title},
                include=["documents", "metadatas"]
            )
            if results["documents"]:
                base_content = " ".join(results["documents"][:5])
                base_keywords = self._extract_keywords(base_content)
            else:
                raise ValueError(f"Paper not found: {paper_title}")
        else:
            raise ValueError("Provide either paper_title or pdf_path")

        # Search for similar papers
        papers = self._search_pubmed(base_keywords, max_results * 2)

        # Enrich with citations
        papers = self._enrich_with_citations(papers)

        # Calculate similarity using embeddings
        base_embedding = self.embedder.embed_query(base_content[:2000])

        for paper in papers:
            paper_text = f"{paper.title} {paper.abstract}"
            paper_embedding = self.embedder.embed_query(paper_text[:2000])
            paper.similarity_score = self._cosine_similarity(base_embedding, paper_embedding) * 100
            paper.recency_score = self._calculate_recency_score(paper.year)
            paper.impact_score = self._calculate_impact_score(paper.citation_count)
            paper.total_score = self._calculate_total_score(paper)

        # Sort and return
        papers = sorted(papers, key=lambda p: p.total_score, reverse=True)
        return papers[:max_results]

    def _search_pubmed(
        self,
        query: str,
        max_results: int = 20,
        min_year: int | None = None
    ) -> list[RecommendedPaper]:
        """Search PubMed for papers."""
        papers = []

        # Build query
        search_query = query
        if min_year:
            search_query += f" AND {min_year}:{datetime.now().year}[pdat]"

        # Add API key if available
        params = {
            "db": "pubmed",
            "term": search_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key

        try:
            # Search for IDs
            response = requests.get(f"{PUBMED_BASE_URL}/esearch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            pmids = data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return papers

            # Fetch details
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract"
            }
            if self.ncbi_api_key:
                fetch_params["api_key"] = self.ncbi_api_key

            time.sleep(0.34)  # Rate limiting (3 requests/second without API key)

            response = requests.get(f"{PUBMED_BASE_URL}/efetch.fcgi", params=fetch_params, timeout=30)
            response.raise_for_status()

            # Parse XML response
            papers = self._parse_pubmed_xml(response.text)

        except Exception as e:
            print(f"PubMed search error: {e}")

        return papers

    def _parse_pubmed_xml(self, xml_text: str) -> list[RecommendedPaper]:
        """Parse PubMed XML response."""
        import xml.etree.ElementTree as ET

        papers = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                paper = RecommendedPaper(title="")

                # Title
                title_elem = article.find(".//ArticleTitle")
                if title_elem is not None and title_elem.text:
                    paper.title = title_elem.text

                # Authors
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    if last_name is not None and last_name.text:
                        name = last_name.text
                        if fore_name is not None and fore_name.text:
                            name = f"{fore_name.text} {name}"
                        authors.append(name)
                paper.authors = authors

                # Year
                year_elem = article.find(".//PubDate/Year")
                if year_elem is not None and year_elem.text:
                    try:
                        paper.year = int(year_elem.text)
                    except ValueError:
                        paper.year = datetime.now().year

                # Abstract
                abstract_parts = []
                for abstract in article.findall(".//AbstractText"):
                    if abstract.text:
                        abstract_parts.append(abstract.text)
                paper.abstract = " ".join(abstract_parts)

                # Journal
                journal_elem = article.find(".//Journal/Title")
                if journal_elem is not None and journal_elem.text:
                    paper.journal = journal_elem.text

                # PMID
                pmid_elem = article.find(".//PMID")
                if pmid_elem is not None and pmid_elem.text:
                    paper.pmid = pmid_elem.text
                    paper.pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/"

                # DOI
                for article_id in article.findall(".//ArticleId"):
                    if article_id.get("IdType") == "doi":
                        paper.doi = article_id.text or ""
                        if paper.doi:
                            paper.doi_url = f"https://doi.org/{paper.doi}"
                    elif article_id.get("IdType") == "pmc":
                        pmc_id = article_id.text or ""
                        if pmc_id:
                            paper.pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"

                if paper.title:
                    papers.append(paper)

        except ET.ParseError as e:
            print(f"XML parsing error: {e}")

        return papers

    def _enrich_with_citations(self, papers: list[RecommendedPaper]) -> list[RecommendedPaper]:
        """Enrich papers with citation counts from Semantic Scholar."""
        for paper in papers:
            if paper.doi:
                try:
                    headers = {}
                    if SEMANTIC_SCHOLAR_API_KEY:
                        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

                    response = requests.get(
                        f"{SEMANTIC_SCHOLAR_URL}/paper/DOI:{paper.doi}",
                        params={"fields": "citationCount,influentialCitationCount"},
                        headers=headers,
                        timeout=10
                    )

                    if response.status_code == 200:
                        data = response.json()
                        paper.citation_count = data.get("citationCount", 0) or 0

                    time.sleep(0.1)  # Rate limiting

                except Exception:
                    pass  # Continue without citation data

        return papers

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> str:
        """Extract keywords from text using simple TF approach."""
        # Remove common words and extract significant terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'their', 'we', 'our', 'you', 'your',
            'which', 'who', 'whom', 'what', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'just', 'should', 'now', 'also',
            'study', 'studies', 'patients', 'patient', 'results', 'conclusion',
            'methods', 'background', 'objective', 'objectives', 'however',
            'using', 'used', 'found', 'shown', 'showed', 'significantly',
            'associated', 'compared', 'between', 'among', 'during', 'after',
            'before', 'through', 'into', 'over', 'under', 'about', 'while'
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:max_keywords]]

        return " ".join(keywords)

    def _calculate_keyword_similarity(self, keywords: str, paper: RecommendedPaper) -> float:
        """Calculate keyword-based similarity score."""
        keywords_lower = set(keywords.lower().split())
        paper_text = f"{paper.title} {paper.abstract}".lower()

        matches = sum(1 for kw in keywords_lower if kw in paper_text)
        return min(100, (matches / max(len(keywords_lower), 1)) * 100)

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _calculate_recency_score(self, year: int) -> float:
        """
        Calculate recency score.
        - 2024-2025: 100
        - 2022-2023: 80
        - 2020-2021: 60
        - 2018-2019: 40
        - Older: 20 (but still valuable for emerging fields)
        """
        current_year = datetime.now().year
        age = current_year - year

        if age <= 1:
            return 100
        elif age <= 3:
            return 80
        elif age <= 5:
            return 60
        elif age <= 7:
            return 40
        else:
            return 20

    def _calculate_impact_score(self, citations: int) -> float:
        """
        Calculate impact score based on citations.
        Uses logarithmic scale to handle wide range.
        """
        import math

        if citations <= 0:
            return 10  # New papers get base score
        elif citations < 10:
            return 30
        elif citations < 50:
            return 50
        elif citations < 100:
            return 70
        elif citations < 500:
            return 85
        else:
            return 100

    def _calculate_total_score(
        self,
        paper: RecommendedPaper,
        weights: dict = None
    ) -> float:
        """
        Calculate weighted total score.

        Default weights:
        - Similarity: 40%
        - Recency: 30%
        - Impact: 30%
        """
        if weights is None:
            weights = {
                "similarity": 0.40,
                "recency": 0.30,
                "impact": 0.30
            }

        total = (
            paper.similarity_score * weights["similarity"] +
            paper.recency_score * weights["recency"] +
            paper.impact_score * weights["impact"]
        )

        return total


def create_recommender(
    disease_domain: str | None = None,
    **kwargs
) -> PaperRecommender:
    """Convenience function to create a recommender."""
    return PaperRecommender(disease_domain=disease_domain, **kwargs)
