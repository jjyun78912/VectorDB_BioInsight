"""
Paper Recommender for RNA-seq Analysis Reports.

Recommends relevant papers based on:
1. Cancer type
2. Top hub genes
3. Key pathways

Uses PubMed E-utilities API for real-time search.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

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


class PaperRecommender:
    """
    Recommends relevant papers for RNA-seq analysis results.

    Uses PubMed API to find papers related to:
    - Specific cancer type
    - Top differentially expressed genes
    - Enriched pathways
    """

    def __init__(
        self,
        cancer_type: str = "",
        email: str = "bioinsight@research.ai",
        api_key: Optional[str] = None
    ):
        self.cancer_type = cancer_type
        self.email = email
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=30)
        self._last_request_time = 0
        self._rate_limit_delay = 0.4  # 400ms between requests (NCBI recommendation)

    async def _rate_limit(self):
        """Ensure we don't exceed PubMed rate limits."""
        import time
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

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
            # Focus on top genes
            top_genes = genes[:5]
            gene_query = " OR ".join([f"{g}[Title/Abstract]" for g in top_genes])
            query = f"({cancer_term}[Title/Abstract]) AND ({gene_query}) AND (RNA-seq OR transcriptome OR gene expression)"

        elif focus == "pathways" and pathways:
            # Focus on pathways
            top_pathways = pathways[:3]
            pathway_terms = " OR ".join([f'"{p}"[Title/Abstract]' for p in top_pathways])
            query = f"({cancer_term}[Title/Abstract]) AND ({pathway_terms})"

        else:
            # General overview
            query = f"({cancer_term}[Title/Abstract]) AND (RNA-seq OR transcriptome) AND (biomarker OR therapeutic target OR prognosis)"

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
    max_papers: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function to get paper recommendations for RNA-seq analysis.

    Args:
        cancer_type: TCGA code or cancer name
        hub_genes: List of hub genes from analysis
        pathways: List of enriched pathways
        output_dir: Optional output directory to save results
        max_papers: Maximum papers to recommend

    Returns:
        List of paper dictionaries
    """
    recommender = PaperRecommender(cancer_type=cancer_type)
    papers = await recommender.recommend_papers(hub_genes, pathways, max_papers)

    if output_dir:
        output_path = Path(output_dir) / "recommended_papers.json"
        recommender.save_recommendations(papers, output_path)

    return [p.to_dict() for p in papers]


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
