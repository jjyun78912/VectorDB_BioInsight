"""
PubMed Paper Fetcher
Fetches recent biomedical papers from PubMed using E-utilities API.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import httpx
from xml.etree import ElementTree as ET

# PubMed E-utilities base URL
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass
class Paper:
    """Represents a PubMed paper."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    doi: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "keywords": self.keywords,
            "mesh_terms": self.mesh_terms,
            "doi": self.doi,
        }


class PubMedFetcher:
    """Fetches papers from PubMed API."""

    # Broad biomedical search query - expanded to cover all predefined hot topics
    DEFAULT_QUERY = (
        # Cancer & Immunotherapy
        "(cancer[tiab] OR immunotherapy[tiab] OR checkpoint inhibitor[tiab] OR "
        "PD-1[tiab] OR PD-L1[tiab] OR CAR-T[tiab] OR CAR T cell[tiab] OR "
        "bispecific antibody[tiab] OR ADC[tiab] OR antibody-drug conjugate[tiab] OR "
        # Gene & Cell Therapy
        "gene therapy[tiab] OR CRISPR[tiab] OR gene editing[tiab] OR "
        "cell therapy[tiab] OR stem cell[tiab] OR iPSC[tiab] OR NK cell therapy[tiab] OR "
        # mRNA & LNP
        "mRNA[tiab] OR mRNA vaccine[tiab] OR lipid nanoparticle[tiab] OR LNP[tiab] OR "
        "siRNA[tiab] OR antisense oligonucleotide[tiab] OR exosome[tiab] OR "
        # GLP-1 & Metabolic
        "GLP-1[tiab] OR semaglutide[tiab] OR tirzepatide[tiab] OR "
        "obesity[tiab] OR weight loss drug[tiab] OR "
        # Neurology
        "Alzheimer[tiab] OR amyloid[tiab] OR tau protein[tiab] OR "
        "neurodegeneration[tiab] OR "
        # AI & Drug Discovery
        "AlphaFold[tiab] OR AI drug discovery[tiab] OR machine learning drug[tiab] OR "
        "drug discovery[tiab] OR precision medicine[tiab] OR "
        # Microbiome & Others
        "microbiome[tiab] OR gut microbiota[tiab] OR radiopharmaceutical[tiab] OR "
        # General terms
        "clinical trial[tiab] OR biomarker[tiab])"
    )

    def __init__(
        self,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        query: Optional[str] = None,
    ):
        self.email = email or os.getenv("PUBMED_EMAIL", "bioinsight@example.com")
        self.api_key = api_key or os.getenv("PUBMED_API_KEY")
        self.query = query or self.DEFAULT_QUERY

    def _build_date_filter(self, days: int = 7) -> str:
        """Build date range filter for PubMed search."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return f'("{start_date.strftime("%Y/%m/%d")}"[PDAT] : "{end_date.strftime("%Y/%m/%d")}"[PDAT])'

    async def search(
        self,
        max_results: int = 100,
        days: int = 7,
        custom_query: Optional[str] = None,
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs.

        Args:
            max_results: Maximum number of results to return
            days: Number of days to look back
            custom_query: Optional custom search query

        Returns:
            List of PMIDs
        """
        query = custom_query or self.query
        date_filter = self._build_date_filter(days)
        full_query = f"{query} AND {date_filter}"

        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub_date",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params)
            response.raise_for_status()

            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

        return pmids

    async def fetch_details(self, pmids: List[str]) -> List[Paper]:
        """
        Fetch detailed information for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of Paper objects
        """
        if not pmids:
            return []

        # Fetch in batches of 100
        papers = []
        batch_size = 100

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_papers = await self._fetch_batch(batch)
            papers.extend(batch_papers)

            # Rate limiting
            if i + batch_size < len(pmids):
                await asyncio.sleep(0.5)

        return papers

    async def _fetch_batch(self, pmids: List[str]) -> List[Paper]:
        """Fetch a batch of papers by PMID."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email,
        }

        if self.api_key:
            params["api_key"] = self.api_key

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params)
            response.raise_for_status()

        return self._parse_xml(response.text)

    def _parse_xml(self, xml_text: str) -> List[Paper]:
        """Parse PubMed XML response into Paper objects."""
        papers = []
        root = ET.fromstring(xml_text)

        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                print(f"Error parsing article: {e}")
                continue

        return papers

    def _parse_article(self, article: ET.Element) -> Optional[Paper]:
        """Parse a single article element."""
        medline = article.find(".//MedlineCitation")
        if medline is None:
            return None

        # PMID
        pmid_elem = medline.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        # Article info
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
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors = []
        for author in article_elem.findall(".//Author"):
            last_name = author.findtext("LastName", "")
            fore_name = author.findtext("ForeName", "")
            if last_name:
                authors.append(f"{fore_name} {last_name}".strip())

        # Journal
        journal_elem = article_elem.find(".//Journal/Title")
        journal = journal_elem.text if journal_elem is not None else ""

        # Publication date
        pub_date_elem = article_elem.find(".//PubDate")
        pub_date = ""
        if pub_date_elem is not None:
            year = pub_date_elem.findtext("Year", "")
            month = pub_date_elem.findtext("Month", "")
            pub_date = f"{year}-{month}" if month else year

        # Keywords
        keywords = []
        for kw in article_elem.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)

        # MeSH terms
        mesh_terms = []
        for mesh in medline.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)

        # DOI
        doi = None
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break

        return Paper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            pub_date=pub_date,
            keywords=keywords,
            mesh_terms=mesh_terms,
            doi=doi,
        )

    async def fetch_recent_papers(
        self,
        max_results: int = 100,
        days: int = 7,
    ) -> List[Paper]:
        """
        Convenience method to search and fetch papers in one call.

        Args:
            max_results: Maximum number of papers
            days: Number of days to look back

        Returns:
            List of Paper objects
        """
        pmids = await self.search(max_results=max_results, days=days)
        papers = await self.fetch_details(pmids)
        return papers

    async def fetch_papers_by_keywords(
        self,
        keyword_queries: Dict[str, str],
        max_per_keyword: int = 30,
        days: int = 2,
    ) -> Dict[str, List[Paper]]:
        """
        Fetch papers for each keyword separately for better coverage.

        Args:
            keyword_queries: Dict mapping topic name to search query
            max_per_keyword: Max papers per keyword
            days: Number of days to look back

        Returns:
            Dict mapping topic name to list of Papers
        """
        results = {}
        all_pmids = set()

        for topic_name, query in keyword_queries.items():
            pmids = await self.search(
                max_results=max_per_keyword,
                days=days,
                custom_query=query,
            )

            # Skip duplicates
            new_pmids = [p for p in pmids if p not in all_pmids]
            all_pmids.update(new_pmids)

            if new_pmids:
                papers = await self.fetch_details(new_pmids)
                results[topic_name] = papers
                print(f"  {topic_name}: {len(papers)}건")
            else:
                results[topic_name] = []

            # Rate limiting
            await asyncio.sleep(0.3)

        return results

    async def fetch_comprehensive(
        self,
        days: int = 2,
        max_total: int = 500,
    ) -> List[Paper]:
        """
        Fetch papers comprehensively covering all predefined hot topics.

        Strategy:
        1. Fetch papers for each hot topic keyword separately
        2. Merge and deduplicate
        3. Also fetch general biomedical papers

        Args:
            days: Number of days to look back
            max_total: Maximum total papers

        Returns:
            List of Paper objects with good coverage
        """
        from .config.hot_topics import PREDEFINED_HOT_TOPICS

        all_pmids = set()
        all_papers = []

        print(f"\n[키워드별 개별 검색 - {days}일]")

        # 1. Keyword-specific searches
        for topic_name, topic_info in PREDEFINED_HOT_TOPICS.items():
            keywords = topic_info["keywords"]
            # Build OR query from keywords
            query_parts = [f'"{kw}"[tiab]' for kw in keywords[:3]]  # Top 3 keywords
            query = " OR ".join(query_parts)

            pmids = await self.search(max_results=30, days=days, custom_query=query)
            new_pmids = [p for p in pmids if p not in all_pmids]

            if new_pmids:
                papers = await self.fetch_details(new_pmids[:20])  # Limit per topic
                all_papers.extend(papers)
                all_pmids.update(p.pmid for p in papers)
                print(f"  ✅ {topic_name}: {len(papers)}건")
            else:
                print(f"  ❌ {topic_name}: 0건")

            await asyncio.sleep(0.3)

            if len(all_papers) >= max_total:
                break

        # 2. Also fetch from default query for general coverage
        print(f"\n[일반 검색 추가]")
        general_pmids = await self.search(max_results=100, days=days)
        new_general = [p for p in general_pmids if p not in all_pmids]

        if new_general:
            general_papers = await self.fetch_details(new_general[:50])
            all_papers.extend(general_papers)
            print(f"  일반 검색: {len(general_papers)}건 추가")

        print(f"\n총 수집 논문: {len(all_papers)}건 (중복 제거됨)")
        return all_papers


async def main():
    """Test the PubMed fetcher."""
    fetcher = PubMedFetcher()

    print("Fetching recent papers from PubMed...")
    papers = await fetcher.fetch_recent_papers(max_results=20, days=3)

    print(f"\nFound {len(papers)} papers:")
    for i, paper in enumerate(papers[:5], 1):
        print(f"\n{i}. [{paper.pmid}] {paper.title[:80]}...")
        print(f"   Journal: {paper.journal}")
        print(f"   Keywords: {', '.join(paper.keywords[:3])}")
        print(f"   MeSH: {', '.join(paper.mesh_terms[:3])}")


if __name__ == "__main__":
    asyncio.run(main())
