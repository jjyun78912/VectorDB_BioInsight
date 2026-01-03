"""
PubMed Paper Collector for BioInsight VectorDB.

Collects research papers from PubMed/PMC for specific diseases and indexes them into the vector database.

Usage:
    python scripts/pubmed_collector.py --disease "pancreatic cancer" --count 30
    python scripts/pubmed_collector.py --all  # Collect all 5 diseases
"""

import os
import sys
import time
import json
import re
import argparse
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.config import PAPERS_DIR, CHROMA_DIR
from backend.app.core.vector_store import BioVectorStore
from backend.app.core.embeddings import get_embedder
from backend.app.core.text_splitter import BioPaperSplitter, TextChunk

# PubMed E-utilities base URLs
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Disease configurations
DISEASE_CONFIGS = {
    "pancreatic_cancer": {
        "name": "Pancreatic Cancer",
        "query": "(pancreatic cancer[Title/Abstract] OR pancreatic adenocarcinoma[Title/Abstract] OR PDAC[Title/Abstract]) AND (treatment OR therapy OR prognosis OR biomarker OR gene expression OR KRAS OR transcriptomics)",
        "kr_name": "췌장암"
    },
    "blood_cancer": {
        "name": "Blood Cancer",
        "query": "(leukemia[Title/Abstract] OR lymphoma[Title/Abstract] OR myeloma[Title/Abstract] OR hematological malignancy[Title/Abstract]) AND (treatment OR therapy OR gene OR molecular OR RNA-seq)",
        "kr_name": "혈액암"
    },
    "glioblastoma": {
        "name": "Glioblastoma",
        "query": "(glioblastoma[Title/Abstract] OR GBM[Title/Abstract] OR brain tumor[Title/Abstract]) AND (treatment OR therapy OR prognosis OR molecular OR MGMT OR IDH OR transcriptome)",
        "kr_name": "교모세포종"
    },
    "alzheimer": {
        "name": "Alzheimer's Disease",
        "query": "(Alzheimer[Title/Abstract] OR Alzheimer's disease[Title/Abstract] OR AD[Title/Abstract] AND dementia) AND (treatment OR therapy OR biomarker OR amyloid OR tau)",
        "kr_name": "알츠하이머"
    },
    "pcos": {
        "name": "Polycystic Ovary Syndrome",
        "query": "(polycystic ovary syndrome[Title/Abstract] OR PCOS[Title/Abstract]) AND (treatment OR therapy OR metabolism OR hormones OR insulin)",
        "kr_name": "다낭성난소증후군"
    },
    "pheochromocytoma": {
        "name": "Pheochromocytoma",
        "query": "(pheochromocytoma[Title/Abstract] OR paraganglioma[Title/Abstract] OR PPGL[Title/Abstract]) AND (genetics OR SDH OR catecholamine OR treatment OR diagnosis)",
        "kr_name": "갈색세포종"
    },
    "lung_cancer": {
        "name": "Lung Cancer",
        "query": "(lung cancer[Title/Abstract] OR NSCLC[Title/Abstract] OR lung adenocarcinoma[Title/Abstract]) AND (EGFR OR ALK OR KRAS OR immunotherapy OR targeted therapy OR gene expression OR RNA-seq)",
        "kr_name": "폐암"
    },
    "breast_cancer": {
        "name": "Breast Cancer",
        "query": "(breast cancer[Title/Abstract] OR breast carcinoma[Title/Abstract]) AND (BRCA1 OR BRCA2 OR HER2 OR ER OR PR OR triple negative OR gene expression OR transcriptomics)",
        "kr_name": "유방암"
    },
    "colorectal_cancer": {
        "name": "Colorectal Cancer",
        "query": "(colorectal cancer[Title/Abstract] OR colon cancer[Title/Abstract] OR CRC[Title/Abstract]) AND (APC OR KRAS OR BRAF OR microsatellite OR gene expression OR RNA-seq)",
        "kr_name": "대장암"
    },
    "liver_cancer": {
        "name": "Liver Cancer",
        "query": "(hepatocellular carcinoma[Title/Abstract] OR HCC[Title/Abstract] OR liver cancer[Title/Abstract]) AND (HBV OR HCV OR TP53 OR CTNNB1 OR gene expression OR transcriptome)",
        "kr_name": "간암"
    },
    "rnaseq_transcriptomics": {
        "name": "RNA-seq Transcriptomics",
        "query": "(RNA-seq[Title/Abstract] OR RNA sequencing[Title/Abstract] OR transcriptomics[Title/Abstract]) AND (cancer OR tumor) AND (differential expression OR gene regulatory network OR hub gene OR driver gene)",
        "kr_name": "RNA-seq 전사체학"
    }
}


@dataclass
class PaperInfo:
    """Information about a collected paper."""
    pmid: str
    pmcid: str = ""
    title: str = ""
    abstract: str = ""
    authors: list = None
    journal: str = ""
    year: str = ""
    doi: str = ""
    keywords: list = None
    full_text: str = ""
    disease_domain: str = ""
    collected_at: str = ""

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.keywords is None:
            self.keywords = []
        if not self.collected_at:
            self.collected_at = datetime.now().isoformat()


class PubMedCollector:
    """Collects papers from PubMed/PMC and indexes them into vector database."""

    def __init__(self, disease_key: str, api_key: str = None):
        """
        Initialize the collector.

        Args:
            disease_key: Key from DISEASE_CONFIGS
            api_key: Optional NCBI API key (increases rate limit)
        """
        if disease_key not in DISEASE_CONFIGS:
            raise ValueError(f"Unknown disease: {disease_key}. Valid options: {list(DISEASE_CONFIGS.keys())}")

        self.disease_key = disease_key
        self.disease_config = DISEASE_CONFIGS[disease_key]
        self.api_key = api_key or os.getenv("NCBI_API_KEY", "")

        # Set up directories
        self.disease_dir = PAPERS_DIR / disease_key
        self.disease_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting (3 requests/sec without API key, 10 with)
        self.request_delay = 0.34 if self.api_key else 1.0

        # Initialize vector store for this disease
        self.vector_store = BioVectorStore(
            disease_domain=disease_key,
            persist_directory=CHROMA_DIR
        )

        # Text splitter
        self.text_splitter = BioPaperSplitter()

        print(f"Initialized collector for: {self.disease_config['name']} ({self.disease_config['kr_name']})")
        print(f"Papers directory: {self.disease_dir}")
        print(f"Vector store collection: {self.vector_store.collection_name}")

    def _make_request(self, url: str, params: dict) -> requests.Response:
        """Make a request with rate limiting."""
        if self.api_key:
            params["api_key"] = self.api_key

        time.sleep(self.request_delay)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response

    def search_pubmed(self, max_results: int = 50) -> list[str]:
        """
        Search PubMed for papers matching the disease query.

        Returns:
            List of PMIDs
        """
        print(f"\nSearching PubMed for: {self.disease_config['name']}")

        params = {
            "db": "pubmed",
            "term": self.disease_config["query"],
            "retmax": max_results,
            "sort": "relevance",
            "retmode": "json",
            "mindate": "2020",
            "maxdate": "2025",
            "datetype": "pdat"
        }

        response = self._make_request(PUBMED_SEARCH_URL, params)
        data = response.json()

        pmids = data.get("esearchresult", {}).get("idlist", [])
        total_count = data.get("esearchresult", {}).get("count", 0)

        print(f"Found {total_count} total papers, retrieved {len(pmids)} PMIDs")
        return pmids

    def fetch_paper_details(self, pmids: list[str]) -> list[PaperInfo]:
        """
        Fetch detailed information for papers.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PaperInfo objects
        """
        print(f"\nFetching details for {len(pmids)} papers...")

        papers = []
        batch_size = 20

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]

            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract"
            }

            response = self._make_request(PUBMED_FETCH_URL, params)
            papers.extend(self._parse_pubmed_xml(response.text))

            print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} papers")

        return papers

    def _parse_pubmed_xml(self, xml_text: str) -> list[PaperInfo]:
        """Parse PubMed XML response into PaperInfo objects."""
        papers = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_article(article)
                    if paper:
                        paper.disease_domain = self.disease_key
                        papers.append(paper)
                except Exception as e:
                    print(f"  Warning: Error parsing article: {e}")
                    continue
        except ET.ParseError as e:
            print(f"  Warning: XML parse error: {e}")

        return papers

    def _parse_article(self, article) -> Optional[PaperInfo]:
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
        title = title_elem.text if title_elem is not None else ""

        # Abstract
        abstract_parts = []
        for abstract_text in article_elem.findall(".//AbstractText"):
            label = abstract_text.get("Label", "")
            text = abstract_text.text or ""
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
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
        year_elem = article_elem.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = article_elem.find(".//PubDate/MedlineDate")
        year = year_elem.text[:4] if year_elem is not None and year_elem.text else ""

        # DOI
        doi = ""
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "doi":
                doi = id_elem.text
                break

        # PMC ID
        pmcid = ""
        for id_elem in article.findall(".//ArticleId"):
            if id_elem.get("IdType") == "pmc":
                pmcid = id_elem.text
                break

        # Keywords
        keywords = []
        for kw in medline.findall(".//Keyword"):
            if kw.text:
                keywords.append(kw.text)

        return PaperInfo(
            pmid=pmid,
            pmcid=pmcid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            year=year,
            doi=doi,
            keywords=keywords
        )

    def fetch_pmc_fulltext(self, pmcid: str) -> Optional[str]:
        """
        Fetch full text from PMC if available.

        Args:
            pmcid: PMC ID (e.g., "PMC1234567")

        Returns:
            Full text content or None
        """
        if not pmcid:
            return None

        # Normalize PMCID
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        try:
            # Use PMC OA service to get full text
            url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{pmcid}/unicode"

            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                # Parse BioC XML to extract text
                return self._extract_text_from_bioc(response.text)

            # Fallback: try to get from PMC directly
            time.sleep(self.request_delay)
            pmc_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                "db": "pmc",
                "id": pmcid.replace("PMC", ""),
                "rettype": "xml"
            }

            if self.api_key:
                params["api_key"] = self.api_key

            response = requests.get(pmc_url, params=params, timeout=30)
            if response.status_code == 200:
                return self._extract_text_from_pmc_xml(response.text)

        except Exception as e:
            print(f"  Warning: Could not fetch full text for {pmcid}: {e}")

        return None

    def _extract_text_from_bioc(self, xml_text: str) -> Optional[str]:
        """Extract text from BioC XML format."""
        try:
            root = ET.fromstring(xml_text)

            texts = []
            for passage in root.findall(".//passage"):
                text_elem = passage.find("text")
                if text_elem is not None and text_elem.text:
                    texts.append(text_elem.text)

            return "\n\n".join(texts) if texts else None
        except:
            return None

    def _extract_text_from_pmc_xml(self, xml_text: str) -> Optional[str]:
        """Extract text from PMC XML format."""
        try:
            root = ET.fromstring(xml_text)

            texts = []

            # Get body paragraphs
            for p in root.findall(".//body//p"):
                text = "".join(p.itertext())
                if text.strip():
                    texts.append(text.strip())

            return "\n\n".join(texts) if texts else None
        except:
            return None

    def collect_papers(self, target_count: int = 30, with_fulltext: bool = True) -> list[PaperInfo]:
        """
        Collect papers for the disease.

        Args:
            target_count: Target number of papers to collect
            with_fulltext: Whether to fetch full text from PMC

        Returns:
            List of collected PaperInfo objects
        """
        print(f"\n{'='*60}")
        print(f"Collecting papers for: {self.disease_config['name']}")
        print(f"Target count: {target_count}")
        print(f"{'='*60}")

        # Search for more papers than needed (some might not have abstracts)
        pmids = self.search_pubmed(max_results=target_count * 2)

        # Fetch paper details
        papers = self.fetch_paper_details(pmids)

        # Filter papers with abstracts
        papers_with_content = [p for p in papers if p.abstract and len(p.abstract) > 100]
        print(f"\nPapers with valid abstracts: {len(papers_with_content)}")

        # Fetch full text for papers with PMC IDs
        if with_fulltext:
            print("\nFetching full text from PMC...")
            fulltext_count = 0
            for paper in papers_with_content[:target_count]:
                if paper.pmcid:
                    fulltext = self.fetch_pmc_fulltext(paper.pmcid)
                    if fulltext:
                        paper.full_text = fulltext
                        fulltext_count += 1
                        print(f"  Got full text for: {paper.pmid} ({paper.pmcid})")

            print(f"  Full text fetched: {fulltext_count}/{len(papers_with_content[:target_count])}")

        # Take only target count
        collected = papers_with_content[:target_count]

        # Save papers to disk
        self._save_papers(collected)

        return collected

    def _save_papers(self, papers: list[PaperInfo]):
        """Save papers to JSON files."""
        print(f"\nSaving {len(papers)} papers to {self.disease_dir}...")

        for paper in papers:
            filename = f"{paper.pmid}.json"
            filepath = self.disease_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(asdict(paper), f, indent=2, ensure_ascii=False)

        # Save index
        index = {
            "disease": self.disease_config["name"],
            "disease_key": self.disease_key,
            "collected_at": datetime.now().isoformat(),
            "paper_count": len(papers),
            "papers": [
                {
                    "pmid": p.pmid,
                    "title": p.title,
                    "year": p.year,
                    "has_fulltext": bool(p.full_text)
                }
                for p in papers
            ]
        }

        with open(self.disease_dir / "_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        print(f"  Saved to: {self.disease_dir}")

    def index_to_vectordb(self, papers: list[PaperInfo] = None):
        """
        Index papers into the vector database.

        Args:
            papers: Papers to index (loads from disk if None)
        """
        if papers is None:
            papers = self._load_papers()

        if not papers:
            print("No papers to index!")
            return

        print(f"\n{'='*60}")
        print(f"Indexing {len(papers)} papers to vector DB")
        print(f"Collection: {self.vector_store.collection_name}")
        print(f"{'='*60}")

        all_chunks = []

        for paper in papers:
            chunks = self._paper_to_chunks(paper)
            all_chunks.extend(chunks)
            print(f"  {paper.pmid}: {len(chunks)} chunks")

        print(f"\nTotal chunks to index: {len(all_chunks)}")

        # Add to vector store
        if all_chunks:
            added = self.vector_store.add_chunks(all_chunks, show_progress=True)
            print(f"Added {added} chunks to vector store")

        # Print stats
        stats = self.vector_store.get_collection_stats()
        print(f"\nCollection stats:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Total papers: {stats['total_papers']}")

    def _paper_to_chunks(self, paper: PaperInfo) -> list[TextChunk]:
        """Convert a paper to text chunks for indexing."""
        chunks = []

        metadata_base = {
            "pmid": paper.pmid,
            "pmcid": paper.pmcid,
            "paper_title": paper.title,
            "authors": ", ".join(paper.authors[:3]) + ("..." if len(paper.authors) > 3 else ""),
            "journal": paper.journal,
            "year": paper.year,
            "doi": paper.doi,
            "keywords": ", ".join(paper.keywords[:5]),
            "disease_domain": paper.disease_domain,
            "source_file": f"{paper.pmid}.json"
        }

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

    def _load_papers(self) -> list[PaperInfo]:
        """Load papers from disk."""
        papers = []

        for json_file in self.disease_dir.glob("*.json"):
            if json_file.name.startswith("_"):
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    papers.append(PaperInfo(**data))
            except Exception as e:
                print(f"  Warning: Could not load {json_file}: {e}")

        return papers


def collect_all_diseases(target_per_disease: int = 30):
    """Collect papers for all configured diseases."""
    print("\n" + "="*70)
    print("BIOINSIGHT VECTORDB - DISEASE PAPER COLLECTION")
    print("="*70)
    print(f"Target papers per disease: {target_per_disease}")
    print(f"Diseases: {list(DISEASE_CONFIGS.keys())}")
    print("="*70 + "\n")

    results = {}

    for disease_key in DISEASE_CONFIGS.keys():
        try:
            collector = PubMedCollector(disease_key)
            papers = collector.collect_papers(target_count=target_per_disease)
            collector.index_to_vectordb(papers)

            results[disease_key] = {
                "success": True,
                "count": len(papers),
                "indexed": collector.vector_store.count
            }
        except Exception as e:
            print(f"ERROR collecting {disease_key}: {e}")
            results[disease_key] = {
                "success": False,
                "error": str(e)
            }

    # Print summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)

    for disease_key, result in results.items():
        name = DISEASE_CONFIGS[disease_key]["kr_name"]
        if result["success"]:
            print(f"✓ {name} ({disease_key}): {result['count']} papers, {result['indexed']} chunks indexed")
        else:
            print(f"✗ {name} ({disease_key}): FAILED - {result['error']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect PubMed papers for BioInsight VectorDB")
    parser.add_argument("--disease", type=str, help="Disease key to collect")
    parser.add_argument("--all", action="store_true", help="Collect all diseases")
    parser.add_argument("--count", type=int, default=30, help="Number of papers per disease")
    parser.add_argument("--index-only", action="store_true", help="Only index existing papers")
    parser.add_argument("--list", action="store_true", help="List available diseases")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable diseases:")
        for key, config in DISEASE_CONFIGS.items():
            print(f"  {key}: {config['name']} ({config['kr_name']})")
        return

    if args.all:
        collect_all_diseases(target_per_disease=args.count)
    elif args.disease:
        collector = PubMedCollector(args.disease)

        if args.index_only:
            collector.index_to_vectordb()
        else:
            papers = collector.collect_papers(target_count=args.count)
            collector.index_to_vectordb(papers)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python scripts/pubmed_collector.py --all --count 30")
        print("  python scripts/pubmed_collector.py --disease pancreatic_cancer --count 30")


if __name__ == "__main__":
    main()
