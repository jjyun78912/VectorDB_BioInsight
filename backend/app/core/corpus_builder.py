"""
Core Corpus Builder for Disease-Specific Literature Collections.

Phase 1: MVP Implementation for ADHD
- Fetches high-quality papers from PubMed (Reviews, Guidelines, High-impact)
- Uses metadata only (title, abstract, MeSH terms)
- Implements quality scoring for paper selection

This module prioritizes DATA QUALITY over QUANTITY.
"""
import json
import time
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse

from .config import PAPERS_DIR
from .medical_vocabulary import get_medical_vocabulary, MeSHTerm


class ArticleType(Enum):
    """Article type classification based on PubMed publication types."""
    REVIEW = "review"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    GUIDELINE = "guideline"
    CLINICAL_TRIAL = "clinical_trial"
    RANDOMIZED_CONTROLLED_TRIAL = "rct"
    CASE_REPORT = "case_report"
    ORIGINAL_RESEARCH = "original_research"
    EDITORIAL = "editorial"
    LETTER = "letter"
    UNKNOWN = "unknown"


# Priority weights for article types (higher = more valuable for core corpus)
ARTICLE_TYPE_PRIORITY = {
    ArticleType.SYSTEMATIC_REVIEW: 1.0,
    ArticleType.META_ANALYSIS: 1.0,
    ArticleType.GUIDELINE: 0.95,
    ArticleType.REVIEW: 0.85,
    ArticleType.RANDOMIZED_CONTROLLED_TRIAL: 0.80,
    ArticleType.CLINICAL_TRIAL: 0.70,
    ArticleType.ORIGINAL_RESEARCH: 0.50,
    ArticleType.CASE_REPORT: 0.30,
    ArticleType.EDITORIAL: 0.20,
    ArticleType.LETTER: 0.10,
    ArticleType.UNKNOWN: 0.25,
}


@dataclass
class EnrichedPaper:
    """
    Enriched paper schema with new metadata fields.

    Phase 2 additions:
    - mesh_terms: Standardized MeSH descriptors
    - disease_tags: Normalized disease identifiers
    - gene_mentions: Extracted gene symbols
    - pathway_mentions: Biological pathways mentioned
    - article_type: Classified article type
    """
    # Core fields (existing)
    pmid: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: str = ""
    doi: str = ""
    keywords: list[str] = field(default_factory=list)

    # Phase 2: New enriched fields
    mesh_terms: list[str] = field(default_factory=list)
    mesh_qualifiers: list[str] = field(default_factory=list)
    disease_tags: list[str] = field(default_factory=list)
    gene_mentions: list[str] = field(default_factory=list)
    pathway_mentions: list[str] = field(default_factory=list)
    article_type: str = "unknown"
    publication_types: list[str] = field(default_factory=list)

    # Quality & collection metadata
    quality_score: float = 0.0
    citation_count: int = 0
    disease_domain: str = ""
    collections: list[str] = field(default_factory=list)  # Can belong to multiple

    # Timestamps
    collected_at: str = ""
    pubmed_date: str = ""

    # Optional full text (out of scope for Phase 1)
    pmcid: str = ""
    full_text: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'EnrichedPaper':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PubMedFetcher:
    """
    PubMed E-utilities fetcher with rate limiting.

    Focuses on fetching high-quality papers:
    - Reviews
    - Guidelines
    - Systematic reviews
    - Meta-analyses
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    RATE_LIMIT_DELAY = 0.4  # 3 requests/second max without API key

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: dict) -> str:
        """Make rate-limited request to PubMed."""
        self._rate_limit()

        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"

        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode('utf-8')

    def search(
        self,
        query: str,
        max_results: int = 100,
        article_types: list[str] | None = None,
        min_year: int | None = None
    ) -> list[str]:
        """
        Search PubMed and return PMIDs.

        Args:
            query: Search query
            max_results: Maximum number of results
            article_types: Filter by publication types
            min_year: Minimum publication year

        Returns:
            List of PMIDs
        """
        # Build query with filters
        search_parts = [query]

        if article_types:
            type_filter = " OR ".join([f'"{t}"[pt]' for t in article_types])
            search_parts.append(f"({type_filter})")

        if min_year:
            search_parts.append(f"{min_year}:3000[dp]")

        full_query = " AND ".join(search_parts)

        params = {
            "db": "pubmed",
            "term": full_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }

        result = self._make_request("esearch.fcgi", params)
        data = json.loads(result)

        return data.get("esearchresult", {}).get("idlist", [])

    def fetch_details(self, pmids: list[str]) -> list[dict]:
        """
        Fetch detailed metadata for PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of paper metadata dictionaries
        """
        if not pmids:
            return []

        # Fetch in batches of 100
        all_papers = []

        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]

            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "xml",
                "retmode": "xml"
            }

            xml_result = self._make_request("efetch.fcgi", params)
            papers = self._parse_pubmed_xml(xml_result)
            all_papers.extend(papers)

        return all_papers

    def _parse_pubmed_xml(self, xml_str: str) -> list[dict]:
        """Parse PubMed XML response into paper dictionaries."""
        papers = []

        try:
            root = ET.fromstring(xml_str)

            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
        except ET.ParseError as e:
            print(f"XML parse error: {e}")

        return papers

    def _parse_article(self, article: ET.Element) -> dict | None:
        """Parse single PubMed article XML."""
        try:
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None

            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            article_elem = medline.find(".//Article")
            if article_elem is None:
                return None

            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_parts = []
            for abs_text in article_elem.findall(".//AbstractText"):
                label = abs_text.get("Label", "")
                text = abs_text.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last = author.find("LastName")
                first = author.find("ForeName")
                if last is not None:
                    name = last.text
                    if first is not None:
                        name = f"{first.text} {name}"
                    authors.append(name)

            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Year
            year_elem = article_elem.find(".//PubDate/Year")
            if year_elem is None:
                year_elem = article_elem.find(".//PubDate/MedlineDate")
            year = ""
            if year_elem is not None and year_elem.text:
                year_match = re.search(r'\d{4}', year_elem.text)
                year = year_match.group() if year_match else ""

            # DOI
            doi = ""
            for id_elem in article_elem.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text or ""
                    break

            # Also check ELocationID
            if not doi:
                for eloc in article_elem.findall(".//ELocationID"):
                    if eloc.get("EIdType") == "doi":
                        doi = eloc.text or ""
                        break

            # Keywords
            keywords = []
            for kw in medline.findall(".//KeywordList/Keyword"):
                if kw.text:
                    keywords.append(kw.text)

            # MeSH terms
            mesh_terms = []
            mesh_qualifiers = []
            for mesh in medline.findall(".//MeshHeadingList/MeshHeading"):
                desc = mesh.find("DescriptorName")
                if desc is not None and desc.text:
                    mesh_terms.append(desc.text)
                for qual in mesh.findall("QualifierName"):
                    if qual.text:
                        mesh_qualifiers.append(qual.text)

            # Publication types
            pub_types = []
            for pt in article_elem.findall(".//PublicationTypeList/PublicationType"):
                if pt.text:
                    pub_types.append(pt.text)

            # PMCID
            pmcid = ""
            for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                if article_id.get("IdType") == "pmc":
                    pmcid = article_id.text or ""
                    break

            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal,
                "year": year,
                "doi": doi,
                "keywords": keywords,
                "mesh_terms": mesh_terms,
                "mesh_qualifiers": mesh_qualifiers,
                "publication_types": pub_types,
                "pmcid": pmcid,
            }

        except Exception as e:
            print(f"Error parsing article: {e}")
            return None


class ArticleTypeClassifier:
    """Classify article type from publication types."""

    # PubMed publication type mappings
    TYPE_MAPPINGS = {
        "Systematic Review": ArticleType.SYSTEMATIC_REVIEW,
        "Meta-Analysis": ArticleType.META_ANALYSIS,
        "Practice Guideline": ArticleType.GUIDELINE,
        "Guideline": ArticleType.GUIDELINE,
        "Consensus Development Conference": ArticleType.GUIDELINE,
        "Review": ArticleType.REVIEW,
        "Randomized Controlled Trial": ArticleType.RANDOMIZED_CONTROLLED_TRIAL,
        "Clinical Trial": ArticleType.CLINICAL_TRIAL,
        "Clinical Trial, Phase I": ArticleType.CLINICAL_TRIAL,
        "Clinical Trial, Phase II": ArticleType.CLINICAL_TRIAL,
        "Clinical Trial, Phase III": ArticleType.CLINICAL_TRIAL,
        "Clinical Trial, Phase IV": ArticleType.CLINICAL_TRIAL,
        "Case Reports": ArticleType.CASE_REPORT,
        "Editorial": ArticleType.EDITORIAL,
        "Letter": ArticleType.LETTER,
        "Comment": ArticleType.LETTER,
    }

    @classmethod
    def classify(cls, publication_types: list[str]) -> ArticleType:
        """
        Classify article type from publication types list.

        Returns highest priority type found.
        """
        best_type = ArticleType.UNKNOWN
        best_priority = -1

        for pt in publication_types:
            article_type = cls.TYPE_MAPPINGS.get(pt, ArticleType.UNKNOWN)
            priority = ARTICLE_TYPE_PRIORITY.get(article_type, 0)

            if priority > best_priority:
                best_priority = priority
                best_type = article_type

        # Default to original research if has abstract but no specific type
        if best_type == ArticleType.UNKNOWN and "Journal Article" in publication_types:
            best_type = ArticleType.ORIGINAL_RESEARCH

        return best_type


class QualityScorer:
    """
    Calculate quality score for papers.

    Factors:
    - Article type (reviews > original research)
    - Has MeSH terms
    - Abstract length
    - Recency
    - Journal quality (future: impact factor)
    """

    @classmethod
    def score(cls, paper: dict, disease_key: str | None = None) -> float:
        """
        Calculate quality score (0-100).

        Args:
            paper: Paper metadata dictionary
            disease_key: Disease to check relevance for

        Returns:
            Quality score 0-100
        """
        score = 0.0

        # 1. Article type (40 points max)
        article_type = ArticleTypeClassifier.classify(paper.get("publication_types", []))
        type_priority = ARTICLE_TYPE_PRIORITY.get(article_type, 0.25)
        score += type_priority * 40

        # 2. MeSH terms presence (15 points)
        mesh_terms = paper.get("mesh_terms", [])
        if mesh_terms:
            mesh_score = min(len(mesh_terms) / 10, 1.0) * 15
            score += mesh_score

        # 3. Abstract quality (20 points)
        abstract = paper.get("abstract", "")
        abstract_len = len(abstract)
        if abstract_len > 1500:
            score += 20
        elif abstract_len > 800:
            score += 15
        elif abstract_len > 300:
            score += 10
        elif abstract_len > 100:
            score += 5

        # 4. Recency (15 points)
        year = paper.get("year", "")
        if year:
            try:
                year_int = int(year)
                current_year = datetime.now().year
                age = current_year - year_int
                if age <= 2:
                    score += 15
                elif age <= 5:
                    score += 12
                elif age <= 10:
                    score += 8
                else:
                    score += 3
            except ValueError:
                pass

        # 5. Disease relevance (10 points)
        if disease_key:
            vocab = get_medical_vocabulary()
            title = paper.get("title", "")
            match_result = vocab.match_score(f"{title} {abstract}", disease_key)
            if match_result["has_primary"]:
                score += 10
            elif match_result["score"] > 0.5:
                score += 7
            elif match_result["score"] > 0.2:
                score += 4

        return min(100, score)


class GeneExtractor:
    """Extract gene symbols from text using pattern matching."""

    # Common gene symbol patterns
    GENE_PATTERN = re.compile(
        r'\b([A-Z][A-Z0-9]{1,5}(?:-[A-Z0-9]+)?)\b'
    )

    # Known genes (subset for common biomarker genes)
    KNOWN_GENES = {
        # Neurodevelopmental
        "DRD4", "DRD5", "DAT1", "SLC6A3", "SLC6A4", "COMT", "SNAP25",
        "BDNF", "HTR1B", "HTR2A", "CHRNA4", "ADRA2A", "DBH", "TPH2",
        "MAOA", "FOXP2", "CNTNAP2", "NRXN1", "SHANK3",
        # Oncogenes
        "KRAS", "BRAF", "TP53", "EGFR", "HER2", "BRCA1", "BRCA2",
        "PIK3CA", "PTEN", "AKT1", "MYC", "RB1", "CDKN2A", "SMAD4",
        # Metabolic
        "MTHFR", "CYP2D6", "CYP2C19", "CYP3A4", "ABCB1",
    }

    @classmethod
    def extract(cls, text: str) -> list[str]:
        """Extract gene symbols from text."""
        if not text:
            return []

        found = set()

        # Pattern matching
        for match in cls.GENE_PATTERN.finditer(text):
            candidate = match.group(1)
            # Filter: must be in known genes or look like a gene
            if candidate in cls.KNOWN_GENES:
                found.add(candidate)
            elif len(candidate) >= 3 and any(c.isdigit() for c in candidate):
                # Likely gene symbol (has digits)
                found.add(candidate)

        return sorted(found)


class PathwayExtractor:
    """Extract biological pathway mentions from text."""

    PATHWAY_KEYWORDS = [
        # Signaling pathways
        "dopamine pathway", "dopaminergic",
        "norepinephrine pathway", "noradrenergic",
        "serotonin pathway", "serotonergic",
        "glutamate pathway", "glutamatergic",
        "GABA pathway", "GABAergic",
        "PI3K/AKT", "PI3K-AKT",
        "MAPK pathway", "ERK pathway",
        "Wnt pathway", "Wnt signaling",
        "Notch pathway", "Notch signaling",
        "JAK-STAT", "JAK/STAT",
        "NF-kB", "NF-kappaB",
        "TGF-beta", "TGF-B",
        # Metabolic
        "glycolysis", "gluconeogenesis",
        "oxidative phosphorylation",
        "fatty acid metabolism",
        # Neuroscience specific
        "reward pathway", "mesolimbic",
        "prefrontal cortex", "striatum",
        "default mode network", "DMN",
        "executive function",
    ]

    @classmethod
    def extract(cls, text: str) -> list[str]:
        """Extract pathway mentions from text."""
        if not text:
            return []

        text_lower = text.lower()
        found = []

        for pathway in cls.PATHWAY_KEYWORDS:
            if pathway.lower() in text_lower:
                found.append(pathway)

        return found


class CoreCorpusBuilder:
    """
    Build disease-specific core corpus collections.

    Strategy:
    1. Fetch high-quality papers (reviews, guidelines, meta-analyses)
    2. Score and rank by quality
    3. Extract entities (MeSH, genes, pathways)
    4. Organize into disease collections
    """

    # Search strategies for different paper types
    SEARCH_STRATEGIES = {
        "reviews": {
            "query_suffix": "",
            "article_types": ["Review", "Systematic Review"],
            "max_results": 50,
        },
        "guidelines": {
            "query_suffix": "guideline OR consensus",
            "article_types": ["Practice Guideline", "Guideline", "Consensus Development Conference"],
            "max_results": 30,
        },
        "meta_analyses": {
            "query_suffix": "meta-analysis",
            "article_types": ["Meta-Analysis"],
            "max_results": 30,
        },
        "clinical_trials": {
            "query_suffix": "treatment OR therapy",
            "article_types": ["Randomized Controlled Trial", "Clinical Trial"],
            "max_results": 30,
        },
    }

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or PAPERS_DIR
        self.fetcher = PubMedFetcher()
        self.vocab = get_medical_vocabulary()

    def build_collection(
        self,
        disease_key: str,
        min_quality_score: float = 40.0,
        max_papers: int = 100
    ) -> list[EnrichedPaper]:
        """
        Build core corpus for a disease.

        Args:
            disease_key: Disease identifier (e.g., 'adhd')
            min_quality_score: Minimum quality score to include
            max_papers: Maximum papers to include

        Returns:
            List of enriched papers
        """
        mesh_term = self.vocab.get_mesh_term(disease_key)
        if not mesh_term:
            print(f"Warning: No MeSH term found for '{disease_key}'")
            base_query = disease_key
        else:
            # Use MeSH term for precise search
            base_query = f'"{mesh_term.primary}"[MeSH] OR "{mesh_term.primary}"[tiab]'
            for syn in mesh_term.synonyms[:3]:
                base_query += f' OR "{syn}"[tiab]'

        print(f"\n{'='*60}")
        print(f"Building core corpus for: {disease_key}")
        print(f"Base query: {base_query[:80]}...")
        print(f"{'='*60}\n")

        all_papers = []
        seen_pmids = set()

        # Fetch papers using different strategies
        for strategy_name, strategy in self.SEARCH_STRATEGIES.items():
            print(f"\n[{strategy_name.upper()}] Fetching...")

            query = base_query
            if strategy["query_suffix"]:
                query = f"({query}) AND ({strategy['query_suffix']})"

            pmids = self.fetcher.search(
                query=query,
                max_results=strategy["max_results"],
                article_types=strategy.get("article_types"),
                min_year=2015  # Focus on recent papers
            )

            # Remove duplicates
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            print(f"  Found {len(pmids)} papers, {len(new_pmids)} new")

            if new_pmids:
                papers = self.fetcher.fetch_details(new_pmids)
                print(f"  Fetched details for {len(papers)} papers")
                all_papers.extend(papers)

        print(f"\nTotal raw papers: {len(all_papers)}")

        # Process and score papers
        enriched_papers = []

        for paper in all_papers:
            enriched = self._enrich_paper(paper, disease_key)

            if enriched.quality_score >= min_quality_score:
                enriched_papers.append(enriched)

        print(f"Papers passing quality threshold ({min_quality_score}): {len(enriched_papers)}")

        # Sort by quality and take top N
        enriched_papers.sort(key=lambda p: p.quality_score, reverse=True)
        enriched_papers = enriched_papers[:max_papers]

        print(f"Final collection size: {len(enriched_papers)}")

        # Save collection
        self._save_collection(disease_key, enriched_papers)

        return enriched_papers

    def _enrich_paper(self, paper: dict, disease_key: str) -> EnrichedPaper:
        """Enrich paper with additional metadata and scoring."""

        # Classify article type
        article_type = ArticleTypeClassifier.classify(paper.get("publication_types", []))

        # Calculate quality score
        quality_score = QualityScorer.score(paper, disease_key)

        # Extract entities
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        genes = GeneExtractor.extract(text)
        pathways = PathwayExtractor.extract(text)

        return EnrichedPaper(
            pmid=paper.get("pmid", ""),
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            authors=paper.get("authors", []),
            journal=paper.get("journal", ""),
            year=paper.get("year", ""),
            doi=paper.get("doi", ""),
            keywords=paper.get("keywords", []),
            mesh_terms=paper.get("mesh_terms", []),
            mesh_qualifiers=paper.get("mesh_qualifiers", []),
            disease_tags=[disease_key],
            gene_mentions=genes,
            pathway_mentions=pathways,
            article_type=article_type.value,
            publication_types=paper.get("publication_types", []),
            quality_score=quality_score,
            disease_domain=disease_key,
            collections=[disease_key],
            collected_at=datetime.now().isoformat(),
            pmcid=paper.get("pmcid", ""),
        )

    def _save_collection(self, disease_key: str, papers: list[EnrichedPaper]):
        """Save collection to disk."""

        collection_dir = self.output_dir / disease_key
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Save individual papers
        for paper in papers:
            paper_file = collection_dir / f"{paper.pmid}.json"
            with open(paper_file, 'w', encoding='utf-8') as f:
                json.dump(paper.to_dict(), f, indent=2, ensure_ascii=False)

        # Save collection index
        index = {
            "disease_key": disease_key,
            "total_papers": len(papers),
            "created_at": datetime.now().isoformat(),
            "paper_types": self._summarize_types(papers),
            "quality_stats": self._quality_stats(papers),
            "papers": [
                {
                    "pmid": p.pmid,
                    "title": p.title,
                    "article_type": p.article_type,
                    "quality_score": p.quality_score,
                    "year": p.year,
                }
                for p in papers
            ]
        }

        index_file = collection_dir / "_collection_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        print(f"\nCollection saved to: {collection_dir}")
        print(f"  - {len(papers)} paper files")
        print(f"  - 1 collection index")

    def _summarize_types(self, papers: list[EnrichedPaper]) -> dict:
        """Summarize article types in collection."""
        types = {}
        for p in papers:
            types[p.article_type] = types.get(p.article_type, 0) + 1
        return types

    def _quality_stats(self, papers: list[EnrichedPaper]) -> dict:
        """Calculate quality statistics."""
        if not papers:
            return {}

        scores = [p.quality_score for p in papers]
        return {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
        }


def build_adhd_corpus():
    """
    Build ADHD core corpus - Phase 1 MVP.

    This is the entry point for testing the pipeline.
    """
    builder = CoreCorpusBuilder()

    papers = builder.build_collection(
        disease_key="adhd",
        min_quality_score=35.0,  # Lower threshold for MVP
        max_papers=50  # Small corpus for validation
    )

    # Print summary
    print("\n" + "="*60)
    print("ADHD CORE CORPUS SUMMARY")
    print("="*60)

    print(f"\nTotal papers: {len(papers)}")

    # By type
    types = {}
    for p in papers:
        types[p.article_type] = types.get(p.article_type, 0) + 1

    print("\nBy Article Type:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # Top papers
    print("\nTop 5 Papers by Quality:")
    for i, p in enumerate(papers[:5], 1):
        print(f"  {i}. [{p.article_type}] {p.title[:60]}...")
        print(f"     Quality: {p.quality_score:.1f}, Year: {p.year}")

    return papers


if __name__ == "__main__":
    build_adhd_corpus()
