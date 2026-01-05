"""
Trend Analyzer - Auto Keyword Extraction
Automatically extracts hot keywords from papers and tracks trends over time.
"""

import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter

from .pubmed_fetcher import Paper


@dataclass
class Trend:
    """Represents a trending keyword."""
    keyword: str
    count: int
    previous_count: int = 0
    week_ago_count: int = 0
    representative_papers: List[Paper] = field(default_factory=list)

    @property
    def day_change(self) -> float:
        """Calculate day-over-day change percentage."""
        if self.previous_count == 0:
            return 100.0 if self.count > 0 else 0.0
        return ((self.count - self.previous_count) / self.previous_count) * 100

    @property
    def week_change(self) -> float:
        """Calculate week-over-week change percentage."""
        if self.week_ago_count == 0:
            return 100.0 if self.count > 0 else 0.0
        return ((self.count - self.week_ago_count) / self.week_ago_count) * 100

    @property
    def trend_indicator(self) -> str:
        """Get trend indicator emoji."""
        if self.day_change >= 50:
            return "🔥"  # Hot
        elif self.day_change >= 10:
            return "⬆️"  # Rising
        elif self.day_change <= -10:
            return "⬇️"  # Declining
        else:
            return "➡️"  # Stable

    def to_dict(self) -> Dict:
        return {
            "keyword": self.keyword,
            "count": self.count,
            "previous_count": self.previous_count,
            "week_ago_count": self.week_ago_count,
            "day_change": round(self.day_change, 1),
            "week_change": round(self.week_change, 1),
            "trend_indicator": self.trend_indicator,
            "paper_count": len(self.representative_papers),
        }


class TrendAnalyzer:
    """Analyzes paper trends and extracts hot keywords."""

    # Stopwords to exclude from keyword extraction
    STOPWORDS = {
        # Common English words
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "this", "that", "these", "those", "it", "its", "we", "our", "their",
        "they", "them", "he", "she", "his", "her", "him", "i", "you", "your",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "not", "only", "same", "so", "than", "too", "very",
        "just", "also", "now", "here", "there", "when", "where", "why", "how",
        "what", "which", "who", "whom", "whose",
        # Scientific common words (too generic)
        "study", "studies", "research", "analysis", "results", "findings",
        "data", "methods", "patients", "cells", "using", "based", "novel",
        "new", "showed", "found", "associated", "significant", "effect",
        "effects", "role", "function", "expression", "level", "levels",
        "high", "low", "increased", "decreased", "compared", "group",
        "groups", "treatment", "treated", "control", "case", "cases",
        "disease", "diseases", "model", "models", "human", "mouse", "mice",
        "vitro", "vivo", "clinical", "therapeutic", "potential", "target",
        "via", "across", "among", "between", "through", "during", "after",
        "before", "following", "first", "second", "two", "three", "one",
        # TOO GENERIC - MeSH terms that are too broad
        "humans", "animals", "male", "female", "adult", "aged", "young",
        "cell", "cells", "cancer", "tumor", "tumors", "neoplasms", "neoplasm",
        "protein", "proteins", "gene", "genes", "cell line", "cell lines",
        "signal transduction", "molecular", "biology", "medicine",
        "prognosis", "diagnosis", "therapy", "risk", "factors",
        "retrospective", "prospective", "cohort", "review", "systematic",
        "meta-analysis", "randomized", "controlled", "trial", "trials",
        "survival", "mortality", "incidence", "prevalence",
        "in vitro", "in vivo", "ex vivo",
        "western blot", "pcr", "elisa", "immunohistochemistry",
        "statistical", "significant", "correlation", "regression",
        "mechanism", "mechanisms", "pathway", "pathways",
        "biomarkers", "markers", "expression", "regulation",
        "inflammation", "immune", "response", "responses",
        "oxidative stress", "apoptosis", "proliferation", "differentiation",
        "mice", "rats", "rodents", "tissue", "tissues", "blood", "serum",
    }

    # Minimum keyword length
    MIN_KEYWORD_LENGTH = 3

    # High-value biomedical terms to prioritize (specific, not generic)
    KNOWN_PHRASES = {
        # Immunotherapy & Cell Therapy
        "car-t", "car t", "car-nk", "pd-1", "pd-l1", "ctla-4", "lag-3", "tim-3",
        "checkpoint inhibitor", "immune checkpoint", "checkpoint blockade",
        "adoptive cell therapy", "tumor infiltrating lymphocyte", "til therapy",
        "bispecific antibody", "bispecific t-cell engager", "bite",
        "chimeric antigen receptor", "tcr therapy",

        # Gene Editing & Gene Therapy
        "crispr", "crispr-cas9", "cas9", "cas12", "cas13",
        "base editing", "prime editing", "gene editing",
        "aav", "adeno-associated virus", "lentiviral vector",
        "gene therapy", "gene silencing", "rnai", "sirna", "antisense",

        # RNA & mRNA Technology
        "mrna vaccine", "mrna therapeutics", "lipid nanoparticle", "lnp",
        "rna-seq", "single-cell rna", "scrna-seq", "spatial transcriptomics",

        # AI & Digital Health
        "machine learning", "deep learning", "artificial intelligence",
        "neural network", "transformer", "large language model", "llm",
        "digital pathology", "radiomics", "drug discovery ai",

        # Specific Cancer Types (more specific than "cancer")
        "pancreatic cancer", "glioblastoma", "triple negative breast",
        "non-small cell lung", "nsclc", "hepatocellular carcinoma", "hcc",
        "acute myeloid leukemia", "aml", "multiple myeloma",
        "colorectal cancer", "melanoma", "ovarian cancer", "prostate cancer",

        # Specific Mechanisms & Targets
        "ferroptosis", "pyroptosis", "necroptosis", "immunogenic cell death",
        "tumor microenvironment", "tme", "cancer-associated fibroblast", "caf",
        "myeloid-derived suppressor", "mdsc", "regulatory t cell", "treg",
        "exhausted t cell", "t cell exhaustion",

        # Drug Modalities
        "antibody-drug conjugate", "adc", "proteac", "degrader",
        "molecular glue", "allosteric inhibitor", "covalent inhibitor",
        "kinase inhibitor", "tyrosine kinase", "egfr", "her2", "kras",
        "braf", "alk", "ros1", "met", "fgfr", "vegf", "vegfr",

        # Omics & Biomarkers
        "liquid biopsy", "ctdna", "circulating tumor dna",
        "proteomics", "metabolomics", "multi-omics", "spatial omics",
        "single cell", "single-cell analysis",

        # Emerging Areas
        "organoid", "organ-on-chip", "3d bioprinting",
        "microbiome", "gut microbiota", "fecal microbiota transplant",
        "exosome", "extracellular vesicle",
        "senolytic", "senolytics", "cellular senescence",
        "glp-1", "glp-1 agonist", "semaglutide", "tirzepatide",
        "alzheimer", "amyloid", "tau protein", "neurodegeneration",
    }

    def __init__(self, history_dir: Optional[Path] = None):
        """
        Initialize trend analyzer.

        Args:
            history_dir: Directory to store trend history files
        """
        self.history_dir = history_dir or Path(__file__).parent.parent / "output" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def extract_keywords(self, papers: List[Paper]) -> Dict[str, int]:
        """
        Extract keywords from papers with frequency counts.

        Args:
            papers: List of Paper objects

        Returns:
            Dictionary of keyword -> count
        """
        keyword_counts = Counter()

        for paper in papers:
            # Combine title and abstract
            text = f"{paper.title} {paper.abstract}".lower()

            # Extract known phrases first (high priority)
            # But only if they appear as distinct terms, not partial matches
            for phrase in self.KNOWN_PHRASES:
                # Skip short ambiguous abbreviations that can be common words
                if len(phrase) <= 4 and phrase in {
                    "met", "set", "map", "can", "act", "age", "aim",
                    "adc", "alk", "ros", "caf", "tme", "bite", "lnp",
                    "aml", "hcc", "her2", "kras", "braf", "vegf",
                }:
                    # Skip these - they'll be picked up from MeSH/keywords if relevant
                    continue
                if phrase in text:
                    normalized = phrase.replace("-", " ").replace("  ", " ").strip()
                    # Known phrases get 3x weight
                    keyword_counts[normalized] += 3

            # Extract MeSH terms (high quality keywords) - with stopword filtering
            for mesh in paper.mesh_terms:
                normalized = mesh.lower().strip()
                if (
                    len(normalized) >= self.MIN_KEYWORD_LENGTH
                    and normalized not in self.STOPWORDS
                    and not self._is_generic_term(normalized)
                ):
                    keyword_counts[normalized] += 2  # MeSH terms get 2x weight

            # Extract author keywords - with stopword filtering
            for kw in paper.keywords:
                normalized = kw.lower().strip()
                if (
                    len(normalized) >= self.MIN_KEYWORD_LENGTH
                    and normalized not in self.STOPWORDS
                    and not self._is_generic_term(normalized)
                ):
                    keyword_counts[normalized] += 2  # Author keywords get 2x weight

        return dict(keyword_counts)

    def _is_generic_term(self, term: str) -> bool:
        """Check if a term is too generic to be useful."""
        generic_patterns = {
            # Single generic words
            "cell", "cells", "cancer", "tumor", "tumors", "human", "humans",
            "animal", "animals", "male", "female", "adult", "aged",
            "protein", "proteins", "gene", "genes",
            "treatment", "therapy", "study", "analysis",
            "patient", "patients", "method", "methods",
            "result", "results", "effect", "effects",
            "activity", "development", "expression",
            "significantly", "promising", "novel", "new",
            "drug", "drugs", "agent", "agents",
            # Ambiguous short abbreviations (could be common words)
            "met", "set", "map", "can", "act", "age", "aim",
            # Generic MeSH terms
            "cell line, tumor", "cell proliferation", "molecular structure",
            "dose-response relationship, drug", "drug screening assays, antitumor",
            "antineoplastic agents", "signal transduction",
            "structure-activity relationship",
            "antineoplastic combined chemotherapy protocols",
            "xenograft model antitumor assays",
        }
        return term in generic_patterns

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove special characters but keep hyphens in words
        text = re.sub(r"[^\w\s-]", " ", text)
        words = text.lower().split()
        return [w.strip("-") for w in words if w.strip("-")]

    def load_history(self, date: Optional[datetime] = None) -> Dict[str, int]:
        """Load keyword counts from history file."""
        if date is None:
            date = datetime.now() - timedelta(days=1)

        filename = f"trends_{date.strftime('%Y%m%d')}.json"
        filepath = self.history_dir / filename

        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)

        return {}

    def save_history(self, keyword_counts: Dict[str, int], date: Optional[datetime] = None):
        """Save keyword counts to history file."""
        if date is None:
            date = datetime.now()

        filename = f"trends_{date.strftime('%Y%m%d')}.json"
        filepath = self.history_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(keyword_counts, f, ensure_ascii=False, indent=2)

    def get_hot_topics(
        self,
        papers: List[Paper],
        top_n: int = 5,
        min_count: int = 2,
    ) -> List[Trend]:
        """
        Get hot topics from papers with trend comparison.

        Args:
            papers: List of Paper objects
            top_n: Number of top trends to return
            min_count: Minimum count to be considered a trend

        Returns:
            List of Trend objects sorted by relevance
        """
        # Extract current keywords
        current_counts = self.extract_keywords(papers)

        # Load historical data
        yesterday = datetime.now() - timedelta(days=1)
        week_ago = datetime.now() - timedelta(days=7)

        yesterday_counts = self.load_history(yesterday)
        week_ago_counts = self.load_history(week_ago)

        # Save today's counts
        self.save_history(current_counts)

        # Build trends
        trends = []
        for keyword, count in current_counts.items():
            if count < min_count:
                continue

            trend = Trend(
                keyword=keyword,
                count=count,
                previous_count=yesterday_counts.get(keyword, 0),
                week_ago_count=week_ago_counts.get(keyword, 0),
            )

            # Find representative papers for this keyword
            trend.representative_papers = self._find_representative_papers(
                keyword, papers, max_papers=3
            )

            trends.append(trend)

        # Sort by combined score: count * trend_boost
        def score(t: Trend) -> float:
            # Base score from count
            base = t.count

            # Boost for rising trends
            if t.day_change >= 50:
                base *= 2.0  # Hot
            elif t.day_change >= 20:
                base *= 1.5  # Rising fast
            elif t.day_change >= 10:
                base *= 1.2  # Rising

            return base

        trends.sort(key=score, reverse=True)

        return trends[:top_n]

    def _find_representative_papers(
        self,
        keyword: str,
        papers: List[Paper],
        max_papers: int = 3,
    ) -> List[Paper]:
        """Find papers that best represent a keyword."""
        scored_papers = []

        keyword_lower = keyword.lower()
        keyword_parts = keyword_lower.split()

        for paper in papers:
            score = 0
            text = f"{paper.title} {paper.abstract}".lower()

            # Title match is strongest
            if keyword_lower in paper.title.lower():
                score += 10

            # Abstract match
            if keyword_lower in paper.abstract.lower():
                score += 5

            # MeSH term match
            for mesh in paper.mesh_terms:
                if keyword_lower in mesh.lower():
                    score += 3

            # Keyword match
            for kw in paper.keywords:
                if keyword_lower in kw.lower():
                    score += 2

            # Partial word matches
            for part in keyword_parts:
                if part in text:
                    score += 1

            if score > 0:
                scored_papers.append((score, paper))

        # Sort by score and return top papers
        scored_papers.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_papers[:max_papers]]

    def select_representative_papers(
        self,
        trends: List[Trend],
        papers: List[Paper],
    ) -> Dict[str, List[Paper]]:
        """
        Select representative papers for each trend.

        Args:
            trends: List of Trend objects
            papers: List of Paper objects

        Returns:
            Dictionary mapping keyword to list of papers
        """
        result = {}

        for trend in trends:
            if not trend.representative_papers:
                trend.representative_papers = self._find_representative_papers(
                    trend.keyword, papers, max_papers=3
                )
            result[trend.keyword] = trend.representative_papers

        return result

    def generate_trend_summary(self, trends: List[Trend]) -> str:
        """Generate a text summary of trends."""
        lines = ["## Hot Topics Today\n"]

        for i, trend in enumerate(trends, 1):
            indicator = trend.trend_indicator
            change_text = ""

            if trend.day_change > 0:
                change_text = f"(+{trend.day_change:.0f}% vs yesterday)"
            elif trend.day_change < 0:
                change_text = f"({trend.day_change:.0f}% vs yesterday)"

            lines.append(f"{i}. {indicator} **{trend.keyword}** - {trend.count} papers {change_text}")

        return "\n".join(lines)


async def main():
    """Test the trend analyzer."""
    from .pubmed_fetcher import PubMedFetcher

    # Fetch papers
    fetcher = PubMedFetcher()
    print("Fetching papers...")
    papers = await fetcher.fetch_recent_papers(max_results=50, days=3)
    print(f"Fetched {len(papers)} papers")

    # Analyze trends
    analyzer = TrendAnalyzer()
    trends = analyzer.get_hot_topics(papers, top_n=5)

    print("\n" + analyzer.generate_trend_summary(trends))

    for trend in trends:
        print(f"\n=== {trend.keyword} ===")
        for paper in trend.representative_papers[:2]:
            print(f"  - {paper.title[:60]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
