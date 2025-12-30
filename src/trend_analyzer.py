"""
Trend Analysis Module for Research Papers.

Features:
- Keyword frequency analysis
- Year-over-year research trend detection
- Topic evolution tracking
- Research gap identification
"""
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Optional
import re
import json

from .vector_store import BioVectorStore, create_vector_store
from .config import GOOGLE_API_KEY, GEMINI_MODEL

# Optional imports for NLP
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class TrendReport:
    """Research trend analysis report."""
    disease_domain: str
    total_papers: int
    year_range: tuple[int, int]

    # Keyword analysis
    top_keywords: list[tuple[str, int]] = field(default_factory=list)
    keyword_by_year: dict[int, list[tuple[str, int]]] = field(default_factory=dict)

    # Trend detection
    emerging_topics: list[str] = field(default_factory=list)  # 최근 급증
    declining_topics: list[str] = field(default_factory=list)  # 감소 추세
    stable_topics: list[str] = field(default_factory=list)     # 꾸준한 관심

    # Research gaps
    potential_gaps: list[str] = field(default_factory=list)

    # Year distribution
    papers_by_year: dict[int, int] = field(default_factory=dict)

    def format(self) -> str:
        """Format report for display."""
        output = []
        output.append("=" * 70)
        output.append(f"📊 Research Trend Analysis: {self.disease_domain}")
        output.append("=" * 70)
        output.append(f"Total Papers: {self.total_papers}")
        output.append(f"Year Range: {self.year_range[0]} - {self.year_range[1]}")

        output.append("\n📅 Papers by Year")
        output.append("-" * 40)
        for year in sorted(self.papers_by_year.keys(), reverse=True):
            count = self.papers_by_year[year]
            bar = "█" * count + "░" * (10 - count)
            output.append(f"  {year}: {bar} {count}")

        output.append("\n🔑 Top Keywords (Overall)")
        output.append("-" * 40)
        for i, (keyword, count) in enumerate(self.top_keywords[:15], 1):
            output.append(f"  {i:2}. {keyword}: {count}")

        if self.emerging_topics:
            output.append("\n🚀 Emerging Topics (Recent Surge)")
            output.append("-" * 40)
            for topic in self.emerging_topics:
                output.append(f"  • {topic}")

        if self.declining_topics:
            output.append("\n📉 Declining Topics")
            output.append("-" * 40)
            for topic in self.declining_topics:
                output.append(f"  • {topic}")

        if self.stable_topics:
            output.append("\n📌 Stable Topics (Consistent Interest)")
            output.append("-" * 40)
            for topic in self.stable_topics:
                output.append(f"  • {topic}")

        if self.potential_gaps:
            output.append("\n🔍 Potential Research Gaps")
            output.append("-" * 40)
            for gap in self.potential_gaps:
                output.append(f"  • {gap}")

        return "\n".join(output)


class TrendAnalyzer:
    """
    Analyze research trends from indexed papers.

    Provides insights on:
    - Keyword frequency and evolution
    - Research direction changes
    - Emerging and declining topics
    """

    # Biomedical stopwords to filter out
    STOPWORDS = {
        'study', 'studies', 'patient', 'patients', 'case', 'cases',
        'result', 'results', 'method', 'methods', 'conclusion', 'conclusions',
        'background', 'introduction', 'discussion', 'figure', 'table',
        'data', 'analysis', 'group', 'groups', 'treatment', 'therapy',
        'level', 'levels', 'value', 'values', 'significant', 'significantly',
        'however', 'therefore', 'although', 'moreover', 'furthermore',
        'using', 'used', 'based', 'associated', 'related', 'compared',
        'showed', 'found', 'observed', 'reported', 'demonstrated',
        'increased', 'decreased', 'higher', 'lower', 'present', 'presence',
        'abstract', 'keywords', 'author', 'authors', 'doi', 'published',
        'journal', 'volume', 'issue', 'page', 'pages', 'year', 'month',
        'university', 'department', 'institute', 'hospital', 'center',
        'email', 'corresponding', 'received', 'accepted', 'available',
    }

    def __init__(self, disease_domain: str):
        """Initialize trend analyzer."""
        self.disease_domain = disease_domain
        self.vector_store = create_vector_store(disease_domain=disease_domain)

    def analyze(self) -> TrendReport:
        """Run full trend analysis."""
        # Get all documents
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        if not all_data["ids"]:
            return TrendReport(
                disease_domain=self.disease_domain,
                total_papers=0,
                year_range=(0, 0)
            )

        # Organize by paper and year
        papers = {}
        docs_by_year = defaultdict(list)

        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            paper_title = meta.get("paper_title", "Unknown")
            year_str = meta.get("year", "")

            # Extract year
            year = self._extract_year(year_str)

            if paper_title not in papers:
                papers[paper_title] = {
                    "year": year,
                    "doi": meta.get("doi", ""),
                    "content": []
                }
            papers[paper_title]["content"].append(doc)

            if year:
                docs_by_year[year].append(doc)

        # Calculate statistics
        years = [p["year"] for p in papers.values() if p["year"]]
        year_range = (min(years), max(years)) if years else (0, 0)

        papers_by_year = Counter(p["year"] for p in papers.values() if p["year"])

        # Keyword extraction
        all_keywords = self._extract_keywords(all_data["documents"])
        keyword_by_year = {}
        for year, docs in docs_by_year.items():
            keyword_by_year[year] = self._extract_keywords(docs)[:20]

        # Trend detection
        emerging, declining, stable = self._detect_trends(keyword_by_year, year_range)

        # Research gap identification
        gaps = self._identify_gaps(all_keywords, papers)

        return TrendReport(
            disease_domain=self.disease_domain,
            total_papers=len(papers),
            year_range=year_range,
            top_keywords=all_keywords[:20],
            keyword_by_year=keyword_by_year,
            emerging_topics=emerging,
            declining_topics=declining,
            stable_topics=stable,
            potential_gaps=gaps,
            papers_by_year=dict(papers_by_year)
        )

    def _extract_year(self, year_str: str) -> int:
        """Extract year from string."""
        if not year_str:
            return 0

        # Try direct conversion
        try:
            year = int(year_str)
            if 1900 <= year <= 2030:
                return year
        except (ValueError, TypeError):
            pass

        # Try to find year in string
        match = re.search(r'(19|20)\d{2}', str(year_str))
        if match:
            return int(match.group())

        return 0

    def _extract_keywords(self, documents: list[str], top_n: int = 50) -> list[tuple[str, int]]:
        """Extract keywords from documents using TF-IDF or simple frequency."""
        # Combine all text
        all_text = " ".join(documents).lower()

        # Extract words (alphanumeric, 3+ chars)
        words = re.findall(r'\b[a-z]{3,}\b', all_text)

        # Filter stopwords
        filtered = [w for w in words if w not in self.STOPWORDS and len(w) > 3]

        # Count frequencies
        counter = Counter(filtered)

        return counter.most_common(top_n)

    def _detect_trends(
        self,
        keyword_by_year: dict[int, list[tuple[str, int]]],
        year_range: tuple[int, int]
    ) -> tuple[list[str], list[str], list[str]]:
        """Detect emerging, declining, and stable topics."""
        if not keyword_by_year or year_range[0] == 0:
            return [], [], []

        # Get recent vs older years
        years = sorted(keyword_by_year.keys())
        if len(years) < 2:
            return [], [], []

        mid_point = len(years) // 2
        recent_years = years[mid_point:]
        older_years = years[:mid_point]

        # Count keyword appearances
        recent_keywords = Counter()
        older_keywords = Counter()

        for year in recent_years:
            for kw, count in keyword_by_year.get(year, []):
                recent_keywords[kw] += count

        for year in older_years:
            for kw, count in keyword_by_year.get(year, []):
                older_keywords[kw] += count

        # Identify trends
        emerging = []
        declining = []
        stable = []

        all_keywords = set(recent_keywords.keys()) | set(older_keywords.keys())

        for kw in all_keywords:
            recent = recent_keywords.get(kw, 0)
            older = older_keywords.get(kw, 0)

            if older == 0 and recent > 5:
                emerging.append(kw)
            elif recent > older * 2 and recent > 5:
                emerging.append(kw)
            elif older > recent * 2 and older > 5:
                declining.append(kw)
            elif recent > 5 and older > 5 and 0.5 <= recent/older <= 2:
                stable.append(kw)

        return emerging[:10], declining[:10], stable[:10]

    def _identify_gaps(
        self,
        keywords: list[tuple[str, int]],
        papers: dict
    ) -> list[str]:
        """Identify potential research gaps."""
        gaps = []

        # Common research gap patterns in biomedical research
        gap_patterns = [
            ("long-term", "longitudinal studies needed"),
            ("pediatric", "more pediatric studies needed"),
            ("prospective", "prospective validation needed"),
            ("randomized", "randomized controlled trials needed"),
            ("biomarker", "biomarker validation studies needed"),
            ("mechanism", "mechanistic studies needed"),
            ("genetic", "genetic correlation studies needed"),
        ]

        keyword_set = {kw for kw, _ in keywords}

        for pattern, gap_msg in gap_patterns:
            if pattern not in keyword_set:
                gaps.append(gap_msg)

        return gaps[:5]


def create_trend_analyzer(disease_domain: str) -> TrendAnalyzer:
    """Create a trend analyzer instance."""
    return TrendAnalyzer(disease_domain=disease_domain)
