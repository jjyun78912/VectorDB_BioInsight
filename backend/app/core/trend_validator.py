"""
Research Trend Validation Pipeline

Purpose: Automatically identify and validate research trends with explainable,
defensible evidence for user trust.

Pipeline Architecture:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                        TREND VALIDATION PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────────────────────────┐
│  1. CORPUS BUILDER (5-year window)                                          │
│     - Domain-specific paper collection                                       │
│     - MeSH term extraction                                                   │
│     - Keyword frequency tracking                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. SPARSE SIGNAL EXTRACTOR                                                  │
│     ┌──────────────────┬──────────────────┬──────────────────┐              │
│     │  MeSH Frequency  │   Growth Rate    │ Journal Diversity │              │
│     │  (Term counts)   │ (YoY % change)   │ (Unique journals) │              │
│     └──────────────────┴──────────────────┴──────────────────┘              │
│                                                                              │
│     Output: Candidate emerging topics with statistical significance          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. DENSE EMBEDDING CLUSTERING                                               │
│     - PubMedBERT embeddings for topic contextualization                      │
│     - UMAP/t-SNE for visualization                                           │
│     - Cluster coherence scoring                                              │
│     - Related topic discovery                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. VALIDATION LAYER                                                         │
│     ┌───────────────────────────────────────────────────────────────┐       │
│     │  Evidence Signals:                                             │       │
│     │  ✓ Review article mentions (systematic reviews, meta-analyses) │       │
│     │  ✓ Clinical guideline references                               │       │
│     │  ✓ Cross-journal support (>N unique high-IF journals)         │       │
│     │  ✓ Funding/grant mentions                                      │       │
│     │  ✓ Clinical trial activity                                     │       │
│     └───────────────────────────────────────────────────────────────┘       │
│                                                                              │
│     Output: Validation score + evidence citations                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. FINAL OUTPUT                                                             │
│     - Validated trend with confidence score (0-100)                          │
│     - Evidence summary (why this is trending)                                │
│     - Supporting citations                                                   │
│     - Visualization-ready data                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Validation Scoring Formula:
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Score = weighted sum of:
  - Publication Growth (25%): YoY growth rate over 5 years
  - Journal Diversity (20%): Unique high-IF journals publishing on topic
  - Review Coverage (20%): Mentions in systematic reviews/meta-analyses
  - Clinical Activity (20%): Active clinical trials
  - Research Gap Signal (15%): "Future research needed" pattern frequency

Confidence Levels:
  - 🟢 High (80-100): Strong evidence across multiple signals
  - 🟡 Medium (50-79): Good evidence but some signals weak
  - 🟠 Emerging (30-49): Early signs, watch closely
  - 🔴 Uncertain (<30): Insufficient evidence
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import httpx
import re
from collections import defaultdict
import numpy as np


# ============================================================================
# Data Models
# ============================================================================

class ConfidenceLevel(str, Enum):
    """Trend confidence levels."""
    HIGH = "high"           # 80-100
    MEDIUM = "medium"       # 50-79
    EMERGING = "emerging"   # 30-49
    UNCERTAIN = "uncertain" # <30


@dataclass
class SparseSignals:
    """Sparse statistical signals for trend detection."""
    keyword: str

    # Publication metrics
    total_papers_5yr: int = 0
    yearly_counts: Dict[int, int] = field(default_factory=dict)
    growth_rate_5yr: float = 0.0
    growth_rate_yoy: float = 0.0

    # MeSH/Keyword frequency
    mesh_terms: List[str] = field(default_factory=list)
    mesh_frequency: Dict[str, int] = field(default_factory=dict)

    # Journal diversity
    unique_journals: int = 0
    high_if_journals: int = 0  # Impact Factor > 5
    journal_list: List[str] = field(default_factory=list)


@dataclass
class ValidationEvidence:
    """Evidence supporting trend validation."""
    keyword: str

    # Review coverage
    systematic_reviews: int = 0
    meta_analyses: int = 0
    review_titles: List[str] = field(default_factory=list)

    # Clinical activity
    active_clinical_trials: int = 0
    trial_phases: Dict[str, int] = field(default_factory=dict)

    # Guideline mentions
    guideline_mentions: int = 0
    guideline_sources: List[str] = field(default_factory=list)

    # Research gap signals
    future_research_mentions: int = 0
    gap_ratio: float = 0.0  # future mentions / total papers

    # Funding signals
    grant_mentions: int = 0


@dataclass
class ValidatedTrend:
    """Fully validated research trend with evidence."""
    keyword: str

    # Scores (0-100)
    publication_score: float = 0.0
    diversity_score: float = 0.0
    review_score: float = 0.0
    clinical_score: float = 0.0
    gap_score: float = 0.0
    total_score: float = 0.0

    # Confidence
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNCERTAIN

    # Raw data
    sparse_signals: Optional[SparseSignals] = None
    validation_evidence: Optional[ValidationEvidence] = None

    # Human-readable summary
    summary: str = ""
    evidence_summary: List[str] = field(default_factory=list)

    # Timestamps
    validated_at: str = ""
    data_period: str = ""


# ============================================================================
# API Configuration
# ============================================================================
import os

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

# NCBI API Key (allows 10 req/sec instead of 3 req/sec)
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Rate limiting: delay between requests (seconds)
# Without API key: 0.34s (3 req/sec), With API key: 0.1s (10 req/sec)
PUBMED_REQUEST_DELAY = 0.1 if NCBI_API_KEY else 0.35

# Cache
_validation_cache: Dict[str, ValidatedTrend] = {}
_cache_expiry: Dict[str, datetime] = {}
CACHE_HOURS = 24

# Year configuration
CURRENT_YEAR = datetime.now().year
TREND_WINDOW_YEARS = 5


# ============================================================================
# Sparse Signal Extraction
# ============================================================================

class SparseSignalExtractor:
    """Extract statistical signals from PubMed data."""

    def __init__(self):
        self.client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)
        return self.client

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def get_pubmed_count(self, query: str, year: int) -> int:
        """Get publication count for a query in a specific year."""
        try:
            # Rate limiting delay
            await asyncio.sleep(PUBMED_REQUEST_DELAY)

            client = await self._get_client()
            params = {
                "db": "pubmed",
                "term": query,
                "rettype": "count",
                "datetype": "pdat",
                "mindate": f"{year}/01/01",
                "maxdate": f"{year}/12/31",
            }
            # Add API key if available
            if NCBI_API_KEY:
                params["api_key"] = NCBI_API_KEY

            response = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
            response.raise_for_status()

            match = re.search(r'<Count>(\d+)</Count>', response.text)
            return int(match.group(1)) if match else 0
        except Exception as e:
            print(f"PubMed count error: {e}")
            return 0

    async def get_journal_diversity(self, keyword: str, limit: int = 500) -> Tuple[int, int, List[str]]:
        """
        Get journal diversity metrics.
        Returns: (unique_journals, high_if_journals, journal_list)
        """
        try:
            # Rate limiting delay
            await asyncio.sleep(PUBMED_REQUEST_DELAY)

            client = await self._get_client()

            # Search recent 2 years for journal diversity
            params = {
                "db": "pubmed",
                "term": f"{keyword}",
                "retmax": limit,
                "rettype": "uilist",
                "datetype": "pdat",
                "mindate": f"{CURRENT_YEAR - 2}/01/01",
                "maxdate": f"{CURRENT_YEAR}/12/31",
            }
            if NCBI_API_KEY:
                params["api_key"] = NCBI_API_KEY

            # Get PMIDs
            response = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
            pmids = re.findall(r'<Id>(\d+)</Id>', response.text)

            if not pmids:
                return 0, 0, []

            # Fetch journal info (sample)
            sample_pmids = pmids[:100]  # Limit for speed

            summary_params = {
                "db": "pubmed",
                "id": ",".join(sample_pmids),
                "retmode": "xml",
            }
            if NCBI_API_KEY:
                summary_params["api_key"] = NCBI_API_KEY

            await asyncio.sleep(PUBMED_REQUEST_DELAY)  # Rate limiting
            response = await client.get(f"{PUBMED_BASE}/efetch.fcgi", params=summary_params)

            # Extract journal names
            journals = re.findall(r'<Title>([^<]+)</Title>', response.text)
            unique_journals = list(set(journals))

            # High-IF journals (simplified - check for known high-IF journals)
            high_if_keywords = [
                'Nature', 'Science', 'Cell', 'Lancet', 'NEJM', 'JAMA',
                'BMJ', 'Cancer', 'Immunity', 'Neuron', 'Blood'
            ]
            high_if_count = sum(
                1 for j in unique_journals
                if any(kw.lower() in j.lower() for kw in high_if_keywords)
            )

            return len(unique_journals), high_if_count, unique_journals[:20]

        except Exception as e:
            print(f"Journal diversity error: {e}")
            return 0, 0, []

    async def extract_signals(self, keyword: str) -> SparseSignals:
        """Extract all sparse signals for a keyword."""
        signals = SparseSignals(keyword=keyword)

        # Get yearly publication counts
        years = list(range(CURRENT_YEAR - TREND_WINDOW_YEARS + 1, CURRENT_YEAR + 1))

        for year in years:
            count = await self.get_pubmed_count(keyword, year)
            signals.yearly_counts[year] = count
            await asyncio.sleep(0.35)  # Rate limiting

        signals.total_papers_5yr = sum(signals.yearly_counts.values())

        # Calculate growth rates
        counts = [signals.yearly_counts.get(y, 0) for y in sorted(signals.yearly_counts.keys())]

        if len(counts) >= 2 and counts[0] > 0:
            signals.growth_rate_5yr = ((counts[-1] - counts[0]) / counts[0]) * 100

        if len(counts) >= 2 and counts[-2] > 0:
            signals.growth_rate_yoy = ((counts[-1] - counts[-2]) / counts[-2]) * 100

        # Get journal diversity
        unique, high_if, journal_list = await self.get_journal_diversity(keyword)
        signals.unique_journals = unique
        signals.high_if_journals = high_if
        signals.journal_list = journal_list

        return signals


# ============================================================================
# Validation Evidence Collector
# ============================================================================

class ValidationEvidenceCollector:
    """Collect validation evidence from multiple sources."""

    def __init__(self):
        self.client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=30.0)
        return self.client

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def get_pubmed_count(self, query: str, years_back: int = 3) -> int:
        """Get publication count for a query."""
        try:
            # Rate limiting delay
            await asyncio.sleep(PUBMED_REQUEST_DELAY)

            client = await self._get_client()
            params = {
                "db": "pubmed",
                "term": query,
                "rettype": "count",
                "datetype": "pdat",
                "mindate": f"{CURRENT_YEAR - years_back}/01/01",
                "maxdate": f"{CURRENT_YEAR}/12/31",
            }
            if NCBI_API_KEY:
                params["api_key"] = NCBI_API_KEY

            response = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
            match = re.search(r'<Count>(\d+)</Count>', response.text)
            return int(match.group(1)) if match else 0
        except Exception as e:
            print(f"PubMed error: {e}")
            return 0

    async def get_review_coverage(self, keyword: str) -> Tuple[int, int, List[str]]:
        """
        Get systematic review and meta-analysis coverage.
        Returns: (systematic_reviews, meta_analyses, sample_titles)
        """
        try:
            client = await self._get_client()

            # Search for systematic reviews
            sr_query = f'"{keyword}" AND (systematic review[pt] OR systematic review[ti])'
            sr_count = await self.get_pubmed_count(sr_query, years_back=5)

            await asyncio.sleep(PUBMED_REQUEST_DELAY)

            # Search for meta-analyses
            ma_query = f'"{keyword}" AND (meta-analysis[pt] OR meta-analysis[ti])'
            ma_count = await self.get_pubmed_count(ma_query, years_back=5)

            return sr_count, ma_count, []

        except Exception as e:
            print(f"Review coverage error: {e}")
            return 0, 0, []

    async def get_clinical_trials(self, keyword: str) -> Tuple[int, Dict[str, int]]:
        """
        Get clinical trial activity from ClinicalTrials.gov.
        Returns: (active_trials, phase_distribution)
        """
        try:
            client = await self._get_client()

            params = {
                "query.term": keyword,
                "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING",
                "countTotal": "true",
                "pageSize": 100,
            }

            response = await client.get(CLINICALTRIALS_API, params=params)

            if response.status_code != 200:
                return 0, {}

            data = response.json()
            total = data.get("totalCount", 0)

            # Count by phase
            phases = defaultdict(int)
            for study in data.get("studies", []):
                phase = study.get("protocolSection", {}).get("designModule", {}).get("phases", ["N/A"])
                if isinstance(phase, list) and phase:
                    phases[phase[0]] += 1

            return total, dict(phases)

        except Exception as e:
            print(f"Clinical trials error: {e}")
            return 0, {}

    async def get_future_research_mentions(self, keyword: str, total_papers: int) -> Tuple[int, float]:
        """
        Get "future research needed" pattern mentions.
        Returns: (mention_count, gap_ratio)
        """
        try:
            query = (
                f'"{keyword}" AND ('
                '"future research" OR "further investigation" OR '
                '"remains unclear" OR "warrants further study" OR '
                '"more research is needed"'
                ')'
            )

            count = await self.get_pubmed_count(query, years_back=3)

            ratio = (count / total_papers * 100) if total_papers > 0 else 0

            return count, ratio

        except Exception as e:
            print(f"Future research error: {e}")
            return 0, 0.0

    async def get_guideline_mentions(self, keyword: str) -> Tuple[int, List[str]]:
        """
        Search for guideline mentions.
        Returns: (count, source_organizations)
        """
        try:
            query = (
                f'"{keyword}" AND ('
                'guideline[pt] OR practice guideline[pt] OR '
                '"clinical guideline" OR "treatment guideline"'
                ')'
            )

            count = await self.get_pubmed_count(query, years_back=5)

            return count, []

        except Exception as e:
            print(f"Guideline error: {e}")
            return 0, []

    async def collect_evidence(self, keyword: str, total_papers: int) -> ValidationEvidence:
        """Collect all validation evidence for a keyword."""
        evidence = ValidationEvidence(keyword=keyword)

        # Get review coverage
        sr, ma, titles = await self.get_review_coverage(keyword)
        evidence.systematic_reviews = sr
        evidence.meta_analyses = ma
        evidence.review_titles = titles

        await asyncio.sleep(0.35)

        # Get clinical trial activity
        trials, phases = await self.get_clinical_trials(keyword)
        evidence.active_clinical_trials = trials
        evidence.trial_phases = phases

        await asyncio.sleep(0.35)

        # Get guideline mentions
        guidelines, sources = await self.get_guideline_mentions(keyword)
        evidence.guideline_mentions = guidelines
        evidence.guideline_sources = sources

        await asyncio.sleep(0.35)

        # Get future research signals
        future, ratio = await self.get_future_research_mentions(keyword, total_papers)
        evidence.future_research_mentions = future
        evidence.gap_ratio = ratio

        return evidence


# ============================================================================
# Trend Validator (Main Class)
# ============================================================================

class TrendValidator:
    """
    Main trend validation pipeline.

    Validates research trends with explainable, defensible evidence.
    """

    # Scoring weights
    WEIGHTS = {
        'publication': 0.25,   # Publication growth
        'diversity': 0.20,     # Journal diversity
        'review': 0.20,        # Review coverage
        'clinical': 0.20,      # Clinical activity
        'gap': 0.15,           # Research gap signals
    }

    def __init__(self):
        self.signal_extractor = SparseSignalExtractor()
        self.evidence_collector = ValidationEvidenceCollector()

    async def close(self):
        """Clean up resources."""
        await self.signal_extractor.close()
        await self.evidence_collector.close()

    def _calculate_publication_score(self, signals: SparseSignals) -> float:
        """
        Calculate publication growth score (0-100).

        Criteria:
        - 5-year growth rate
        - YoY momentum
        - Absolute volume
        """
        score = 0.0

        # Growth rate component (max 50 points)
        if signals.growth_rate_5yr >= 100:
            score += 50
        elif signals.growth_rate_5yr >= 50:
            score += 40
        elif signals.growth_rate_5yr >= 25:
            score += 30
        elif signals.growth_rate_5yr >= 10:
            score += 20
        elif signals.growth_rate_5yr > 0:
            score += 10

        # YoY momentum (max 30 points)
        if signals.growth_rate_yoy >= 30:
            score += 30
        elif signals.growth_rate_yoy >= 15:
            score += 20
        elif signals.growth_rate_yoy >= 5:
            score += 10

        # Volume bonus (max 20 points)
        if signals.total_papers_5yr >= 10000:
            score += 20
        elif signals.total_papers_5yr >= 5000:
            score += 15
        elif signals.total_papers_5yr >= 1000:
            score += 10
        elif signals.total_papers_5yr >= 100:
            score += 5

        return min(100, score)

    def _calculate_diversity_score(self, signals: SparseSignals) -> float:
        """
        Calculate journal diversity score (0-100).

        Criteria:
        - Number of unique journals
        - Presence in high-IF journals
        """
        score = 0.0

        # Unique journals (max 60 points)
        if signals.unique_journals >= 50:
            score += 60
        elif signals.unique_journals >= 30:
            score += 45
        elif signals.unique_journals >= 15:
            score += 30
        elif signals.unique_journals >= 5:
            score += 15

        # High-IF journals (max 40 points)
        if signals.high_if_journals >= 5:
            score += 40
        elif signals.high_if_journals >= 3:
            score += 30
        elif signals.high_if_journals >= 1:
            score += 15

        return min(100, score)

    def _calculate_review_score(self, evidence: ValidationEvidence) -> float:
        """
        Calculate review coverage score (0-100).

        Criteria:
        - Systematic reviews
        - Meta-analyses
        """
        score = 0.0

        # Systematic reviews (max 60 points)
        if evidence.systematic_reviews >= 50:
            score += 60
        elif evidence.systematic_reviews >= 20:
            score += 45
        elif evidence.systematic_reviews >= 10:
            score += 30
        elif evidence.systematic_reviews >= 5:
            score += 20
        elif evidence.systematic_reviews >= 1:
            score += 10

        # Meta-analyses (max 40 points)
        if evidence.meta_analyses >= 20:
            score += 40
        elif evidence.meta_analyses >= 10:
            score += 30
        elif evidence.meta_analyses >= 5:
            score += 20
        elif evidence.meta_analyses >= 1:
            score += 10

        return min(100, score)

    def _calculate_clinical_score(self, evidence: ValidationEvidence) -> float:
        """
        Calculate clinical activity score (0-100).

        Criteria:
        - Active clinical trials
        - Phase distribution
        """
        score = 0.0

        # Active trials (max 70 points)
        if evidence.active_clinical_trials >= 100:
            score += 70
        elif evidence.active_clinical_trials >= 50:
            score += 55
        elif evidence.active_clinical_trials >= 20:
            score += 40
        elif evidence.active_clinical_trials >= 10:
            score += 25
        elif evidence.active_clinical_trials >= 1:
            score += 10

        # Phase 3 bonus (max 30 points)
        phase3 = evidence.trial_phases.get('PHASE3', 0)
        if phase3 >= 10:
            score += 30
        elif phase3 >= 5:
            score += 20
        elif phase3 >= 1:
            score += 10

        return min(100, score)

    def _calculate_gap_score(self, evidence: ValidationEvidence) -> float:
        """
        Calculate research gap score (0-100).

        Higher gap ratio indicates more research opportunities.
        """
        # Gap ratio scoring
        ratio = evidence.gap_ratio

        if ratio >= 10:
            return 100
        elif ratio >= 7:
            return 80
        elif ratio >= 5:
            return 60
        elif ratio >= 3:
            return 40
        elif ratio >= 1:
            return 20
        else:
            return 10  # Baseline score

    def _determine_confidence(self, total_score: float) -> ConfidenceLevel:
        """Determine confidence level from total score."""
        if total_score >= 80:
            return ConfidenceLevel.HIGH
        elif total_score >= 50:
            return ConfidenceLevel.MEDIUM
        elif total_score >= 30:
            return ConfidenceLevel.EMERGING
        else:
            return ConfidenceLevel.UNCERTAIN

    def _generate_summary(self, trend: ValidatedTrend) -> str:
        """Generate human-readable trend summary."""
        signals = trend.sparse_signals
        evidence = trend.validation_evidence

        parts = []

        # Growth description
        if signals.growth_rate_5yr >= 50:
            parts.append(f"급성장 중 (5년간 {signals.growth_rate_5yr:.0f}% 증가)")
        elif signals.growth_rate_5yr >= 20:
            parts.append(f"꾸준한 성장세 (5년간 {signals.growth_rate_5yr:.0f}% 증가)")
        elif signals.growth_rate_5yr > 0:
            parts.append(f"완만한 성장 (5년간 {signals.growth_rate_5yr:.0f}% 증가)")

        # Clinical activity
        if evidence.active_clinical_trials >= 20:
            parts.append(f"{evidence.active_clinical_trials}개 활성 임상시험 진행 중")

        # Review coverage
        total_reviews = evidence.systematic_reviews + evidence.meta_analyses
        if total_reviews >= 10:
            parts.append(f"{total_reviews}편의 체계적 문헌고찰/메타분석 발표")

        return " • ".join(parts) if parts else "분석 데이터 수집 중"

    def _generate_evidence_summary(self, trend: ValidatedTrend) -> List[str]:
        """Generate list of evidence points."""
        evidence_points = []
        signals = trend.sparse_signals
        evidence = trend.validation_evidence

        # Publication evidence
        evidence_points.append(
            f"📊 최근 5년간 {signals.total_papers_5yr:,}편 논문 발표 "
            f"(연평균 성장률: {signals.growth_rate_5yr:.1f}%)"
        )

        # Journal diversity
        if signals.unique_journals > 0:
            evidence_points.append(
                f"📰 {signals.unique_journals}개 학술지에서 연구 발표 "
                f"(고영향력 저널 {signals.high_if_journals}개 포함)"
            )

        # Review coverage
        if evidence.systematic_reviews > 0 or evidence.meta_analyses > 0:
            evidence_points.append(
                f"📚 체계적 문헌고찰 {evidence.systematic_reviews}편, "
                f"메타분석 {evidence.meta_analyses}편 발표"
            )

        # Clinical trials
        if evidence.active_clinical_trials > 0:
            evidence_points.append(
                f"🏥 현재 {evidence.active_clinical_trials}개 임상시험 활성화"
            )

        # Research gaps
        if evidence.future_research_mentions > 0:
            evidence_points.append(
                f"🔬 {evidence.future_research_mentions}편의 논문에서 "
                f"추가 연구 필요성 언급 (연구 갭 비율: {evidence.gap_ratio:.1f}%)"
            )

        return evidence_points

    async def validate_keyword(self, keyword: str, use_cache: bool = True) -> ValidatedTrend:
        """
        Validate a single keyword as a research trend.

        Args:
            keyword: The keyword/topic to validate
            use_cache: Whether to use cached results

        Returns:
            ValidatedTrend with scores and evidence
        """
        # Check cache
        cache_key = keyword.lower().strip()
        if use_cache and cache_key in _validation_cache:
            if datetime.now() < _cache_expiry.get(cache_key, datetime.min):
                return _validation_cache[cache_key]

        # Extract sparse signals
        signals = await self.signal_extractor.extract_signals(keyword)

        # Collect validation evidence
        evidence = await self.evidence_collector.collect_evidence(
            keyword, signals.total_papers_5yr
        )

        # Calculate scores
        pub_score = self._calculate_publication_score(signals)
        div_score = self._calculate_diversity_score(signals)
        rev_score = self._calculate_review_score(evidence)
        clin_score = self._calculate_clinical_score(evidence)
        gap_score = self._calculate_gap_score(evidence)

        # Weighted total
        total_score = (
            pub_score * self.WEIGHTS['publication'] +
            div_score * self.WEIGHTS['diversity'] +
            rev_score * self.WEIGHTS['review'] +
            clin_score * self.WEIGHTS['clinical'] +
            gap_score * self.WEIGHTS['gap']
        )

        # Build validated trend
        trend = ValidatedTrend(
            keyword=keyword,
            publication_score=round(pub_score, 1),
            diversity_score=round(div_score, 1),
            review_score=round(rev_score, 1),
            clinical_score=round(clin_score, 1),
            gap_score=round(gap_score, 1),
            total_score=round(total_score, 1),
            confidence_level=self._determine_confidence(total_score),
            sparse_signals=signals,
            validation_evidence=evidence,
            validated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            data_period=f"{CURRENT_YEAR - TREND_WINDOW_YEARS + 1}-{CURRENT_YEAR}"
        )

        # Generate summaries
        trend.summary = self._generate_summary(trend)
        trend.evidence_summary = self._generate_evidence_summary(trend)

        # Cache result
        _validation_cache[cache_key] = trend
        _cache_expiry[cache_key] = datetime.now() + timedelta(hours=CACHE_HOURS)

        return trend

    async def validate_keywords(
        self,
        keywords: List[str],
        min_score: float = 30.0
    ) -> List[ValidatedTrend]:
        """
        Validate multiple keywords and filter by minimum score.

        Args:
            keywords: List of keywords to validate
            min_score: Minimum total score to include

        Returns:
            List of ValidatedTrend sorted by score (descending)
        """
        trends = []

        for keyword in keywords:
            trend = await self.validate_keyword(keyword)
            if trend.total_score >= min_score:
                trends.append(trend)

        # Sort by score
        trends.sort(key=lambda t: t.total_score, reverse=True)

        return trends

    async def get_validated_defaults(self) -> List[ValidatedTrend]:
        """
        Get validated default keywords for TrendAnalysis.

        Returns top 5 validated hot topics.
        """
        # Candidate keywords (broader pool)
        candidates = [
            'CRISPR',
            'CAR-T therapy',
            'mRNA vaccine',
            'AlphaFold',
            'single-cell RNA-seq',
            'immune checkpoint inhibitor',
            'liquid biopsy',
            'spatial transcriptomics',
            'tumor microenvironment',
            'gut-brain axis',
        ]

        validated = await self.validate_keywords(candidates, min_score=40.0)

        return validated[:5]


# ============================================================================
# Convenience Functions
# ============================================================================

async def validate_trend(keyword: str) -> ValidatedTrend:
    """Convenience function to validate a single keyword."""
    validator = TrendValidator()
    try:
        return await validator.validate_keyword(keyword)
    finally:
        await validator.close()


async def get_validated_hot_topics(
    domain_keywords: List[str],
    min_score: float = 40.0,
    limit: int = 10
) -> List[ValidatedTrend]:
    """
    Get validated hot topics for a domain.

    Args:
        domain_keywords: List of candidate keywords
        min_score: Minimum validation score
        limit: Maximum number of results

    Returns:
        List of validated trends
    """
    validator = TrendValidator()
    try:
        validated = await validator.validate_keywords(domain_keywords, min_score)
        return validated[:limit]
    finally:
        await validator.close()
