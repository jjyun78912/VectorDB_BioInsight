"""
Trend Analysis API - Multi-dimensional Hot Topics Discovery

Features:
- Rising Keywords: YoY growth rate from PubMed
- Research Interest: PubMed search trends, citation velocity
- Active Research: Publication volume, clinical trials
- Future Directions: "future research needed" pattern analysis

Data Sources:
- PubMed E-utilities API
- ClinicalTrials.gov API
- Paper abstract analysis
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import asyncio
import re
from datetime import datetime
from collections import defaultdict

router = APIRouter()

# API endpoints
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"

# Cache for trend data
_trend_cache: Dict[str, Dict] = {}
_cache_expiry: Dict[str, datetime] = {}
CACHE_DURATION_HOURS = 24

# Dynamic year calculation
CURRENT_YEAR = datetime.now().year
DEFAULT_START_YEAR = CURRENT_YEAR - 4
DEFAULT_END_YEAR = CURRENT_YEAR


# ============== Models ==============

class YearlyCount(BaseModel):
    """Yearly publication count."""
    year: int
    count: int
    growth_rate: Optional[float] = None


class KeywordTrend(BaseModel):
    """Keyword trend over years."""
    keyword: str
    total_count: int
    yearly_counts: List[YearlyCount]
    trend_direction: str  # "rising", "stable", "declining"
    growth_5yr: Optional[float] = None
    peak_year: Optional[int] = None


class TrendAnalysisResponse(BaseModel):
    """Trend analysis response."""
    query: str
    years: List[int]
    trends: List[KeywordTrend]
    comparison_keywords: List[str]
    analysis_date: str


class MultiDimensionalScore(BaseModel):
    """Multi-dimensional hot topic score."""
    rising_score: float        # YoY growth rate (0-100)
    interest_score: float      # Search/citation interest (0-100)
    activity_score: float      # Active research volume (0-100)
    future_score: float        # Future research potential (0-100)
    total_score: float         # Weighted composite score


class EnhancedHotTopic(BaseModel):
    """Enhanced hot topic with multi-dimensional analysis."""
    keyword: str
    scores: MultiDimensionalScore

    # Raw metrics
    current_year_papers: int
    previous_year_papers: int
    growth_rate: float
    clinical_trials: int
    future_mentions: int

    # Insights
    trend_label: str           # "Explosive", "Rising", "Stable", "Emerging"
    research_stage: str        # "Early", "Growing", "Mature", "Declining"
    recommendation: str        # Brief insight


class EnhancedHotTopicsResponse(BaseModel):
    """Enhanced hot topics response."""
    domain: str
    hot_topics: List[EnhancedHotTopic]
    analysis_period: str
    methodology: str
    last_updated: str


# Legacy models for backward compatibility
class HotTopic(BaseModel):
    """Hot topic with recent surge."""
    keyword: str
    recent_count: int
    previous_count: int
    growth_rate: float
    sample_titles: List[str] = []


class HotTopicsResponse(BaseModel):
    """Hot topics response."""
    domain: str
    hot_topics: List[HotTopic]
    analysis_period: str


# ============== Data Fetching Functions ==============

async def get_pubmed_count(keyword: str, year: int) -> int:
    """Get publication count for a keyword in a specific year."""
    from datetime import timedelta

    cache_key = f"{keyword}_{year}"
    if cache_key in _trend_cache:
        if datetime.now() < _cache_expiry.get(cache_key, datetime.min):
            return _trend_cache[cache_key]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "db": "pubmed",
                "term": keyword,
                "rettype": "count",
                "datetype": "pdat",
                "mindate": f"{year}/01/01",
                "maxdate": f"{year}/12/31",
            }
            response = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
            response.raise_for_status()

            match = re.search(r'<Count>(\d+)</Count>', response.text)
            count = int(match.group(1)) if match else 0

            _trend_cache[cache_key] = count
            _cache_expiry[cache_key] = datetime.now() + timedelta(hours=CACHE_DURATION_HOURS)
            return count

    except Exception as e:
        print(f"PubMed API error for {keyword} ({year}): {e}")
        return 0


async def get_clinical_trials_count(keyword: str) -> int:
    """Get active clinical trials count from ClinicalTrials.gov."""
    cache_key = f"ct_{keyword}"
    if cache_key in _trend_cache:
        if datetime.now() < _cache_expiry.get(cache_key, datetime.min):
            return _trend_cache[cache_key]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "query.term": keyword,
                "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING",
                "countTotal": "true",
                "pageSize": 1,
            }
            response = await client.get(CLINICALTRIALS_API, params=params)

            if response.status_code == 200:
                data = response.json()
                count = data.get("totalCount", 0)
            else:
                count = 0

            from datetime import timedelta
            _trend_cache[cache_key] = count
            _cache_expiry[cache_key] = datetime.now() + timedelta(hours=CACHE_DURATION_HOURS)
            return count

    except Exception as e:
        print(f"ClinicalTrials API error for {keyword}: {e}")
        return 0


async def get_future_research_mentions(keyword: str) -> int:
    """
    Count papers mentioning 'future research' or 'further investigation'
    related to the keyword (indicates research gaps/interest).
    """
    cache_key = f"future_{keyword}"
    if cache_key in _trend_cache:
        if datetime.now() < _cache_expiry.get(cache_key, datetime.min):
            return _trend_cache[cache_key]

    try:
        # Search for keyword + future research indicators
        future_terms = [
            f'"{keyword}" AND ("future research" OR "further investigation" OR "remains unclear" OR "warrants further study")',
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            total_count = 0
            for term in future_terms:
                params = {
                    "db": "pubmed",
                    "term": term,
                    "rettype": "count",
                    "datetype": "pdat",
                    "mindate": f"{CURRENT_YEAR-2}/01/01",
                    "maxdate": f"{CURRENT_YEAR}/12/31",
                }
                response = await client.get(f"{PUBMED_BASE}/esearch.fcgi", params=params)
                if response.status_code == 200:
                    match = re.search(r'<Count>(\d+)</Count>', response.text)
                    if match:
                        total_count += int(match.group(1))
                await asyncio.sleep(0.35)

            from datetime import timedelta
            _trend_cache[cache_key] = total_count
            _cache_expiry[cache_key] = datetime.now() + timedelta(hours=CACHE_DURATION_HOURS)
            return total_count

    except Exception as e:
        print(f"Future research search error for {keyword}: {e}")
        return 0


async def get_recent_citation_velocity(keyword: str) -> float:
    """
    Estimate citation velocity by comparing recent vs older paper counts.
    Higher ratio = more recent interest.
    """
    try:
        recent = await get_pubmed_count(keyword, CURRENT_YEAR)
        older = await get_pubmed_count(keyword, CURRENT_YEAR - 3)

        if older > 0:
            return (recent / older) * 100
        return 100.0 if recent > 0 else 0.0
    except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as e:
        return 0.0


# ============== Scoring Functions ==============

def calculate_rising_score(growth_rate: float) -> float:
    """Convert growth rate to 0-100 score."""
    if growth_rate >= 50:
        return 100.0
    elif growth_rate >= 30:
        return 80.0 + (growth_rate - 30) * 1.0
    elif growth_rate >= 15:
        return 60.0 + (growth_rate - 15) * 1.33
    elif growth_rate >= 5:
        return 40.0 + (growth_rate - 5) * 2.0
    elif growth_rate >= 0:
        return 20.0 + growth_rate * 4.0
    else:
        return max(0, 20.0 + growth_rate)


def calculate_interest_score(citation_velocity: float, total_papers: int) -> float:
    """Calculate interest score based on citation velocity and volume."""
    # Normalize citation velocity (100 = same as 3 years ago, 200 = doubled)
    velocity_score = min(100, citation_velocity / 2)

    # Volume bonus (more papers = more interest)
    if total_papers >= 10000:
        volume_bonus = 30
    elif total_papers >= 5000:
        volume_bonus = 20
    elif total_papers >= 1000:
        volume_bonus = 10
    else:
        volume_bonus = 0

    return min(100, velocity_score + volume_bonus)


def calculate_activity_score(current_papers: int, clinical_trials: int) -> float:
    """Calculate active research score."""
    # Paper volume score
    if current_papers >= 5000:
        paper_score = 50
    elif current_papers >= 1000:
        paper_score = 40
    elif current_papers >= 500:
        paper_score = 30
    elif current_papers >= 100:
        paper_score = 20
    else:
        paper_score = 10

    # Clinical trials bonus (indicates translational research)
    if clinical_trials >= 100:
        trial_score = 50
    elif clinical_trials >= 50:
        trial_score = 40
    elif clinical_trials >= 20:
        trial_score = 30
    elif clinical_trials >= 5:
        trial_score = 20
    else:
        trial_score = clinical_trials * 2

    return min(100, paper_score + trial_score)


def calculate_future_score(future_mentions: int, total_papers: int) -> float:
    """Calculate future research potential score."""
    if total_papers == 0:
        return 0

    # Ratio of papers mentioning future research needs
    ratio = (future_mentions / total_papers) * 100

    # Higher ratio = more research gaps = higher potential
    if ratio >= 10:
        return 100.0
    elif ratio >= 5:
        return 70.0 + ratio * 3
    elif ratio >= 2:
        return 40.0 + ratio * 6
    else:
        return ratio * 20


def get_trend_label(scores: MultiDimensionalScore) -> str:
    """Get descriptive trend label."""
    if scores.rising_score >= 80 and scores.total_score >= 70:
        return "🔥 Explosive"
    elif scores.rising_score >= 60:
        return "📈 Rising"
    elif scores.future_score >= 70 and scores.activity_score < 50:
        return "🌱 Emerging"
    elif scores.activity_score >= 70:
        return "⭐ Established"
    else:
        return "📊 Stable"


def get_research_stage(activity_score: float, rising_score: float, future_score: float) -> str:
    """Determine research stage."""
    if activity_score < 30 and rising_score >= 50:
        return "Early Stage"
    elif activity_score >= 30 and rising_score >= 30:
        return "Growth Phase"
    elif activity_score >= 60 and rising_score < 20:
        return "Mature Field"
    elif activity_score >= 50 and future_score >= 60:
        return "Active Innovation"
    else:
        return "Developing"


def generate_recommendation(keyword: str, scores: MultiDimensionalScore, clinical_trials: int) -> str:
    """Generate brief insight/recommendation."""
    insights = []

    if scores.rising_score >= 70:
        insights.append("Rapidly growing interest")
    if scores.future_score >= 60:
        insights.append("Many research gaps remain")
    if clinical_trials >= 20:
        insights.append(f"{clinical_trials} active clinical trials")
    if scores.activity_score >= 70 and scores.rising_score < 30:
        insights.append("Well-established field")
    if scores.interest_score >= 70:
        insights.append("High researcher attention")

    if not insights:
        insights.append("Moderate research activity")

    return " • ".join(insights[:2])


# ============== Domain Keywords ==============

# Enhanced keyword lists with emerging topics
HOT_TOPIC_DOMAINS = {
    "oncology": [
        "CAR-T therapy", "immune checkpoint inhibitor", "tumor microenvironment",
        "liquid biopsy", "targeted therapy", "immunotherapy resistance",
        "cancer vaccine", "oncolytic virus", "tumor organoid",
        "bispecific antibody", "antibody drug conjugate", "tumor infiltrating lymphocyte",
        "cancer metabolism", "ferroptosis cancer", "circulating tumor DNA"
    ],
    "neuroscience": [
        "Alzheimer biomarker", "neuroinflammation", "gut-brain axis",
        "alpha-synuclein", "tau protein aggregation", "microglia activation",
        "blood-brain barrier", "neurodegeneration therapy", "brain organoid",
        "neuroimmunology", "synaptic plasticity", "glymphatic system",
        "brain-computer interface", "optogenetics", "psychedelic therapy"
    ],
    "genomics": [
        "CRISPR", "single-cell RNA-seq", "spatial transcriptomics",
        "long-read sequencing", "epigenome editing", "base editing",
        "prime editing", "gene therapy", "mRNA therapeutics",
        "cell-free DNA", "multiomics integration", "genome-wide association",
        "polygenic risk score", "gene drive", "synthetic biology"
    ],
    "infectious_disease": [
        "mRNA vaccine", "monoclonal antibody therapy", "antiviral resistance",
        "pandemic preparedness", "viral evolution", "host-pathogen interaction",
        "broad-spectrum antiviral", "vaccine platform", "antimicrobial resistance",
        "bacteriophage therapy", "viral vector", "mucosal immunity",
        "long COVID", "zoonotic disease", "one health"
    ],
    "ai_medicine": [
        "machine learning diagnosis", "AI drug discovery", "deep learning pathology",
        "clinical decision support", "federated learning healthcare",
        "AlphaFold protein", "foundation model medicine", "multimodal AI health",
        "large language model medicine", "generative AI healthcare",
        "computer vision radiology", "natural language processing EHR",
        "digital twin healthcare", "AI clinical trial", "explainable AI medicine"
    ]
}


# ============== API Endpoints ==============

async def get_yearly_counts(keyword: str, start_year: int = None, end_year: int = None) -> List[YearlyCount]:
    """Get publication counts for each year."""
    if start_year is None:
        start_year = DEFAULT_START_YEAR
    if end_year is None:
        end_year = DEFAULT_END_YEAR

    years = list(range(start_year, end_year + 1))
    counts = []

    for year in years:
        count = await get_pubmed_count(keyword, year)
        counts.append(count)
        await asyncio.sleep(0.4)

    yearly_counts = []
    for i, (year, count) in enumerate(zip(years, counts)):
        growth_rate = None
        if i > 0 and counts[i-1] > 0:
            growth_rate = ((count - counts[i-1]) / counts[i-1]) * 100

        yearly_counts.append(YearlyCount(
            year=year,
            count=count,
            growth_rate=round(growth_rate, 1) if growth_rate is not None else None
        ))

    return yearly_counts


def calculate_trend_direction(yearly_counts: List[YearlyCount]) -> str:
    """Determine trend direction."""
    if len(yearly_counts) < 2:
        return "stable"

    mid = len(yearly_counts) // 2
    first_half_avg = sum(yc.count for yc in yearly_counts[:mid]) / mid if mid > 0 else 0
    second_half_avg = sum(yc.count for yc in yearly_counts[mid:]) / (len(yearly_counts) - mid)

    if second_half_avg > first_half_avg * 1.2:
        return "rising"
    elif second_half_avg < first_half_avg * 0.8:
        return "declining"
    return "stable"


def calculate_5yr_growth(yearly_counts: List[YearlyCount]) -> Optional[float]:
    """Calculate 5-year growth rate."""
    if len(yearly_counts) < 2:
        return None

    first_count = yearly_counts[0].count
    last_count = yearly_counts[-1].count

    if first_count == 0:
        return None

    return round(((last_count - first_count) / first_count) * 100, 1)


@router.get("/keyword", response_model=TrendAnalysisResponse)
async def analyze_keyword_trend(
    keywords: str = Query(..., description="Comma-separated keywords"),
    start_year: int = Query(default=None, ge=2000, le=CURRENT_YEAR),
    end_year: int = Query(default=None, ge=2000, le=CURRENT_YEAR)
):
    """Analyze publication trends for keywords over time."""
    if start_year is None:
        start_year = DEFAULT_START_YEAR
    if end_year is None:
        end_year = DEFAULT_END_YEAR

    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    if not keyword_list:
        raise HTTPException(status_code=400, detail="At least one keyword required")
    if len(keyword_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 keywords allowed")

    trends = []
    years = list(range(start_year, end_year + 1))

    for keyword in keyword_list:
        yearly_counts = await get_yearly_counts(keyword, start_year, end_year)
        total_count = sum(yc.count for yc in yearly_counts)

        peak_year = None
        max_count = 0
        for yc in yearly_counts:
            if yc.count > max_count:
                max_count = yc.count
                peak_year = yc.year

        trend = KeywordTrend(
            keyword=keyword,
            total_count=total_count,
            yearly_counts=yearly_counts,
            trend_direction=calculate_trend_direction(yearly_counts),
            growth_5yr=calculate_5yr_growth(yearly_counts),
            peak_year=peak_year
        )
        trends.append(trend)

    trends.sort(key=lambda t: t.total_count, reverse=True)

    return TrendAnalysisResponse(
        query=keywords,
        years=years,
        trends=trends,
        comparison_keywords=keyword_list,
        analysis_date=datetime.now().strftime("%Y-%m-%d")
    )


@router.get("/compare", response_model=TrendAnalysisResponse)
async def compare_keywords(
    keyword1: str = Query(...),
    keyword2: str = Query(...),
    keyword3: Optional[str] = Query(None),
    start_year: int = Query(default=None, ge=2000, le=CURRENT_YEAR),
    end_year: int = Query(default=None, ge=2000, le=CURRENT_YEAR)
):
    """Compare publication trends between keywords."""
    keywords = [keyword1, keyword2]
    if keyword3:
        keywords.append(keyword3)

    return await analyze_keyword_trend(
        keywords=",".join(keywords),
        start_year=start_year,
        end_year=end_year
    )


@router.get("/hot-topics/{domain}", response_model=HotTopicsResponse)
async def get_hot_topics(
    domain: str,
    limit: int = Query(10, ge=1, le=20)
):
    """
    Get hot topics (legacy endpoint for backward compatibility).
    Use /hot-topics-enhanced/{domain} for multi-dimensional analysis.
    """
    if domain not in HOT_TOPIC_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain. Available: {list(HOT_TOPIC_DOMAINS.keys())}"
        )

    keywords = HOT_TOPIC_DOMAINS[domain][:10]  # Limit for speed
    current_year = datetime.now().year

    hot_topics = []

    for keyword in keywords:
        current_count = await get_pubmed_count(keyword, current_year)
        previous_count = await get_pubmed_count(keyword, current_year - 1)

        if previous_count > 0:
            growth_rate = ((current_count - previous_count) / previous_count) * 100
        else:
            growth_rate = 100.0 if current_count > 0 else 0.0

        hot_topics.append(HotTopic(
            keyword=keyword,
            recent_count=current_count,
            previous_count=previous_count,
            growth_rate=round(growth_rate, 1),
            sample_titles=[]
        ))

    hot_topics.sort(key=lambda t: t.growth_rate, reverse=True)

    return HotTopicsResponse(
        domain=domain,
        hot_topics=hot_topics[:limit],
        analysis_period=f"{current_year-1} vs {current_year}"
    )


@router.get("/hot-topics-enhanced/{domain}", response_model=EnhancedHotTopicsResponse)
async def get_enhanced_hot_topics(
    domain: str,
    limit: int = Query(10, ge=1, le=15)
):
    """
    Get hot topics with multi-dimensional analysis.

    Dimensions:
    - Rising Score: Year-over-year growth rate
    - Interest Score: Citation velocity and search trends
    - Activity Score: Publication volume + clinical trials
    - Future Score: Research gap indicators

    Each dimension is scored 0-100, with weighted total score.
    """
    if domain not in HOT_TOPIC_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain. Available: {list(HOT_TOPIC_DOMAINS.keys())}"
        )

    keywords = HOT_TOPIC_DOMAINS[domain]
    current_year = datetime.now().year

    enhanced_topics = []

    for keyword in keywords:
        # Gather all metrics
        current_papers = await get_pubmed_count(keyword, current_year)
        previous_papers = await get_pubmed_count(keyword, current_year - 1)
        clinical_trials = await get_clinical_trials_count(keyword)
        future_mentions = await get_future_research_mentions(keyword)
        citation_velocity = await get_recent_citation_velocity(keyword)

        # Calculate growth rate
        if previous_papers > 0:
            growth_rate = ((current_papers - previous_papers) / previous_papers) * 100
        else:
            growth_rate = 100.0 if current_papers > 0 else 0.0

        total_papers = current_papers + previous_papers

        # Calculate dimension scores
        rising_score = calculate_rising_score(growth_rate)
        interest_score = calculate_interest_score(citation_velocity, total_papers)
        activity_score = calculate_activity_score(current_papers, clinical_trials)
        future_score = calculate_future_score(future_mentions, total_papers)

        # Weighted total score
        # Weights: Rising=30%, Interest=25%, Activity=25%, Future=20%
        total_score = (
            rising_score * 0.30 +
            interest_score * 0.25 +
            activity_score * 0.25 +
            future_score * 0.20
        )

        scores = MultiDimensionalScore(
            rising_score=round(rising_score, 1),
            interest_score=round(interest_score, 1),
            activity_score=round(activity_score, 1),
            future_score=round(future_score, 1),
            total_score=round(total_score, 1)
        )

        enhanced_topics.append(EnhancedHotTopic(
            keyword=keyword,
            scores=scores,
            current_year_papers=current_papers,
            previous_year_papers=previous_papers,
            growth_rate=round(growth_rate, 1),
            clinical_trials=clinical_trials,
            future_mentions=future_mentions,
            trend_label=get_trend_label(scores),
            research_stage=get_research_stage(activity_score, rising_score, future_score),
            recommendation=generate_recommendation(keyword, scores, clinical_trials)
        ))

        # Rate limiting
        await asyncio.sleep(0.3)

    # Sort by total score (highest first)
    enhanced_topics.sort(key=lambda t: t.scores.total_score, reverse=True)

    return EnhancedHotTopicsResponse(
        domain=domain,
        hot_topics=enhanced_topics[:limit],
        analysis_period=f"{current_year-1} vs {current_year}",
        methodology="Multi-dimensional analysis: Rising (30%) + Interest (25%) + Activity (25%) + Future Potential (20%)",
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M")
    )


@router.get("/domains")
async def list_domains():
    """List available domains for hot topic analysis."""
    return {
        "domains": [
            {"key": "oncology", "name": "Oncology", "name_ko": "종양학", "keyword_count": len(HOT_TOPIC_DOMAINS["oncology"])},
            {"key": "neuroscience", "name": "Neuroscience", "name_ko": "신경과학", "keyword_count": len(HOT_TOPIC_DOMAINS["neuroscience"])},
            {"key": "genomics", "name": "Genomics & Gene Therapy", "name_ko": "유전체학", "keyword_count": len(HOT_TOPIC_DOMAINS["genomics"])},
            {"key": "infectious_disease", "name": "Infectious Disease", "name_ko": "감염병", "keyword_count": len(HOT_TOPIC_DOMAINS["infectious_disease"])},
            {"key": "ai_medicine", "name": "AI in Medicine", "name_ko": "AI 의학", "keyword_count": len(HOT_TOPIC_DOMAINS["ai_medicine"])},
        ]
    }


class MeSHTrendRequest(BaseModel):
    """MeSH term trend request."""
    mesh_terms: List[str]
    start_year: Optional[int] = None
    end_year: Optional[int] = None


@router.post("/mesh-trends")
async def analyze_mesh_trends(request: MeSHTrendRequest):
    """Analyze trends for specific MeSH terms."""
    start_year = request.start_year if request.start_year else DEFAULT_START_YEAR
    end_year = request.end_year if request.end_year else DEFAULT_END_YEAR

    trends = []

    for mesh_term in request.mesh_terms[:5]:
        search_term = f'"{mesh_term}"[MeSH Terms]'
        yearly_counts = await get_yearly_counts(search_term, start_year, end_year)

        trends.append({
            "mesh_term": mesh_term,
            "yearly_counts": [{"year": yc.year, "count": yc.count} for yc in yearly_counts],
            "total": sum(yc.count for yc in yearly_counts),
            "trend": calculate_trend_direction(yearly_counts)
        })

    return {
        "mesh_terms": request.mesh_terms,
        "trends": trends,
        "analysis_date": datetime.now().strftime("%Y-%m-%d")
    }


@router.get("/emerging")
async def find_emerging_topics(
    base_keyword: str = Query(..., description="Base research area"),
    limit: int = Query(10, ge=1, le=20)
):
    """Find emerging sub-topics within a research area."""
    modifiers = [
        "biomarker", "therapy", "immunotherapy", "targeted therapy",
        "early detection", "prognosis", "resistance", "metastasis",
        "microenvironment", "organoid", "liquid biopsy", "AI diagnosis",
        "single cell", "spatial", "multiomics"
    ]

    current_year = datetime.now().year
    emerging = []

    for modifier in modifiers:
        combined_term = f"{base_keyword} {modifier}"

        current_count = await get_pubmed_count(combined_term, current_year)
        prev_count = await get_pubmed_count(combined_term, current_year - 2)

        if prev_count > 0:
            growth = ((current_count - prev_count) / prev_count) * 100
        else:
            growth = 100.0 if current_count > 10 else 0.0

        if current_count >= 5:
            emerging.append({
                "topic": combined_term,
                "modifier": modifier,
                "current_year_count": current_count,
                "two_years_ago_count": prev_count,
                "growth_rate": round(growth, 1)
            })

    emerging.sort(key=lambda x: x["growth_rate"], reverse=True)

    return {
        "base_keyword": base_keyword,
        "emerging_topics": emerging[:limit],
        "analysis_period": f"{current_year-2} to {current_year}"
    }


# ============================================================================
# Validated Trend Endpoints
# ============================================================================

from ...core.trend_validator import (
    TrendValidator,
    ValidatedTrend,
    ConfidenceLevel,
    validate_trend,
    get_validated_hot_topics,
)


class ValidatedTrendResponse(BaseModel):
    """Validated trend with evidence."""
    keyword: str

    # Scores
    publication_score: float
    diversity_score: float
    review_score: float
    clinical_score: float
    gap_score: float
    total_score: float

    # Confidence
    confidence_level: str
    confidence_emoji: str

    # Evidence
    summary: str
    evidence_summary: List[str]

    # Raw metrics
    total_papers_5yr: int
    growth_rate_5yr: float
    growth_rate_yoy: float
    unique_journals: int
    high_if_journals: int
    systematic_reviews: int
    meta_analyses: int
    active_clinical_trials: int
    future_research_mentions: int

    # Metadata
    validated_at: str
    data_period: str


class ValidatedTrendsResponse(BaseModel):
    """Multiple validated trends response."""
    trends: List[ValidatedTrendResponse]
    total_validated: int
    methodology: str
    last_updated: str


def _trend_to_response(trend: ValidatedTrend) -> ValidatedTrendResponse:
    """Convert ValidatedTrend to API response."""
    confidence_emojis = {
        ConfidenceLevel.HIGH: "🟢",
        ConfidenceLevel.MEDIUM: "🟡",
        ConfidenceLevel.EMERGING: "🟠",
        ConfidenceLevel.UNCERTAIN: "🔴",
    }

    return ValidatedTrendResponse(
        keyword=trend.keyword,
        publication_score=trend.publication_score,
        diversity_score=trend.diversity_score,
        review_score=trend.review_score,
        clinical_score=trend.clinical_score,
        gap_score=trend.gap_score,
        total_score=trend.total_score,
        confidence_level=trend.confidence_level.value,
        confidence_emoji=confidence_emojis.get(trend.confidence_level, "⚪"),
        summary=trend.summary,
        evidence_summary=trend.evidence_summary,
        total_papers_5yr=trend.sparse_signals.total_papers_5yr if trend.sparse_signals else 0,
        growth_rate_5yr=trend.sparse_signals.growth_rate_5yr if trend.sparse_signals else 0,
        growth_rate_yoy=trend.sparse_signals.growth_rate_yoy if trend.sparse_signals else 0,
        unique_journals=trend.sparse_signals.unique_journals if trend.sparse_signals else 0,
        high_if_journals=trend.sparse_signals.high_if_journals if trend.sparse_signals else 0,
        systematic_reviews=trend.validation_evidence.systematic_reviews if trend.validation_evidence else 0,
        meta_analyses=trend.validation_evidence.meta_analyses if trend.validation_evidence else 0,
        active_clinical_trials=trend.validation_evidence.active_clinical_trials if trend.validation_evidence else 0,
        future_research_mentions=trend.validation_evidence.future_research_mentions if trend.validation_evidence else 0,
        validated_at=trend.validated_at,
        data_period=trend.data_period,
    )


@router.get("/validate/{keyword}", response_model=ValidatedTrendResponse)
async def validate_keyword_trend(keyword: str):
    """
    Validate a single keyword as a research trend.

    Returns comprehensive validation with:
    - Multi-dimensional scores (publication, diversity, review, clinical, gap)
    - Confidence level (high/medium/emerging/uncertain)
    - Evidence summary with citations
    - Raw metrics for transparency

    This endpoint provides explainable, defensible trend validation
    for user trust.
    """
    try:
        trend = await validate_trend(keyword)
        return _trend_to_response(trend)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.get("/validated-defaults", response_model=ValidatedTrendsResponse)
async def get_validated_default_keywords():
    """
    Get validated default keywords for TrendAnalysis.

    Returns top 5 keywords that have passed validation criteria:
    - Minimum total score of 40
    - Evidence from multiple sources
    - Statistical significance

    These keywords are suitable for display as default trends.
    """
    validator = TrendValidator()
    try:
        # Candidate pool
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
            'large language model medicine',
        ]

        validated = await validator.validate_keywords(candidates, min_score=40.0)
        top_5 = validated[:5]

        return ValidatedTrendsResponse(
            trends=[_trend_to_response(t) for t in top_5],
            total_validated=len(top_5),
            methodology=(
                "Multi-signal validation: Publication Growth (25%) + "
                "Journal Diversity (20%) + Review Coverage (20%) + "
                "Clinical Activity (20%) + Research Gap (15%)"
            ),
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    finally:
        await validator.close()


@router.get("/validated-domain/{domain}", response_model=ValidatedTrendsResponse)
async def get_validated_domain_trends(
    domain: str,
    limit: int = Query(10, ge=1, le=15),
    min_score: float = Query(30.0, ge=0, le=100)
):
    """
    Get validated hot topics for a specific domain.

    Domains: oncology, neuroscience, genomics, infectious_disease, ai_medicine

    Each topic is validated with multi-dimensional scoring:
    - Publication growth & volume
    - Journal diversity (including high-IF journals)
    - Review coverage (systematic reviews, meta-analyses)
    - Clinical trial activity
    - Research gap signals

    Only topics meeting min_score threshold are returned.
    """
    if domain not in HOT_TOPIC_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain. Available: {list(HOT_TOPIC_DOMAINS.keys())}"
        )

    validator = TrendValidator()
    try:
        candidates = HOT_TOPIC_DOMAINS[domain]
        validated = await validator.validate_keywords(candidates, min_score=min_score)

        return ValidatedTrendsResponse(
            trends=[_trend_to_response(t) for t in validated[:limit]],
            total_validated=len(validated[:limit]),
            methodology=(
                "Multi-signal validation: Publication Growth (25%) + "
                "Journal Diversity (20%) + Review Coverage (20%) + "
                "Clinical Activity (20%) + Research Gap (15%)"
            ),
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    finally:
        await validator.close()
