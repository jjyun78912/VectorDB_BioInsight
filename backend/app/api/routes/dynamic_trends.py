"""
Dynamic Trend Discovery API - Real-time Hot Topics from Latest Papers

Flow:
1. Fetch latest papers from PubMed (last 30-90 days)
2. Extract keywords from titles/abstracts (NLP)
3. Analyze keyword frequency and growth
4. Return dynamically discovered hot topics

This replaces hardcoded keyword lists with data-driven discovery.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Tuple
import httpx
import asyncio
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict

router = APIRouter()

# API endpoints
PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Cache
_dynamic_cache: Dict[str, any] = {}
_cache_expiry: Dict[str, datetime] = {}
CACHE_HOURS = 6  # Shorter cache for dynamic data


# ============== Models ==============

class DiscoveredKeyword(BaseModel):
    """A dynamically discovered trending keyword."""
    keyword: str
    frequency: int              # How many times it appears in recent papers
    paper_count: int            # Number of papers containing this keyword
    growth_signal: str          # "new", "rising", "stable"
    sample_titles: List[str]    # Example paper titles
    first_seen_date: Optional[str] = None


class DynamicTrendResponse(BaseModel):
    """Response for dynamic trend discovery."""
    domain: str
    time_window: str            # e.g., "last 30 days"
    total_papers_analyzed: int
    discovered_keywords: List[DiscoveredKeyword]
    methodology: str
    analysis_date: str


class KeywordGrowth(BaseModel):
    """Keyword with growth analysis."""
    keyword: str
    recent_count: int           # Last 30 days
    previous_count: int         # 30-60 days ago
    growth_rate: float
    is_emerging: bool           # New or rapidly growing


class GrowthAnalysisResponse(BaseModel):
    """Response for keyword growth analysis."""
    domain: str
    emerging_keywords: List[KeywordGrowth]
    declining_keywords: List[KeywordGrowth]
    stable_keywords: List[KeywordGrowth]
    analysis_period: str


# ============== Keyword Extraction ==============

# Biomedical stopwords (common words to filter out)
BIO_STOPWORDS = {
    # English common words
    "the", "and", "for", "with", "this", "that", "from", "were", "was", "are",
    "been", "have", "has", "had", "will", "would", "could", "should", "may",
    "can", "not", "but", "all", "some", "any", "each", "more", "most", "other",
    "than", "then", "only", "also", "such", "both", "into", "over", "after",
    "before", "between", "through", "during", "under", "about", "however",
    "these", "those", "their", "there", "here", "where", "when", "which",
    "while", "being", "been", "because", "very", "well", "first", "second",

    # Numbers and codes
    "x2009", "x2013", "x2014", "x2019", "nbsp",

    # General research
    "study", "studies", "analysis", "results", "result", "effect", "effects", "role",
    "using", "based", "novel", "new", "recent", "review", "case", "report", "reports",
    "patients", "patient", "treatment", "treatments", "clinical", "data", "group",
    "method", "methods", "approach", "research", "findings", "outcomes", "outcome",
    "associated", "association", "relationship", "impact", "evidence", "showed",
    "model", "models", "development", "evaluation", "assessment", "compared",
    "total", "mean", "average", "median", "standard", "significantly",

    # Common medical
    "disease", "diseases", "disorder", "disorders", "syndrome", "condition",
    "therapy", "therapies", "diagnosis", "prognosis", "risk", "factor",
    "factors", "mechanism", "mechanisms", "pathway", "pathways", "showed",

    # Generic
    "high", "low", "increased", "decreased", "human", "animal", "cell",
    "cells", "tissue", "tissues", "level", "levels", "expression", "higher",
    "activity", "function", "potential", "significant", "important", "lower",
    "conclusion", "conclusions", "background", "objective", "objectives",
    "purpose", "aim", "aims", "introduction", "abstract", "keywords",
}

# Important biomedical terms to prioritize
PRIORITY_TERMS = {
    # Technologies
    "crispr", "car-t", "mrna", "single-cell", "spatial", "organoid",
    "multiomics", "proteomics", "metabolomics", "epigenomics",

    # AI/ML
    "machine learning", "deep learning", "artificial intelligence",
    "neural network", "transformer", "llm", "gpt", "foundation model",

    # Hot topics
    "immunotherapy", "checkpoint", "microbiome", "gut-brain",
    "liquid biopsy", "ctdna", "exosome", "nanoparticle",
    "gene editing", "base editing", "prime editing",
}


def extract_keywords_from_text(text: str, min_length: int = 3) -> List[str]:
    """Extract meaningful keywords from text using simple NLP."""
    if not text:
        return []

    # Lowercase and clean
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)

    # Extract n-grams (1-3 words)
    words = text.split()
    keywords = []

    # Unigrams
    for word in words:
        if len(word) >= min_length and word not in BIO_STOPWORDS:
            keywords.append(word)

    # Bigrams
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if words[i] not in BIO_STOPWORDS or words[i+1] not in BIO_STOPWORDS:
            if len(bigram) >= 6:
                keywords.append(bigram)

    # Trigrams (for specific terms like "single cell rna")
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
        if any(term in trigram for term in PRIORITY_TERMS):
            keywords.append(trigram)

    return keywords


def filter_meaningful_keywords(keyword_counts: Counter, min_count: int = 3) -> Dict[str, int]:
    """Filter and rank keywords by importance."""
    filtered = {}

    # Common bigram stopwords
    BIGRAM_STOPWORDS = {
        "in the", "of the", "to the", "on the", "for the", "at the",
        "and the", "is the", "was the", "are the", "be the", "by the",
        "as the", "it is", "we have", "we found", "our study", "this study",
        "in this", "of this", "to this", "in our", "of our",
    }

    for keyword, count in keyword_counts.items():
        if count < min_count:
            continue

        # Skip pure numbers
        if keyword.isdigit():
            continue

        # Skip very short keywords
        if len(keyword) < 4:
            continue

        # Skip generic terms (single words)
        if keyword in BIO_STOPWORDS:
            continue

        # Skip bigram stopwords
        if keyword in BIGRAM_STOPWORDS:
            continue

        # Skip if starts/ends with stopword (bigrams/trigrams)
        words = keyword.split()
        if len(words) >= 2:
            if words[0] in BIO_STOPWORDS or words[-1] in BIO_STOPWORDS:
                # But keep priority terms
                if not any(term in keyword for term in PRIORITY_TERMS):
                    continue

        # Boost priority terms
        score = count
        if any(term in keyword for term in PRIORITY_TERMS):
            score *= 2

        filtered[keyword] = score

    return filtered


# ============== PubMed Fetching ==============

async def fetch_recent_pubmed_papers(
    query: str,
    days: int = 30,
    max_results: int = 200
) -> List[Dict]:
    """Fetch recent papers from PubMed."""
    cache_key = f"pubmed_recent_{query}_{days}"
    if cache_key in _dynamic_cache:
        if datetime.now() < _cache_expiry.get(cache_key, datetime.min):
            return _dynamic_cache[cache_key]

    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        async with httpx.AsyncClient(timeout=60.0) as client:
            # URL encode the query properly
            import urllib.parse
            encoded_query = urllib.parse.quote(query)

            # Search for papers - use URL string directly for proper encoding
            search_url = f"{PUBMED_BASE}/esearch.fcgi?db=pubmed&term={encoded_query}&retmax={max_results}&retmode=json&datetype=pdat&mindate={start_date.strftime('%Y/%m/%d')}&maxdate={end_date.strftime('%Y/%m/%d')}&sort=date"

            search_response = await client.get(search_url)

            if search_response.status_code != 200:
                return []

            search_data = search_response.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # Fetch paper details
            await asyncio.sleep(0.5)

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list[:100]),  # Limit to 100 for speed
                "retmode": "xml",
                "rettype": "abstract"
            }

            fetch_response = await client.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params=fetch_params
            )

            if fetch_response.status_code != 200:
                return []

            # Parse XML response
            papers = parse_pubmed_xml(fetch_response.text)

            _dynamic_cache[cache_key] = papers
            _cache_expiry[cache_key] = datetime.now() + timedelta(hours=CACHE_HOURS)

            return papers

    except Exception as e:
        print(f"PubMed fetch error: {e}")
        return []


def parse_pubmed_xml(xml_text: str) -> List[Dict]:
    """Parse PubMed XML response to extract paper info."""
    papers = []

    # Simple regex-based parsing (faster than full XML parsing)
    article_pattern = r'<PubmedArticle>(.*?)</PubmedArticle>'
    title_pattern = r'<ArticleTitle>(.*?)</ArticleTitle>'
    abstract_pattern = r'<AbstractText[^>]*>(.*?)</AbstractText>'
    pmid_pattern = r'<PMID[^>]*>(\d+)</PMID>'
    year_pattern = r'<PubDate>.*?<Year>(\d{4})</Year>'

    for match in re.finditer(article_pattern, xml_text, re.DOTALL):
        article_xml = match.group(1)

        title_match = re.search(title_pattern, article_xml, re.DOTALL)
        abstract_matches = re.findall(abstract_pattern, article_xml, re.DOTALL)
        pmid_match = re.search(pmid_pattern, article_xml)
        year_match = re.search(year_pattern, article_xml, re.DOTALL)

        if title_match:
            # Clean HTML tags
            title = re.sub(r'<[^>]+>', '', title_match.group(1))
            abstract = ' '.join(re.sub(r'<[^>]+>', '', a) for a in abstract_matches)

            papers.append({
                "pmid": pmid_match.group(1) if pmid_match else None,
                "title": title,
                "abstract": abstract,
                "year": int(year_match.group(1)) if year_match else None
            })

    return papers


# ============== Domain Queries ==============

DOMAIN_QUERIES = {
    "oncology": "(cancer OR tumor OR oncology OR carcinoma) AND (therapy OR treatment OR immunotherapy)",
    "neuroscience": "(neuroscience OR brain OR neurodegenerative OR alzheimer OR parkinson)",
    "genomics": "(genomics OR CRISPR OR gene editing OR sequencing OR transcriptomics)",
    "infectious_disease": "(infectious disease OR vaccine OR virus OR pathogen OR antimicrobial)",
    "ai_medicine": "(artificial intelligence OR machine learning OR deep learning) AND (medicine OR clinical OR diagnosis)",
    "immunology": "(immunology OR immune OR T cell OR antibody OR immunotherapy)",
    "drug_discovery": "(drug discovery OR drug development OR pharmaceutical OR compound screening)",
}


# ============== API Endpoints ==============

@router.get("/discover/{domain}", response_model=DynamicTrendResponse)
async def discover_trending_keywords(
    domain: str,
    days: int = Query(30, ge=7, le=90, description="Time window in days"),
    min_frequency: int = Query(3, ge=2, le=10),
    limit: int = Query(20, ge=10, le=50)
):
    """
    Dynamically discover trending keywords from recent papers.

    This analyzes the latest papers in a domain to find
    what researchers are actually writing about.
    """
    if domain not in DOMAIN_QUERIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain. Available: {list(DOMAIN_QUERIES.keys())}"
        )

    query = DOMAIN_QUERIES[domain]

    # Fetch recent papers
    papers = await fetch_recent_pubmed_papers(query, days=days, max_results=200)

    if not papers:
        return DynamicTrendResponse(
            domain=domain,
            time_window=f"last {days} days",
            total_papers_analyzed=0,
            discovered_keywords=[],
            methodology="No papers found in time window",
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    # Extract keywords from all papers
    all_keywords = []
    keyword_to_titles = defaultdict(list)

    for paper in papers:
        text = f"{paper['title']} {paper.get('abstract', '')}"
        keywords = extract_keywords_from_text(text)

        for kw in keywords:
            all_keywords.append(kw)
            if len(keyword_to_titles[kw]) < 3:  # Keep max 3 sample titles
                keyword_to_titles[kw].append(paper['title'][:100])

    # Count and filter keywords
    keyword_counts = Counter(all_keywords)
    filtered_keywords = filter_meaningful_keywords(keyword_counts, min_count=min_frequency)

    # Sort by score
    sorted_keywords = sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)

    # Build response
    discovered = []
    for keyword, score in sorted_keywords[:limit]:
        count = keyword_counts[keyword]
        paper_count = len(keyword_to_titles[keyword])

        # Determine growth signal (simplified - would need historical data for real analysis)
        if any(term in keyword for term in PRIORITY_TERMS):
            growth_signal = "rising"
        elif count >= 10:
            growth_signal = "stable"
        else:
            growth_signal = "new"

        discovered.append(DiscoveredKeyword(
            keyword=keyword,
            frequency=count,
            paper_count=paper_count,
            growth_signal=growth_signal,
            sample_titles=keyword_to_titles[keyword][:3]
        ))

    return DynamicTrendResponse(
        domain=domain,
        time_window=f"last {days} days",
        total_papers_analyzed=len(papers),
        discovered_keywords=discovered,
        methodology="NLP keyword extraction from recent PubMed papers, ranked by frequency and biomedical relevance",
        analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )


@router.get("/growth-analysis/{domain}", response_model=GrowthAnalysisResponse)
async def analyze_keyword_growth(
    domain: str,
    limit: int = Query(15, ge=5, le=30)
):
    """
    Compare keywords between two time periods to find
    emerging and declining trends.
    """
    if domain not in DOMAIN_QUERIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain. Available: {list(DOMAIN_QUERIES.keys())}"
        )

    query = DOMAIN_QUERIES[domain]

    # Fetch recent papers (last 30 days)
    recent_papers = await fetch_recent_pubmed_papers(query, days=30, max_results=150)
    await asyncio.sleep(1)  # Rate limiting

    # Fetch previous period papers (30-60 days ago)
    # For this, we need to adjust the query
    previous_papers = await fetch_previous_period_papers(query, days_ago=30, window=30, max_results=150)

    # Extract keywords from both periods
    recent_keywords = Counter()
    for paper in recent_papers:
        text = f"{paper['title']} {paper.get('abstract', '')}"
        for kw in extract_keywords_from_text(text):
            recent_keywords[kw] += 1

    previous_keywords = Counter()
    for paper in previous_papers:
        text = f"{paper['title']} {paper.get('abstract', '')}"
        for kw in extract_keywords_from_text(text):
            previous_keywords[kw] += 1

    # Analyze growth
    all_keywords = set(recent_keywords.keys()) | set(previous_keywords.keys())

    emerging = []
    declining = []
    stable = []

    for kw in all_keywords:
        if kw in BIO_STOPWORDS or len(kw) < 4:
            continue

        recent_count = recent_keywords.get(kw, 0)
        previous_count = previous_keywords.get(kw, 0)

        # Skip low frequency keywords
        if recent_count < 2 and previous_count < 2:
            continue

        # Calculate growth rate
        if previous_count > 0:
            growth_rate = ((recent_count - previous_count) / previous_count) * 100
        else:
            growth_rate = 100.0 if recent_count > 0 else 0.0

        is_emerging = (recent_count > 0 and previous_count == 0) or growth_rate > 50

        keyword_growth = KeywordGrowth(
            keyword=kw,
            recent_count=recent_count,
            previous_count=previous_count,
            growth_rate=round(growth_rate, 1),
            is_emerging=is_emerging
        )

        if growth_rate > 30:
            emerging.append(keyword_growth)
        elif growth_rate < -30:
            declining.append(keyword_growth)
        else:
            stable.append(keyword_growth)

    # Sort
    emerging.sort(key=lambda x: x.growth_rate, reverse=True)
    declining.sort(key=lambda x: x.growth_rate)
    stable.sort(key=lambda x: x.recent_count, reverse=True)

    return GrowthAnalysisResponse(
        domain=domain,
        emerging_keywords=emerging[:limit],
        declining_keywords=declining[:limit],
        stable_keywords=stable[:limit],
        analysis_period="Last 30 days vs Previous 30 days"
    )


async def fetch_previous_period_papers(
    query: str,
    days_ago: int = 30,
    window: int = 30,
    max_results: int = 150
) -> List[Dict]:
    """Fetch papers from a previous time period."""
    try:
        end_date = datetime.now() - timedelta(days=days_ago)
        start_date = end_date - timedelta(days=window)

        async with httpx.AsyncClient(timeout=60.0) as client:
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "datetype": "pdat",
                "mindate": start_date.strftime("%Y/%m/%d"),
                "maxdate": end_date.strftime("%Y/%m/%d"),
                "sort": "date"
            }

            search_response = await client.get(
                f"{PUBMED_BASE}/esearch.fcgi",
                params=search_params
            )

            if search_response.status_code != 200:
                return []

            search_data = search_response.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            await asyncio.sleep(0.5)

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list[:100]),
                "retmode": "xml",
                "rettype": "abstract"
            }

            fetch_response = await client.get(
                f"{PUBMED_BASE}/efetch.fcgi",
                params=fetch_params
            )

            if fetch_response.status_code != 200:
                return []

            return parse_pubmed_xml(fetch_response.text)

    except Exception as e:
        print(f"PubMed previous period fetch error: {e}")
        return []


@router.get("/domains")
async def list_available_domains():
    """List available domains for dynamic trend discovery."""
    return {
        "domains": [
            {"key": k, "query_preview": v[:50] + "..."}
            for k, v in DOMAIN_QUERIES.items()
        ]
    }
