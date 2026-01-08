"""
Research Gap Analysis API - Discover Unexplored Research Areas

Features:
- Identify understudied topics within a research domain
- Find "future research needed" patterns in recent papers
- Detect emerging questions without answers
- Suggest potential research directions

Methodology:
1. Analyze "future research", "remains unclear", "further investigation needed" patterns
2. Compare publication volume vs gap mentions ratio
3. Identify topic combinations with low coverage
4. Extract unanswered questions from abstracts
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
import asyncio
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict

router = APIRouter()

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Cache
_gap_cache: Dict[str, any] = {}
_cache_expiry: Dict[str, datetime] = {}
CACHE_HOURS = 12


# ============== Models ==============

class ResearchGap(BaseModel):
    """An identified research gap."""
    topic: str
    gap_type: str                    # "understudied", "unclear_mechanism", "lacking_data", "emerging_question"
    gap_description: str
    evidence_count: int              # Papers mentioning this gap
    total_papers: int                # Total papers in topic
    gap_ratio: float                 # evidence_count / total_papers
    sample_contexts: List[str]       # Example sentences mentioning the gap
    suggested_questions: List[str]   # Potential research questions
    priority_score: float            # 0-100 based on gap ratio and recency


class GapAnalysisResponse(BaseModel):
    """Response for research gap analysis."""
    domain: str
    total_gaps_found: int
    research_gaps: List[ResearchGap]
    methodology: str
    analysis_date: str


class UnderstudiedArea(BaseModel):
    """An understudied research area."""
    topic_combination: str           # e.g., "CRISPR + rare disease"
    base_topic_papers: int
    combined_topic_papers: int
    coverage_ratio: float            # combined / base
    opportunity_score: float         # Lower coverage = higher opportunity


class UnderstudiedResponse(BaseModel):
    """Response for understudied areas."""
    base_topic: str
    understudied_areas: List[UnderstudiedArea]
    methodology: str
    analysis_date: str


class EmergingQuestion(BaseModel):
    """An emerging research question."""
    question: str
    source_paper: str
    context: str
    question_type: str               # "mechanism", "therapeutic", "diagnostic", "epidemiological"
    relevance_score: float


class EmergingQuestionsResponse(BaseModel):
    """Response for emerging questions."""
    domain: str
    questions: List[EmergingQuestion]
    analysis_date: str


# ============== Gap Detection Patterns ==============

# Patterns indicating research gaps
GAP_PATTERNS = {
    "future_research": [
        r"future (?:research|studies|investigations?) (?:is|are|should|will|need)",
        r"(?:further|more) (?:research|studies|investigation) (?:is|are) (?:needed|required|warranted)",
        r"warrants? further (?:investigation|study|research)",
        r"remains? to be (?:determined|elucidated|investigated|clarified)",
    ],
    "unclear_mechanism": [
        r"(?:mechanism|pathway) (?:remains?|is) (?:unclear|unknown|poorly understood)",
        r"underlying (?:mechanism|cause) (?:is|remains?) not (?:fully )? understood",
        r"(?:exact|precise) (?:mechanism|pathway) (?:is|has) not been (?:identified|determined)",
        r"how .+ (?:remains|is) (?:unclear|unknown)",
    ],
    "lacking_data": [
        r"(?:limited|scarce|insufficient) (?:data|evidence|studies)",
        r"(?:lack|absence) of (?:clinical|experimental) (?:data|evidence)",
        r"(?:few|limited) studies have (?:examined|investigated|explored)",
        r"(?:no|little) (?:data|evidence) (?:exists?|is available)",
    ],
    "emerging_question": [
        r"(?:whether|if) .+ (?:remains|is) (?:unclear|unknown|to be determined)",
        r"(?:it is|remains?) (?:unclear|unknown) (?:whether|if|how)",
        r"(?:the|a) (?:question|issue) of .+ (?:remains|is) (?:open|unresolved)",
        r"(?:controversial|debated) (?:whether|if)",
    ],
}

# Topic modifiers for gap analysis
RESEARCH_MODIFIERS = [
    "biomarker", "mechanism", "therapy", "treatment", "prevention",
    "diagnosis", "prognosis", "epidemiology", "pathogenesis", "etiology",
    "clinical trial", "long-term", "pediatric", "elderly", "combination therapy",
    "resistance", "side effects", "quality of life", "cost-effectiveness",
]


# ============== PubMed Helpers ==============

async def search_pubmed_with_text(
    query: str,
    max_results: int = 50,
    get_abstracts: bool = True
) -> List[Dict]:
    """Search PubMed and optionally get abstracts."""
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(query)

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Search
            search_url = f"{PUBMED_BASE}/esearch.fcgi?db=pubmed&term={encoded_query}&retmax={max_results}&retmode=json&sort=date"
            search_resp = await client.get(search_url)

            if search_resp.status_code != 200:
                return []

            search_data = search_resp.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list or not get_abstracts:
                return [{"pmid": pmid} for pmid in id_list]

            # Fetch abstracts
            await asyncio.sleep(0.5)
            fetch_url = f"{PUBMED_BASE}/efetch.fcgi?db=pubmed&id={','.join(id_list[:30])}&retmode=xml&rettype=abstract"
            fetch_resp = await client.get(fetch_url)

            if fetch_resp.status_code != 200:
                return []

            return parse_abstracts(fetch_resp.text)

    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def parse_abstracts(xml_text: str) -> List[Dict]:
    """Parse PubMed XML to extract titles and abstracts."""
    papers = []

    article_pattern = r'<PubmedArticle>(.*?)</PubmedArticle>'
    title_pattern = r'<ArticleTitle>(.*?)</ArticleTitle>'
    abstract_pattern = r'<AbstractText[^>]*>(.*?)</AbstractText>'
    pmid_pattern = r'<PMID[^>]*>(\d+)</PMID>'

    for match in re.finditer(article_pattern, xml_text, re.DOTALL):
        article = match.group(1)

        title_match = re.search(title_pattern, article, re.DOTALL)
        abstract_matches = re.findall(abstract_pattern, article, re.DOTALL)
        pmid_match = re.search(pmid_pattern, article)

        if title_match:
            title = re.sub(r'<[^>]+>', '', title_match.group(1))
            abstract = ' '.join(re.sub(r'<[^>]+>', '', a) for a in abstract_matches)

            papers.append({
                "pmid": pmid_match.group(1) if pmid_match else None,
                "title": title,
                "abstract": abstract
            })

    return papers


async def get_pubmed_count(query: str) -> int:
    """Get publication count for a query."""
    try:
        import urllib.parse
        encoded_query = urllib.parse.quote(query)

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{PUBMED_BASE}/esearch.fcgi?db=pubmed&term={encoded_query}&rettype=count"
            response = await client.get(url)

            if response.status_code == 200:
                match = re.search(r'<Count>(\d+)</Count>', response.text)
                return int(match.group(1)) if match else 0
            return 0
    except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as e:
        return 0


# ============== Gap Analysis Functions ==============

def extract_gap_sentences(text: str, gap_type: str) -> List[str]:
    """Extract sentences containing gap indicators."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    gap_sentences = []

    patterns = GAP_PATTERNS.get(gap_type, [])

    for sentence in sentences:
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                # Clean and truncate
                clean_sentence = sentence.strip()[:300]
                if len(clean_sentence) > 50:
                    gap_sentences.append(clean_sentence)
                break

    return gap_sentences[:5]  # Limit to 5 examples


def generate_research_questions(topic: str, gap_type: str) -> List[str]:
    """Generate potential research questions based on gap type."""
    questions = []

    if gap_type == "unclear_mechanism":
        questions = [
            f"What are the molecular mechanisms underlying {topic}?",
            f"How does {topic} affect cellular signaling pathways?",
            f"What are the key regulators of {topic}?",
        ]
    elif gap_type == "lacking_data":
        questions = [
            f"What is the clinical efficacy of {topic} in real-world settings?",
            f"What are the long-term outcomes of {topic}?",
            f"How does {topic} compare to existing treatments?",
        ]
    elif gap_type == "future_research":
        questions = [
            f"Can {topic} be improved with novel approaches?",
            f"What are the optimal conditions for {topic}?",
            f"How can {topic} be translated to clinical practice?",
        ]
    elif gap_type == "emerging_question":
        questions = [
            f"Is {topic} effective in specific patient populations?",
            f"What factors predict response to {topic}?",
            f"Are there biomarkers for {topic} efficacy?",
        ]

    return questions


def calculate_priority_score(gap_ratio: float, evidence_count: int, total_papers: int) -> float:
    """Calculate priority score for a research gap."""
    # Higher gap ratio = more mentions of gaps = higher priority
    ratio_score = min(100, gap_ratio * 1000)  # Scale to 0-100

    # Moderate evidence count = better (too few = noise, too many = well known)
    if evidence_count < 5:
        evidence_score = 30
    elif evidence_count < 20:
        evidence_score = 80
    elif evidence_count < 50:
        evidence_score = 60
    else:
        evidence_score = 40

    # Lower total papers but significant gaps = emerging opportunity
    if total_papers < 100:
        volume_score = 90
    elif total_papers < 500:
        volume_score = 70
    elif total_papers < 2000:
        volume_score = 50
    else:
        volume_score = 30

    # Weighted combination
    return round(ratio_score * 0.4 + evidence_score * 0.35 + volume_score * 0.25, 1)


# ============== API Endpoints ==============

@router.get("/analyze/{domain}", response_model=GapAnalysisResponse)
async def analyze_research_gaps(
    domain: str,
    limit: int = Query(10, ge=5, le=20)
):
    """
    Analyze research gaps in a domain by finding papers mentioning
    "future research needed", "unclear mechanism", etc.
    """
    # Domain queries
    domain_queries = {
        "oncology": "cancer OR tumor OR oncology",
        "neuroscience": "neuroscience OR neurodegenerative OR brain disorder",
        "genomics": "genomics OR CRISPR OR gene therapy",
        "immunology": "immunology OR immunotherapy OR immune response",
        "infectious_disease": "infectious disease OR pathogen OR vaccine",
        "ai_medicine": "artificial intelligence medicine OR machine learning clinical",
    }

    if domain not in domain_queries:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain. Available: {list(domain_queries.keys())}"
        )

    base_query = domain_queries[domain]
    gaps = []

    for gap_type, patterns in GAP_PATTERNS.items():
        # Create search query with gap indicators
        gap_terms = " OR ".join([f'"{p[:30]}"' for p in patterns[:2]])
        search_query = f"({base_query}) AND ({gap_terms})"

        # Get papers mentioning gaps
        gap_papers = await search_pubmed_with_text(search_query, max_results=30)
        await asyncio.sleep(0.5)

        # Get total papers in domain
        total_count = await get_pubmed_count(base_query)

        if gap_papers:
            # Extract gap sentences from abstracts
            all_contexts = []
            for paper in gap_papers:
                text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                contexts = extract_gap_sentences(text, gap_type)
                all_contexts.extend(contexts)

            evidence_count = len(gap_papers)
            gap_ratio = evidence_count / max(1, total_count)

            gaps.append(ResearchGap(
                topic=domain,
                gap_type=gap_type,
                gap_description=f"Research gaps related to {gap_type.replace('_', ' ')} in {domain}",
                evidence_count=evidence_count,
                total_papers=total_count,
                gap_ratio=round(gap_ratio, 4),
                sample_contexts=all_contexts[:3],
                suggested_questions=generate_research_questions(domain, gap_type),
                priority_score=calculate_priority_score(gap_ratio, evidence_count, total_count)
            ))

        await asyncio.sleep(0.3)

    # Sort by priority score
    gaps.sort(key=lambda x: x.priority_score, reverse=True)

    return GapAnalysisResponse(
        domain=domain,
        total_gaps_found=len(gaps),
        research_gaps=gaps[:limit],
        methodology="NLP pattern matching for gap indicators in recent PubMed abstracts",
        analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )


@router.get("/understudied", response_model=UnderstudiedResponse)
async def find_understudied_areas(
    topic: str = Query(..., description="Base research topic"),
    limit: int = Query(10, ge=5, le=20)
):
    """
    Find understudied sub-areas within a research topic.

    Compares publication counts for topic + modifier combinations
    to identify areas with low coverage.
    """
    understudied = []

    # Get base topic count
    base_count = await get_pubmed_count(topic)

    if base_count == 0:
        return UnderstudiedResponse(
            base_topic=topic,
            understudied_areas=[],
            methodology="No papers found for base topic",
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    for modifier in RESEARCH_MODIFIERS:
        combined_query = f"{topic} AND {modifier}"
        combined_count = await get_pubmed_count(combined_query)

        coverage_ratio = combined_count / base_count

        # Consider understudied if coverage < 5%
        if coverage_ratio < 0.05 and combined_count >= 10:
            opportunity_score = (1 - coverage_ratio) * 100

            understudied.append(UnderstudiedArea(
                topic_combination=f"{topic} + {modifier}",
                base_topic_papers=base_count,
                combined_topic_papers=combined_count,
                coverage_ratio=round(coverage_ratio, 4),
                opportunity_score=round(opportunity_score, 1)
            ))

        await asyncio.sleep(0.3)

    # Sort by opportunity score (highest first)
    understudied.sort(key=lambda x: x.opportunity_score, reverse=True)

    return UnderstudiedResponse(
        base_topic=topic,
        understudied_areas=understudied[:limit],
        methodology="Publication coverage analysis: combined_papers / base_papers ratio",
        analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M")
    )


@router.get("/emerging-questions/{domain}")
async def find_emerging_questions(
    domain: str,
    limit: int = Query(10, ge=5, le=20)
):
    """
    Extract emerging research questions from recent papers.

    Looks for question patterns in conclusions/discussions.
    """
    domain_queries = {
        "oncology": "cancer therapy recent advances",
        "neuroscience": "neurodegenerative disease treatment",
        "genomics": "gene editing applications",
        "immunology": "immunotherapy cancer",
        "infectious_disease": "vaccine development",
        "ai_medicine": "artificial intelligence clinical diagnosis",
    }

    if domain not in domain_queries:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain. Available: {list(domain_queries.keys())}"
        )

    # Get recent papers
    papers = await search_pubmed_with_text(domain_queries[domain], max_results=50)

    questions = []

    # Question patterns
    question_patterns = [
        (r"whether .+ (?:can|could|will|should|is|are)", "exploratory"),
        (r"how .+ (?:affects?|influences?|regulates?|contributes?)", "mechanism"),
        (r"(?:the|a) role of .+ in", "functional"),
        (r"(?:optimal|best) .+ for", "optimization"),
        (r"(?:safety|efficacy) of .+ in", "clinical"),
    ]

    for paper in papers:
        text = paper.get("abstract", "")

        for pattern, q_type in question_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:1]:  # One per paper
                # Construct question from match
                question_text = match.strip()
                if len(question_text) > 20:
                    questions.append({
                        "question": f"What {question_text}?",
                        "source_paper": paper.get("title", "")[:100],
                        "context": text[:200] if text else "",
                        "question_type": q_type,
                        "relevance_score": 0.7  # Default score
                    })

    # Deduplicate and limit
    seen = set()
    unique_questions = []
    for q in questions:
        q_lower = q["question"].lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_questions.append(q)

    return {
        "domain": domain,
        "questions": unique_questions[:limit],
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }


@router.get("/opportunity-matrix")
async def get_opportunity_matrix(
    topic: str = Query(..., description="Research topic"),
):
    """
    Generate a research opportunity matrix showing
    gap areas by type and potential impact.
    """
    # Get papers with gap indicators
    gap_counts = {}

    for gap_type in GAP_PATTERNS.keys():
        patterns = GAP_PATTERNS[gap_type]
        gap_term = patterns[0][:25]  # Use first pattern
        query = f"{topic} AND \"{gap_term}\""

        count = await get_pubmed_count(query)
        gap_counts[gap_type] = count
        await asyncio.sleep(0.3)

    total = await get_pubmed_count(topic)

    matrix = {
        "topic": topic,
        "total_papers": total,
        "gap_distribution": {
            gap_type: {
                "count": count,
                "percentage": round(count / max(1, total) * 100, 2)
            }
            for gap_type, count in gap_counts.items()
        },
        "top_opportunity": max(gap_counts.items(), key=lambda x: x[1])[0] if gap_counts else None,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    return matrix
