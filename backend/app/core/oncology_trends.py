"""
Oncology Trends Definition System

Defines evidence-based oncology trends with:
1. Trend keywords and themes (defined FIRST)
2. Why each trend matters (context)
3. Paper-to-trend mapping logic
4. Sub-trend grouping structure

Architecture:
  트렌드 정의 → 논문 매핑 → 컨텍스트 제공
  (NOT: 논문 나열 → "이게 트렌드야")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import re


class TrendCategory(str, Enum):
    """High-level trend categories within Oncology."""
    TUMOR_EVOLUTION = "tumor_evolution"
    IMMUNOTHERAPY = "immunotherapy"
    PRECISION_MEDICINE = "precision_medicine"
    CANCER_PREVENTION = "cancer_prevention"
    TREATMENT_RESISTANCE = "treatment_resistance"
    EMERGING_TARGETS = "emerging_targets"


@dataclass
class TrendDefinition:
    """
    A defined oncology trend with evidence-based context.

    This is the SOURCE OF TRUTH - papers are mapped TO trends,
    not the other way around.
    """
    id: str
    name: str
    category: TrendCategory

    # Core identifiers
    keywords: List[str]  # Primary matching keywords
    mesh_terms: List[str] = field(default_factory=list)  # MeSH terms
    gene_symbols: List[str] = field(default_factory=list)  # Key genes

    # Context (WHY this is a trend)
    why_trending: str = ""  # "최근 3년간 Nature/Cell 계열 급증"
    clinical_relevance: str = ""  # Why clinicians care
    research_impact: str = ""  # Why researchers care

    # Evidence
    key_publications: List[str] = field(default_factory=list)  # Landmark PMIDs
    emergence_year: int = 2020  # When trend emerged
    peak_journals: List[str] = field(default_factory=list)  # Top publishing journals

    # Display
    emoji: str = "🔬"
    color: str = "#3B82F6"  # Tailwind blue-500

    def matches_paper(self, title: str, abstract: str, keywords: List[str] = None) -> float:
        """
        Calculate how well a paper matches this trend.

        Returns:
            Score 0-100 indicating match strength
        """
        text = f"{title} {abstract}".lower()
        paper_keywords = set(k.lower() for k in (keywords or []))

        score = 0
        matched_terms = []

        # Keyword matching (primary signal)
        for kw in self.keywords:
            if kw.lower() in text:
                score += 15
                matched_terms.append(kw)

        # Gene symbol matching
        for gene in self.gene_symbols:
            if re.search(rf'\b{gene}\b', text, re.IGNORECASE):
                score += 10
                matched_terms.append(gene)

        # MeSH term matching
        for mesh in self.mesh_terms:
            if mesh.lower() in text or mesh.lower() in paper_keywords:
                score += 8

        # Title match bonus (stronger signal)
        title_lower = title.lower()
        for kw in self.keywords[:3]:  # Top 3 keywords
            if kw.lower() in title_lower:
                score += 20

        return min(100, score), matched_terms


# ============================================================
# ONCOLOGY TREND DEFINITIONS (Evidence-Based)
# ============================================================

ONCOLOGY_TRENDS: Dict[str, TrendDefinition] = {

    # === TUMOR EVOLUTION & PLASTICITY ===

    "lineage_plasticity": TrendDefinition(
        id="lineage_plasticity",
        name="Lineage Plasticity & Cell State Transitions",
        category=TrendCategory.TUMOR_EVOLUTION,
        keywords=[
            "lineage plasticity", "cell state transition", "phenotypic plasticity",
            "neuroendocrine differentiation", "transdifferentiation",
            "epithelial-mesenchymal", "EMT", "dedifferentiation",
            "stem cell plasticity", "cell fate"
        ],
        gene_symbols=["NSD2", "SOX2", "ASCL1", "NEUROD1", "RB1", "TP53"],
        why_trending="2021-2024 Nature/Cell/Cancer Cell 집중 게재. 치료 저항성의 핵심 메커니즘으로 재조명",
        clinical_relevance="표적치료 및 면역치료 내성의 주요 원인. 새로운 치료 전략 필요",
        research_impact="단일세포 분석 기술 발전으로 세포 상태 전환 실시간 추적 가능",
        emergence_year=2021,
        peak_journals=["Nature", "Cell", "Cancer Cell", "Cancer Discovery"],
        emoji="🧬",
        color="#8B5CF6"
    ),

    "ecdna": TrendDefinition(
        id="ecdna",
        name="Extrachromosomal DNA (ecDNA)",
        category=TrendCategory.TUMOR_EVOLUTION,
        keywords=[
            "extrachromosomal DNA", "ecDNA", "circular DNA",
            "double minute", "chromothripsis", "gene amplification",
            "oncogene amplification", "circular extrachromosomal"
        ],
        gene_symbols=["EGFR", "MYC", "ERBB2", "MDM2"],
        why_trending="2020-2024 암 진화의 새로운 패러다임. Science/Nature에서 ecDNA 특집",
        clinical_relevance="표적치료 저항성 및 종양 이질성의 원인. 새로운 치료 타겟",
        research_impact="유전체 불안정성과 종양 진화를 설명하는 핵심 메커니즘",
        emergence_year=2020,
        peak_journals=["Nature", "Science", "Cancer Cell", "Nature Genetics"],
        emoji="🔄",
        color="#EC4899"
    ),

    # === IMMUNOTHERAPY ===

    "immunotherapy_resistance": TrendDefinition(
        id="immunotherapy_resistance",
        name="Immunotherapy Resistance Mechanisms",
        category=TrendCategory.IMMUNOTHERAPY,
        keywords=[
            "immunotherapy resistance", "checkpoint inhibitor resistance",
            "PD-1 resistance", "PD-L1 resistance", "immune escape",
            "tumor immune evasion", "immunosuppressive microenvironment",
            "T cell exhaustion", "immune checkpoint failure"
        ],
        gene_symbols=["PTEN", "JAK1", "JAK2", "B2M", "STK11", "KEAP1"],
        why_trending="면역항암제 40-60% 무반응 환자 해결 시급. 병용요법 개발 핵심",
        clinical_relevance="면역치료 반응 예측 바이오마커 및 병용전략 개발",
        research_impact="종양미세환경, 네오항원, 면역세포 기능 연구 융합",
        emergence_year=2019,
        peak_journals=["Nature Medicine", "Cancer Discovery", "JCO", "Immunity"],
        emoji="🛡️",
        color="#EF4444"
    ),

    "tumor_microenvironment": TrendDefinition(
        id="tumor_microenvironment",
        name="Tumor Microenvironment & Spatial Biology",
        category=TrendCategory.IMMUNOTHERAPY,
        keywords=[
            "tumor microenvironment", "TME", "spatial transcriptomics",
            "spatial proteomics", "immune infiltration", "CAF",
            "cancer-associated fibroblast", "myeloid cells",
            "spatial analysis", "multiplexed imaging"
        ],
        mesh_terms=["Tumor Microenvironment"],
        why_trending="공간전사체학 기술 폭발적 성장 (10x Visium, MERFISH). 2023-2024 핵심 연구 주제",
        clinical_relevance="면역치료 반응 예측, 종양 침윤 패턴 분석",
        research_impact="단일세포 + 공간정보 통합 분석의 새로운 표준",
        emergence_year=2022,
        peak_journals=["Nature", "Cell", "Nature Methods", "Cancer Discovery"],
        emoji="🔬",
        color="#F59E0B"
    ),

    # === PRECISION MEDICINE ===

    "liquid_biopsy": TrendDefinition(
        id="liquid_biopsy",
        name="Liquid Biopsy & ctDNA",
        category=TrendCategory.PRECISION_MEDICINE,
        keywords=[
            "liquid biopsy", "ctDNA", "circulating tumor DNA",
            "cell-free DNA", "cfDNA", "minimal residual disease",
            "MRD detection", "circulating tumor cells", "CTC"
        ],
        why_trending="FDA 승인 액체생검 검사 확대. 조기 발견 및 MRD 모니터링 표준화",
        clinical_relevance="비침습적 암 진단, 치료 반응 실시간 모니터링",
        research_impact="조기 암 발견 패널 (GRAIL, Guardant) 대규모 임상시험 진행",
        emergence_year=2020,
        peak_journals=["NEJM", "Lancet Oncology", "JCO", "Nature Medicine"],
        emoji="💉",
        color="#06B6D4"
    ),

    "ai_oncology": TrendDefinition(
        id="ai_oncology",
        name="AI in Oncology & Digital Pathology",
        category=TrendCategory.PRECISION_MEDICINE,
        keywords=[
            "artificial intelligence", "machine learning", "deep learning",
            "digital pathology", "AI diagnosis", "computational pathology",
            "radiomics", "pathomics", "foundation model", "multimodal AI"
        ],
        why_trending="GPT-4V, Med-PaLM 등 멀티모달 AI 발전. FDA AI 진단 승인 급증",
        clinical_relevance="병리 진단 자동화, 예후 예측, 치료 반응 예측",
        research_impact="Foundation model 기반 범용 암 진단 모델 개발 경쟁",
        emergence_year=2023,
        peak_journals=["Nature Medicine", "Lancet Digital Health", "npj Digital Medicine"],
        emoji="🤖",
        color="#10B981"
    ),

    # === CANCER PREVENTION ===

    "cancer_interception": TrendDefinition(
        id="cancer_interception",
        name="Cancer Interception & Prevention",
        category=TrendCategory.CANCER_PREVENTION,
        keywords=[
            "cancer prevention", "cancer interception", "chemoprevention",
            "premalignant", "precancerous", "risk reduction",
            "H. pylori eradication", "early detection", "screening"
        ],
        why_trending="NCI Cancer Moonshot의 핵심 목표. 조기 발견 + 예방 패러다임 전환",
        clinical_relevance="고위험군 조기 개입으로 암 발생률 감소",
        research_impact="전암병변 생물학 및 암 진행 차단 연구 활성화",
        emergence_year=2021,
        peak_journals=["NEJM", "Lancet", "Cancer Prevention Research", "Gut"],
        emoji="🛡️",
        color="#22C55E"
    ),

    # === TREATMENT RESISTANCE ===

    "drug_resistance": TrendDefinition(
        id="drug_resistance",
        name="Drug Resistance & Persistence",
        category=TrendCategory.TREATMENT_RESISTANCE,
        keywords=[
            "drug resistance", "acquired resistance", "therapeutic resistance",
            "drug tolerant", "persister cells", "residual disease",
            "resistance mechanism", "treatment failure"
        ],
        gene_symbols=["EGFR", "ALK", "KRAS", "BRAF", "AR"],
        why_trending="표적치료 내성 극복이 정밀의학의 핵심 과제",
        clinical_relevance="내성 발생 전 예측 및 병용요법 개발",
        research_impact="단일세포 수준 내성 메커니즘 규명",
        emergence_year=2019,
        peak_journals=["Cancer Discovery", "Nature Medicine", "Cancer Cell"],
        emoji="💊",
        color="#F97316"
    ),

    # === EMERGING TARGETS ===

    "epigenetic_therapy": TrendDefinition(
        id="epigenetic_therapy",
        name="Epigenetic Reprogramming & Therapy",
        category=TrendCategory.EMERGING_TARGETS,
        keywords=[
            "epigenetic", "epigenome", "DNA methylation", "histone",
            "chromatin remodeling", "epigenetic therapy", "DNMT inhibitor",
            "HDAC inhibitor", "BET inhibitor", "EZH2"
        ],
        gene_symbols=["DNMT1", "DNMT3A", "EZH2", "KMT2A", "TET2", "IDH1", "IDH2"],
        why_trending="후성유전학적 취약성을 표적으로 하는 신약 개발 활성화",
        clinical_relevance="혈액암 및 고형암에서 후성유전 치료제 임상시험 확대",
        research_impact="암 세포 리프로그래밍 및 분화 치료 전략",
        emergence_year=2020,
        peak_journals=["Nature", "Cancer Cell", "Molecular Cell", "Nature Genetics"],
        emoji="🧪",
        color="#A855F7"
    ),

    "rna_therapeutics": TrendDefinition(
        id="rna_therapeutics",
        name="RNA Therapeutics in Oncology",
        category=TrendCategory.EMERGING_TARGETS,
        keywords=[
            "mRNA vaccine", "mRNA therapy", "siRNA", "ASO",
            "antisense oligonucleotide", "RNA interference",
            "cancer vaccine", "neoantigen vaccine", "personalized vaccine"
        ],
        why_trending="COVID mRNA 백신 성공 이후 암 mRNA 백신 개발 가속화",
        clinical_relevance="개인맞춤형 암백신 임상시험 (BioNTech, Moderna)",
        research_impact="네오항원 기반 개인화 면역치료 플랫폼 구축",
        emergence_year=2021,
        peak_journals=["Nature", "Science", "Nature Medicine", "Cell"],
        emoji="💉",
        color="#3B82F6"
    ),
}


@dataclass
class TrendMatch:
    """Result of matching a paper to trends."""
    trend_id: str
    trend_name: str
    category: str
    score: float
    matched_terms: List[str]
    why_trending: str
    emoji: str
    color: str


@dataclass
class TrendGroupedPapers:
    """Papers grouped by trend category."""
    category: TrendCategory
    category_name: str
    emoji: str
    trends: Dict[str, List[dict]]  # trend_id -> papers
    total_papers: int


class OncologyTrendMatcher:
    """
    Matches papers to defined oncology trends.

    Key principle: Trends are defined FIRST, then papers are mapped to them.
    """

    def __init__(self, min_score: float = 30.0):
        self.trends = ONCOLOGY_TRENDS
        self.min_score = min_score

    def match_paper(
        self,
        title: str,
        abstract: str,
        keywords: List[str] = None
    ) -> List[TrendMatch]:
        """
        Find all trends that match a paper.

        Returns:
            List of TrendMatch sorted by score (highest first)
        """
        matches = []

        for trend_id, trend in self.trends.items():
            score, matched_terms = trend.matches_paper(title, abstract, keywords)

            if score >= self.min_score:
                matches.append(TrendMatch(
                    trend_id=trend_id,
                    trend_name=trend.name,
                    category=trend.category.value,
                    score=score,
                    matched_terms=matched_terms,
                    why_trending=trend.why_trending,
                    emoji=trend.emoji,
                    color=trend.color
                ))

        # Sort by score
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def get_primary_trend(
        self,
        title: str,
        abstract: str,
        keywords: List[str] = None
    ) -> Optional[TrendMatch]:
        """Get the single best-matching trend for a paper."""
        matches = self.match_paper(title, abstract, keywords)
        return matches[0] if matches else None

    def group_papers_by_trend(
        self,
        papers: List[dict]
    ) -> Dict[str, TrendGroupedPapers]:
        """
        Group papers by trend category.

        Returns structure:
        {
            "tumor_evolution": TrendGroupedPapers(
                category="tumor_evolution",
                trends={
                    "lineage_plasticity": [paper1, paper2],
                    "ecdna": [paper3]
                }
            )
        }
        """
        # Initialize groups
        category_groups: Dict[TrendCategory, Dict[str, List[dict]]] = {
            cat: {} for cat in TrendCategory
        }

        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            keywords = paper.get("keywords", [])

            matches = self.match_paper(title, abstract, keywords)

            if matches:
                # Assign to primary trend
                primary = matches[0]
                category = TrendCategory(primary.category)
                trend_id = primary.trend_id

                if trend_id not in category_groups[category]:
                    category_groups[category][trend_id] = []

                # Add trend info to paper
                paper_with_trend = paper.copy()
                paper_with_trend["trend_match"] = {
                    "trend_id": primary.trend_id,
                    "trend_name": primary.trend_name,
                    "score": primary.score,
                    "matched_terms": primary.matched_terms,
                    "why_trending": primary.why_trending,
                    "emoji": primary.emoji,
                }

                category_groups[category][trend_id].append(paper_with_trend)

        # Convert to TrendGroupedPapers
        result = {}
        category_names = {
            TrendCategory.TUMOR_EVOLUTION: ("Cancer Evolution & Plasticity", "🧬"),
            TrendCategory.IMMUNOTHERAPY: ("Immunotherapy & TME", "🛡️"),
            TrendCategory.PRECISION_MEDICINE: ("Precision Medicine", "🎯"),
            TrendCategory.CANCER_PREVENTION: ("Cancer Prevention", "🛡️"),
            TrendCategory.TREATMENT_RESISTANCE: ("Treatment Resistance", "💊"),
            TrendCategory.EMERGING_TARGETS: ("Emerging Targets", "🎯"),
        }

        for category, trends in category_groups.items():
            if trends:  # Only include non-empty categories
                name, emoji = category_names.get(category, (category.value, "🔬"))
                total = sum(len(papers) for papers in trends.values())

                result[category.value] = TrendGroupedPapers(
                    category=category,
                    category_name=name,
                    emoji=emoji,
                    trends=trends,
                    total_papers=total
                )

        return result

    def get_all_trends(self) -> List[Dict]:
        """Get all defined trends for display."""
        return [
            {
                "id": t.id,
                "name": t.name,
                "category": t.category.value,
                "keywords": t.keywords[:5],
                "why_trending": t.why_trending,
                "clinical_relevance": t.clinical_relevance,
                "emoji": t.emoji,
                "color": t.color,
            }
            for t in self.trends.values()
        ]


# Singleton instance
_matcher: Optional[OncologyTrendMatcher] = None


def get_trend_matcher() -> OncologyTrendMatcher:
    """Get or create trend matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = OncologyTrendMatcher()
    return _matcher
