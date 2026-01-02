"""
Core Paper Reranker Module

Re-ranks search results to prioritize "core papers" that define the field,
not just topically relevant papers.

Scoring dimensions:
1. Article Type (Review > Guideline > Original > Letter)
2. Field Centrality (general vs niche topics)
3. Clinical/Translational Importance
4. Recency (last 3-5 years preferred)
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class CorePaperScore:
    """Detailed scoring breakdown for a paper."""
    article_type_score: float  # 0-30
    centrality_score: float    # 0-25
    clinical_score: float      # 0-25
    recency_score: float       # 0-20
    total_score: float         # 0-100
    article_type: str          # Detected article type
    is_core_paper: bool        # True if total_score >= 60
    explanation: str           # Human-readable explanation


class CorePaperReranker:
    """
    Re-ranks papers to prioritize core/representative papers over niche studies.
    """

    # Article type patterns and scores
    ARTICLE_TYPE_PATTERNS = {
        "systematic_review": {
            "patterns": [
                r"systematic\s+review",
                r"meta-analysis",
                r"meta\s+analysis",
                r"metaanalysis",
                r"umbrella\s+review",
                r"scoping\s+review",
            ],
            "score": 30,
            "label": "Systematic Review/Meta-analysis"
        },
        "guideline": {
            "patterns": [
                r"guideline",
                r"clinical\s+practice",
                r"consensus\s+statement",
                r"recommendations?\s+for",
                r"expert\s+consensus",
                r"position\s+statement",
            ],
            "score": 28,
            "label": "Clinical Guideline"
        },
        "review": {
            "patterns": [
                r"\breview\b",
                r"state\s+of\s+the\s+art",
                r"current\s+perspectives?",
                r"comprehensive\s+overview",
                r"narrative\s+review",
            ],
            "score": 25,
            "label": "Review Article"
        },
        "clinical_trial": {
            "patterns": [
                r"clinical\s+trial",
                r"randomized\s+controlled",
                r"randomised\s+controlled",
                r"phase\s+[I1-4]+",
                r"rct\b",
                r"double-blind",
                r"placebo-controlled",
            ],
            "score": 22,
            "label": "Clinical Trial"
        },
        "cohort_study": {
            "patterns": [
                r"cohort\s+study",
                r"prospective\s+study",
                r"retrospective\s+study",
                r"longitudinal\s+study",
                r"population-based",
            ],
            "score": 18,
            "label": "Cohort Study"
        },
        "original_research": {
            "patterns": [
                r"original\s+research",
                r"research\s+article",
                r"we\s+investigated",
                r"we\s+examined",
                r"we\s+analyzed",
                r"we\s+studied",
            ],
            "score": 15,
            "label": "Original Research"
        },
        "case_series": {
            "patterns": [
                r"case\s+series",
                r"case\s+reports?",
                r"case\s+study",
            ],
            "score": 10,
            "label": "Case Report/Series"
        },
        "letter": {
            "patterns": [
                r"\bletter\b",
                r"correspondence",
                r"comment\s+on",
                r"reply\s+to",
                r"editorial",
                r"brief\s+communication",
            ],
            "score": 5,
            "label": "Letter/Editorial"
        },
    }

    # Core topic keywords (general, field-defining terms)
    CORE_TOPIC_KEYWORDS = {
        "breast_cancer": [
            "breast cancer", "breast carcinoma", "mammary",
            "treatment", "therapy", "survival", "prognosis",
            "screening", "diagnosis", "staging",
            "chemotherapy", "endocrine therapy", "targeted therapy",
            "HER2", "ER-positive", "triple-negative",
            "mastectomy", "lumpectomy", "radiation",
            "outcomes", "recurrence", "metastatic",
        ],
        "oncology": [
            "cancer", "tumor", "carcinoma", "malignant",
            "treatment", "therapy", "survival", "prognosis",
            "chemotherapy", "immunotherapy", "targeted therapy",
            "outcomes", "clinical", "patients",
        ],
    }

    # Niche/specialized keywords (indicate less central papers)
    NICHE_KEYWORDS = [
        "mechanism", "pathway", "signaling", "expression",
        "cell line", "in vitro", "mouse model", "xenograft",
        "protein", "gene", "mutation", "polymorphism",
        "biomarker", "marker", "sequencing",
        "computational", "bioinformatics", "machine learning",
        "novel", "new", "emerging",
    ]

    # Clinical importance indicators
    CLINICAL_KEYWORDS = [
        "patient", "patients", "clinical", "treatment",
        "therapy", "survival", "outcome", "prognosis",
        "efficacy", "safety", "trial", "cohort",
        "response", "remission", "progression",
        "quality of life", "adverse events",
        "real-world", "evidence-based",
    ]

    # Translational importance indicators
    TRANSLATIONAL_KEYWORDS = [
        "translational", "bench to bedside",
        "therapeutic target", "drug target",
        "precision medicine", "personalized",
        "biomarker", "predictive", "prognostic",
    ]

    def __init__(self, domain: str = "oncology"):
        """
        Initialize reranker with domain-specific settings.

        Args:
            domain: Research domain (oncology, breast_cancer, etc.)
        """
        self.domain = domain
        self.core_keywords = self.CORE_TOPIC_KEYWORDS.get(
            domain, self.CORE_TOPIC_KEYWORDS.get("oncology", [])
        )

    def _detect_article_type(self, title: str, abstract: str) -> tuple[str, float, str]:
        """
        Detect article type from title and abstract.

        Returns:
            (type_key, score, label)
        """
        text = f"{title} {abstract}".lower()

        for type_key, config in self.ARTICLE_TYPE_PATTERNS.items():
            for pattern in config["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    return type_key, config["score"], config["label"]

        # Default to original research
        return "original_research", 15, "Original Research"

    def _calculate_centrality_score(self, title: str, abstract: str) -> float:
        """
        Calculate field centrality score (0-25).

        Higher score = more central/representative paper
        Lower score = more niche/specialized paper
        """
        text = f"{title} {abstract}".lower()

        # Count core keyword matches
        core_matches = sum(
            1 for kw in self.core_keywords
            if kw.lower() in text
        )

        # Count niche keyword matches
        niche_matches = sum(
            1 for kw in self.NICHE_KEYWORDS
            if kw.lower() in text
        )

        # Base score from core matches
        if core_matches >= 8:
            base_score = 25
        elif core_matches >= 5:
            base_score = 20
        elif core_matches >= 3:
            base_score = 15
        elif core_matches >= 1:
            base_score = 10
        else:
            base_score = 5

        # Penalty for being too niche
        if niche_matches >= 5:
            penalty = 10
        elif niche_matches >= 3:
            penalty = 5
        else:
            penalty = 0

        return max(0, base_score - penalty)

    def _calculate_clinical_score(self, title: str, abstract: str) -> float:
        """
        Calculate clinical/translational importance score (0-25).
        """
        text = f"{title} {abstract}".lower()

        # Count clinical keywords
        clinical_matches = sum(
            1 for kw in self.CLINICAL_KEYWORDS
            if kw.lower() in text
        )

        # Count translational keywords
        translational_matches = sum(
            1 for kw in self.TRANSLATIONAL_KEYWORDS
            if kw.lower() in text
        )

        # Patient numbers indicator
        has_patient_numbers = bool(re.search(
            r"\b(\d+)\s*(patients?|subjects?|participants?|cases?)\b",
            text
        ))

        # Calculate score
        score = 0

        # Clinical content
        if clinical_matches >= 6:
            score += 15
        elif clinical_matches >= 3:
            score += 10
        elif clinical_matches >= 1:
            score += 5

        # Translational content
        if translational_matches >= 2:
            score += 5
        elif translational_matches >= 1:
            score += 3

        # Patient data
        if has_patient_numbers:
            score += 5

        return min(25, score)

    def _calculate_recency_score(self, year: int) -> float:
        """
        Calculate recency score (0-20).

        Prefers papers from last 3-5 years.
        """
        current_year = datetime.now().year

        age = current_year - year

        if age <= 1:
            return 20  # Very recent
        elif age <= 3:
            return 18  # Recent
        elif age <= 5:
            return 15  # Moderately recent
        elif age <= 7:
            return 10
        elif age <= 10:
            return 5
        else:
            return 2

    def score_paper(
        self,
        title: str,
        abstract: str,
        year: int,
        journal: str = "",
        citation_count: int = 0
    ) -> CorePaperScore:
        """
        Calculate comprehensive core paper score.

        Args:
            title: Paper title
            abstract: Paper abstract
            year: Publication year
            journal: Journal name (optional, for future use)
            citation_count: Number of citations (optional, for future use)

        Returns:
            CorePaperScore with detailed breakdown
        """
        # Detect article type
        article_type_key, article_type_score, article_type_label = \
            self._detect_article_type(title, abstract)

        # Calculate other scores
        centrality_score = self._calculate_centrality_score(title, abstract)
        clinical_score = self._calculate_clinical_score(title, abstract)
        recency_score = self._calculate_recency_score(year)

        # Total score
        total_score = (
            article_type_score +
            centrality_score +
            clinical_score +
            recency_score
        )

        # Determine if core paper
        is_core_paper = total_score >= 60

        # Generate explanation
        explanation_parts = []

        if article_type_score >= 25:
            explanation_parts.append(f"High-value article type ({article_type_label})")
        elif article_type_score >= 15:
            explanation_parts.append(f"Standard article type ({article_type_label})")
        else:
            explanation_parts.append(f"Lower-priority article type ({article_type_label})")

        if centrality_score >= 20:
            explanation_parts.append("Central to field")
        elif centrality_score >= 10:
            explanation_parts.append("Moderately central")
        else:
            explanation_parts.append("Specialized/niche topic")

        if clinical_score >= 15:
            explanation_parts.append("Strong clinical focus")
        elif clinical_score >= 8:
            explanation_parts.append("Some clinical relevance")

        if recency_score >= 15:
            explanation_parts.append("Recent publication")

        return CorePaperScore(
            article_type_score=article_type_score,
            centrality_score=centrality_score,
            clinical_score=clinical_score,
            recency_score=recency_score,
            total_score=total_score,
            article_type=article_type_label,
            is_core_paper=is_core_paper,
            explanation="; ".join(explanation_parts)
        )

    def rerank_papers(
        self,
        papers: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank a list of papers by core paper score.

        Args:
            papers: List of paper dicts with title, abstract, year
            top_k: Optional limit on returned papers

        Returns:
            Re-ranked list of papers with added scoring info
        """
        scored_papers = []

        for paper in papers:
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            year = paper.get("year", datetime.now().year)
            journal = paper.get("journal", "")
            citations = paper.get("citation_count", 0)

            score = self.score_paper(
                title=title,
                abstract=abstract,
                year=year,
                journal=journal,
                citation_count=citations
            )

            # Add scoring info to paper
            paper_with_score = paper.copy()
            paper_with_score["core_score"] = score.total_score
            paper_with_score["article_type"] = score.article_type
            paper_with_score["is_core_paper"] = score.is_core_paper
            paper_with_score["score_breakdown"] = {
                "article_type": score.article_type_score,
                "centrality": score.centrality_score,
                "clinical": score.clinical_score,
                "recency": score.recency_score,
            }
            paper_with_score["score_explanation"] = score.explanation

            scored_papers.append(paper_with_score)

        # Sort by core score (descending)
        scored_papers.sort(key=lambda p: p["core_score"], reverse=True)

        if top_k:
            return scored_papers[:top_k]

        return scored_papers


# Singleton instance
_reranker: Optional[CorePaperReranker] = None


def get_reranker(domain: str = "oncology") -> CorePaperReranker:
    """Get or create reranker instance."""
    global _reranker
    if _reranker is None or _reranker.domain != domain:
        _reranker = CorePaperReranker(domain=domain)
    return _reranker
