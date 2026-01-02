"""
Lens Classifier Module

Classifies papers into four research exploration lenses:
1. Overview - Reviews, consensus, core concepts
2. Trend - Recent high-impact, emerging topics
3. Mechanism - Gene/pathway/molecular mechanisms
4. Clinical - Trials, biomarkers, guidelines, translational

Papers can belong to multiple lenses with varying confidence scores.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum


class Lens(str, Enum):
    OVERVIEW = "overview"
    TREND = "trend"
    MECHANISM = "mechanism"
    CLINICAL = "clinical"


@dataclass
class LensScore:
    """Score for a single lens."""
    lens: Lens
    score: float  # 0-100
    confidence: str  # "high", "medium", "low"
    reasons: List[str] = field(default_factory=list)


@dataclass
class LensClassification:
    """Complete lens classification for a paper."""
    primary_lens: Lens
    primary_score: float
    all_scores: Dict[Lens, LensScore]
    explanation: str


class LensClassifier:
    """
    Classifies papers into research exploration lenses.

    Lens definitions:
    - Overview: Helps understand the topic (reviews, summaries, definitions)
    - Trend: What's changing now (recent, high-impact, emerging)
    - Mechanism: How it works (genes, pathways, experiments)
    - Clinical: How it's applied (trials, treatments, guidelines)
    """

    # Overview lens indicators
    OVERVIEW_ARTICLE_TYPES = [
        "systematic review", "meta-analysis", "review article",
        "narrative review", "scoping review", "guideline",
        "consensus statement", "position statement"
    ]

    OVERVIEW_KEYWORDS = [
        "review", "overview", "comprehensive", "summary",
        "state of the art", "current understanding", "perspectives",
        "update on", "advances in", "introduction to",
        "principles", "fundamentals", "concepts",
    ]

    # Trend lens indicators
    TREND_KEYWORDS = [
        "novel", "emerging", "breakthrough", "first-in-class",
        "paradigm shift", "promising", "innovative", "cutting-edge",
        "next-generation", "latest", "recent advances",
        "new approach", "frontier", "revolutionizing",
    ]

    # Mechanism lens indicators
    MECHANISM_KEYWORDS = [
        # Molecular/genetic
        "gene", "genes", "mutation", "mutations", "variant",
        "expression", "regulation", "signaling", "pathway",
        "protein", "kinase", "receptor", "ligand", "enzyme",
        "transcription", "translation", "epigenetic",
        # Experimental
        "mechanism", "mechanistic", "molecular basis",
        "in vitro", "in vivo", "cell line", "mouse model",
        "xenograft", "knockdown", "knockout", "overexpression",
        "western blot", "pcr", "sequencing", "rnaseq",
        # Specific pathways
        "pi3k", "akt", "mtor", "mapk", "erk", "jak", "stat",
        "wnt", "notch", "hedgehog", "nf-kb", "p53", "rb",
    ]

    # Clinical lens indicators
    CLINICAL_ARTICLE_TYPES = [
        "clinical trial", "randomized controlled", "phase i",
        "phase ii", "phase iii", "phase iv", "cohort study",
        "clinical guideline", "practice guideline"
    ]

    CLINICAL_KEYWORDS = [
        # Study types
        "patients", "clinical trial", "randomized", "cohort",
        "prospective", "retrospective", "real-world",
        # Treatment
        "treatment", "therapy", "therapeutic", "regimen",
        "chemotherapy", "immunotherapy", "radiation",
        "surgery", "intervention", "drug", "medication",
        # Outcomes
        "survival", "response", "remission", "progression",
        "overall survival", "progression-free", "disease-free",
        "outcome", "prognosis", "efficacy", "safety",
        # Clinical markers
        "biomarker", "predictive", "prognostic", "diagnostic",
        "staging", "grading", "risk stratification",
        # Translation
        "translational", "bench to bedside", "clinical practice",
        "guideline", "recommendation", "standard of care",
    ]

    def __init__(self, trend_years: int = 3):
        """
        Initialize classifier.

        Args:
            trend_years: Papers within this many years are eligible for Trend lens
        """
        self.trend_years = trend_years
        self.current_year = datetime.now().year

    def _count_keyword_matches(self, text: str, keywords: List[str]) -> Tuple[int, List[str]]:
        """Count keyword matches and return matched keywords."""
        text_lower = text.lower()
        matches = []
        for kw in keywords:
            if kw.lower() in text_lower:
                matches.append(kw)
        return len(matches), matches

    def _calculate_overview_score(
        self,
        title: str,
        abstract: str,
        article_type: str
    ) -> LensScore:
        """Calculate Overview lens score."""
        text = f"{title} {abstract}".lower()
        article_type_lower = article_type.lower() if article_type else ""

        score = 0
        reasons = []

        # Article type boost (strongest signal)
        for otype in self.OVERVIEW_ARTICLE_TYPES:
            if otype in article_type_lower or otype in text[:200]:
                score += 50
                reasons.append(f"Article type: {otype}")
                break

        # Keyword matches
        match_count, matched = self._count_keyword_matches(text, self.OVERVIEW_KEYWORDS)
        if match_count >= 3:
            score += 30
            reasons.append(f"Multiple overview keywords: {', '.join(matched[:3])}")
        elif match_count >= 1:
            score += 15
            reasons.append(f"Overview keywords: {', '.join(matched)}")

        # Title signals (strong indicator)
        title_lower = title.lower()
        if any(kw in title_lower for kw in ["review", "overview", "update", "advances"]):
            score += 20
            reasons.append("Title indicates overview content")

        confidence = "high" if score >= 60 else "medium" if score >= 30 else "low"
        return LensScore(
            lens=Lens.OVERVIEW,
            score=min(100, score),
            confidence=confidence,
            reasons=reasons
        )

    def _calculate_trend_score(
        self,
        title: str,
        abstract: str,
        year: int,
        citation_count: int = 0
    ) -> LensScore:
        """Calculate Trend lens score."""
        text = f"{title} {abstract}".lower()

        score = 0
        reasons = []

        # Recency is primary factor
        years_old = self.current_year - year
        if years_old <= 1:
            score += 40
            reasons.append("Published within last year")
        elif years_old <= 2:
            score += 30
            reasons.append("Published within last 2 years")
        elif years_old <= self.trend_years:
            score += 20
            reasons.append(f"Published within last {self.trend_years} years")
        else:
            # Older papers get significant penalty for trend lens
            score -= 20

        # Trend keywords
        match_count, matched = self._count_keyword_matches(text, self.TREND_KEYWORDS)
        if match_count >= 2:
            score += 30
            reasons.append(f"Trend keywords: {', '.join(matched[:3])}")
        elif match_count >= 1:
            score += 15
            reasons.append(f"Trend keyword: {matched[0]}")

        # Citation velocity (if available and recent)
        if citation_count > 0 and years_old <= 2:
            citations_per_year = citation_count / max(1, years_old)
            if citations_per_year >= 50:
                score += 20
                reasons.append(f"High citation velocity: {citations_per_year:.0f}/year")
            elif citations_per_year >= 20:
                score += 10
                reasons.append(f"Good citation velocity: {citations_per_year:.0f}/year")

        confidence = "high" if score >= 60 else "medium" if score >= 30 else "low"
        return LensScore(
            lens=Lens.TREND,
            score=max(0, min(100, score)),
            confidence=confidence,
            reasons=reasons
        )

    def _calculate_mechanism_score(
        self,
        title: str,
        abstract: str
    ) -> LensScore:
        """Calculate Mechanism lens score."""
        text = f"{title} {abstract}".lower()

        score = 0
        reasons = []

        # Keyword matches
        match_count, matched = self._count_keyword_matches(text, self.MECHANISM_KEYWORDS)

        if match_count >= 8:
            score += 60
            reasons.append(f"Strong mechanistic focus: {', '.join(matched[:5])}")
        elif match_count >= 5:
            score += 45
            reasons.append(f"Mechanistic content: {', '.join(matched[:4])}")
        elif match_count >= 3:
            score += 30
            reasons.append(f"Some mechanistic elements: {', '.join(matched[:3])}")
        elif match_count >= 1:
            score += 15
            reasons.append(f"Mechanistic keywords: {', '.join(matched)}")

        # Gene/protein name patterns (e.g., TP53, BRCA1, HER2)
        gene_pattern = r'\b[A-Z][A-Z0-9]{1,5}\b'
        gene_matches = re.findall(gene_pattern, f"{title} {abstract}")
        # Filter common non-gene abbreviations
        non_genes = {"DNA", "RNA", "PCR", "USA", "FDA", "WHO", "BMI", "MRI", "PET", "CT"}
        genes = [g for g in gene_matches if g not in non_genes and len(g) >= 2]

        if len(genes) >= 5:
            score += 25
            reasons.append(f"Multiple gene/protein names: {', '.join(list(set(genes))[:4])}")
        elif len(genes) >= 2:
            score += 15
            reasons.append(f"Gene/protein names: {', '.join(list(set(genes))[:3])}")

        # Experimental methodology indicators
        exp_keywords = ["western blot", "pcr", "rnaseq", "rna-seq", "chip-seq",
                       "crispr", "transfection", "knockdown", "knockout"]
        exp_count, exp_matched = self._count_keyword_matches(text, exp_keywords)
        if exp_count >= 2:
            score += 15
            reasons.append(f"Experimental methods: {', '.join(exp_matched)}")

        confidence = "high" if score >= 60 else "medium" if score >= 30 else "low"
        return LensScore(
            lens=Lens.MECHANISM,
            score=min(100, score),
            confidence=confidence,
            reasons=reasons
        )

    def _calculate_clinical_score(
        self,
        title: str,
        abstract: str,
        article_type: str
    ) -> LensScore:
        """Calculate Clinical lens score."""
        text = f"{title} {abstract}".lower()
        article_type_lower = article_type.lower() if article_type else ""

        score = 0
        reasons = []

        # Clinical article types
        for ctype in self.CLINICAL_ARTICLE_TYPES:
            if ctype in article_type_lower or ctype in text[:300]:
                score += 40
                reasons.append(f"Clinical study type: {ctype}")
                break

        # Keyword matches
        match_count, matched = self._count_keyword_matches(text, self.CLINICAL_KEYWORDS)

        if match_count >= 8:
            score += 40
            reasons.append(f"Strong clinical focus: {', '.join(matched[:5])}")
        elif match_count >= 5:
            score += 30
            reasons.append(f"Clinical content: {', '.join(matched[:4])}")
        elif match_count >= 3:
            score += 20
            reasons.append(f"Some clinical elements: {', '.join(matched[:3])}")
        elif match_count >= 1:
            score += 10
            reasons.append(f"Clinical keywords: {', '.join(matched)}")

        # Patient numbers (strong clinical signal)
        patient_pattern = r'\b(\d+)\s*(patients?|subjects?|participants?|cases?)\b'
        patient_match = re.search(patient_pattern, text)
        if patient_match:
            n_patients = int(patient_match.group(1))
            if n_patients >= 100:
                score += 20
                reasons.append(f"Large patient cohort: {n_patients}")
            elif n_patients >= 20:
                score += 10
                reasons.append(f"Patient cohort: {n_patients}")

        confidence = "high" if score >= 60 else "medium" if score >= 30 else "low"
        return LensScore(
            lens=Lens.CLINICAL,
            score=min(100, score),
            confidence=confidence,
            reasons=reasons
        )

    def classify(
        self,
        title: str,
        abstract: str,
        year: int,
        article_type: str = "",
        citation_count: int = 0
    ) -> LensClassification:
        """
        Classify a paper into research lenses.

        Args:
            title: Paper title
            abstract: Paper abstract
            year: Publication year
            article_type: Detected article type (e.g., "Review Article")
            citation_count: Number of citations

        Returns:
            LensClassification with primary lens and all scores
        """
        # Calculate all lens scores
        overview_score = self._calculate_overview_score(title, abstract, article_type)
        trend_score = self._calculate_trend_score(title, abstract, year, citation_count)
        mechanism_score = self._calculate_mechanism_score(title, abstract)
        clinical_score = self._calculate_clinical_score(title, abstract, article_type)

        all_scores = {
            Lens.OVERVIEW: overview_score,
            Lens.TREND: trend_score,
            Lens.MECHANISM: mechanism_score,
            Lens.CLINICAL: clinical_score,
        }

        # Determine primary lens
        sorted_scores = sorted(
            all_scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )
        primary_lens = sorted_scores[0][0]
        primary_score = sorted_scores[0][1].score

        # Generate explanation
        top_reasons = sorted_scores[0][1].reasons[:2]
        explanation = f"Primary: {primary_lens.value.title()}"
        if top_reasons:
            explanation += f" ({'; '.join(top_reasons)})"

        # Add secondary lens if significant
        if sorted_scores[1][1].score >= 40:
            secondary = sorted_scores[1][0].value.title()
            explanation += f". Also relevant: {secondary}"

        return LensClassification(
            primary_lens=primary_lens,
            primary_score=primary_score,
            all_scores=all_scores,
            explanation=explanation
        )

    def classify_batch(
        self,
        papers: List[Dict]
    ) -> Dict[Lens, List[Dict]]:
        """
        Classify and group multiple papers by lens.

        Args:
            papers: List of paper dicts with title, abstract, year, etc.

        Returns:
            Dict mapping each lens to list of papers assigned to it
        """
        grouped = {lens: [] for lens in Lens}

        for paper in papers:
            classification = self.classify(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                year=paper.get("year", self.current_year),
                article_type=paper.get("article_type", ""),
                citation_count=paper.get("citation_count", 0)
            )

            # Add classification info to paper
            paper_with_lens = paper.copy()
            paper_with_lens["primary_lens"] = classification.primary_lens.value
            paper_with_lens["lens_scores"] = {
                lens.value: {
                    "score": score.score,
                    "confidence": score.confidence
                }
                for lens, score in classification.all_scores.items()
            }
            paper_with_lens["lens_explanation"] = classification.explanation

            # Add to primary lens group
            grouped[classification.primary_lens].append(paper_with_lens)

        # Sort each group by lens-specific criteria
        for lens in Lens:
            if lens == Lens.OVERVIEW:
                # Sort by core_score (reviews/guidelines first)
                grouped[lens].sort(
                    key=lambda p: p.get("core_score", 0),
                    reverse=True
                )
            elif lens == Lens.TREND:
                # Sort by year then citation velocity
                grouped[lens].sort(
                    key=lambda p: (p.get("year", 0), p.get("citation_count", 0)),
                    reverse=True
                )
            elif lens == Lens.MECHANISM:
                # Sort by mechanism score
                grouped[lens].sort(
                    key=lambda p: p.get("lens_scores", {}).get("mechanism", {}).get("score", 0),
                    reverse=True
                )
            elif lens == Lens.CLINICAL:
                # Sort by clinical score then recency
                grouped[lens].sort(
                    key=lambda p: (
                        p.get("lens_scores", {}).get("clinical", {}).get("score", 0),
                        p.get("year", 0)
                    ),
                    reverse=True
                )

        return grouped


# Singleton instance
_classifier: Optional[LensClassifier] = None


def get_lens_classifier() -> LensClassifier:
    """Get or create lens classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = LensClassifier()
    return _classifier
