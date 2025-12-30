"""
Research Validation Module.

Features:
- Result reliability scoring
- Cross-paper validation
- Summary consistency check
- Similarity-based confidence assessment
"""
from dataclasses import dataclass, field
from typing import Optional
import math
from collections import Counter

from .vector_store import create_vector_store, SearchResult
from .embeddings import get_embedder


@dataclass
class ValidationResult:
    """Validation result for a research finding or summary."""
    item_validated: str  # What was validated (paper title, query, etc.)
    validation_type: str  # "summary", "finding", "claim"

    # Scores (0-100)
    overall_confidence: float = 0.0
    consistency_score: float = 0.0      # How consistent across sources
    coverage_score: float = 0.0         # How many sources mention it
    similarity_score: float = 0.0       # Semantic similarity to sources

    # Supporting evidence
    supporting_papers: list[dict] = field(default_factory=list)
    conflicting_papers: list[dict] = field(default_factory=list)

    # Details
    validation_notes: list[str] = field(default_factory=list)

    def format(self) -> str:
        """Format validation result for display."""
        output = []
        output.append("=" * 70)
        output.append("✅ Validation Report")
        output.append("=" * 70)
        output.append(f"Validated: {self.item_validated[:60]}...")
        output.append(f"Type: {self.validation_type}")

        # Confidence bar
        conf_bar = "█" * int(self.overall_confidence / 5) + "░" * (20 - int(self.overall_confidence / 5))
        output.append(f"\n📊 Overall Confidence: [{conf_bar}] {self.overall_confidence:.1f}%")

        output.append("\n📈 Score Breakdown:")
        output.append(f"  • Consistency: {self.consistency_score:.1f}%")
        output.append(f"  • Coverage:    {self.coverage_score:.1f}%")
        output.append(f"  • Similarity:  {self.similarity_score:.1f}%")

        # Interpretation
        if self.overall_confidence >= 80:
            interpretation = "High confidence - well supported by multiple sources"
        elif self.overall_confidence >= 60:
            interpretation = "Moderate confidence - some supporting evidence"
        elif self.overall_confidence >= 40:
            interpretation = "Low confidence - limited evidence"
        else:
            interpretation = "Very low confidence - may need more research"

        output.append(f"\n💡 Interpretation: {interpretation}")

        if self.supporting_papers:
            output.append("\n📚 Supporting Papers:")
            for paper in self.supporting_papers[:5]:
                output.append(f"  ✓ {paper['title'][:50]}... (sim: {paper.get('similarity', 0):.1%})")

        if self.conflicting_papers:
            output.append("\n⚠️ Potentially Conflicting Papers:")
            for paper in self.conflicting_papers[:3]:
                output.append(f"  ✗ {paper['title'][:50]}...")

        if self.validation_notes:
            output.append("\n📝 Notes:")
            for note in self.validation_notes:
                output.append(f"  • {note}")

        return "\n".join(output)


class ResearchValidator:
    """
    Validate research findings and summaries.

    Provides confidence scoring based on:
    - Cross-reference with other papers
    - Semantic similarity analysis
    - Consistency checking
    """

    def __init__(self, disease_domain: str):
        """Initialize validator."""
        self.disease_domain = disease_domain
        self.vector_store = create_vector_store(disease_domain=disease_domain)
        self.embeddings = get_embedder()

    def validate_summary(
        self,
        paper_title: str,
        summary_text: str
    ) -> ValidationResult:
        """
        Validate a paper summary against the original content.

        Args:
            paper_title: Title of the summarized paper
            summary_text: The summary to validate

        Returns:
            ValidationResult with confidence scores
        """
        # Get original paper content
        results = self.vector_store.collection.get(
            where={"paper_title": paper_title},
            include=["documents", "embeddings"]
        )

        if not results["ids"]:
            return ValidationResult(
                item_validated=paper_title,
                validation_type="summary",
                validation_notes=["Original paper not found in database"]
            )

        # Embed summary
        summary_embedding = self.embeddings.embed_text(summary_text)

        # Calculate similarity to original chunks
        similarities = []
        for emb in results["embeddings"]:
            sim = self._cosine_similarity(summary_embedding, emb)
            similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0

        # Check coverage - what percentage of key concepts are in summary
        original_text = " ".join(results["documents"])
        coverage = self._calculate_coverage(summary_text, original_text)

        # Calculate scores
        similarity_score = avg_similarity * 100
        coverage_score = coverage * 100
        consistency_score = (max_similarity * 0.6 + avg_similarity * 0.4) * 100

        overall = (similarity_score * 0.4 + coverage_score * 0.3 + consistency_score * 0.3)

        return ValidationResult(
            item_validated=paper_title,
            validation_type="summary",
            overall_confidence=overall,
            consistency_score=consistency_score,
            coverage_score=coverage_score,
            similarity_score=similarity_score,
            validation_notes=[
                f"Summary covers {coverage:.0%} of key concepts",
                f"Max chunk similarity: {max_similarity:.2f}",
                f"Avg chunk similarity: {avg_similarity:.2f}"
            ]
        )

    def validate_claim(
        self,
        claim: str,
        top_k: int = 10
    ) -> ValidationResult:
        """
        Validate a research claim against indexed papers.

        Args:
            claim: The claim to validate (e.g., "SDHB mutations increase metastatic risk")
            top_k: Number of papers to check

        Returns:
            ValidationResult with supporting/conflicting evidence
        """
        # Search for relevant content
        results = self.vector_store.search(claim, top_k=top_k)

        if not results:
            return ValidationResult(
                item_validated=claim,
                validation_type="claim",
                validation_notes=["No relevant papers found for this claim"]
            )

        # Analyze results
        supporting = []
        conflicting = []
        similarities = []

        for result in results:
            sim = result.relevance_score / 100
            similarities.append(sim)

            paper_info = {
                "title": result.metadata.get("paper_title", "Unknown"),
                "section": result.metadata.get("section", "Unknown"),
                "similarity": sim,
                "content_preview": result.content[:200]
            }

            # High similarity suggests support
            if sim > 0.6:
                supporting.append(paper_info)
            elif sim < 0.3:
                # Low similarity might indicate conflict (simplified heuristic)
                conflicting.append(paper_info)

        # Calculate scores
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        max_sim = max(similarities) if similarities else 0

        # Coverage: how many unique papers mention related content
        unique_papers = set(r.metadata.get("paper_title", "") for r in results)
        coverage = len(unique_papers) / max(self.vector_store.count / 50, 1)  # Normalize

        similarity_score = avg_sim * 100
        coverage_score = min(coverage * 100, 100)
        consistency_score = (len(supporting) / len(results)) * 100 if results else 0

        overall = (similarity_score * 0.4 + coverage_score * 0.3 + consistency_score * 0.3)

        return ValidationResult(
            item_validated=claim,
            validation_type="claim",
            overall_confidence=overall,
            consistency_score=consistency_score,
            coverage_score=coverage_score,
            similarity_score=similarity_score,
            supporting_papers=supporting,
            conflicting_papers=conflicting,
            validation_notes=[
                f"Found {len(supporting)} supporting sources",
                f"Found {len(conflicting)} potentially conflicting sources",
                f"Claim appears in {len(unique_papers)} unique papers"
            ]
        )

    def validate_paper_consistency(self, paper_title: str) -> ValidationResult:
        """
        Check internal consistency of a paper.

        Validates that different sections of a paper are consistent
        with each other (e.g., Abstract matches Conclusions).
        """
        results = self.vector_store.collection.get(
            where={"paper_title": paper_title},
            include=["documents", "metadatas", "embeddings"]
        )

        if not results["ids"]:
            return ValidationResult(
                item_validated=paper_title,
                validation_type="consistency",
                validation_notes=["Paper not found"]
            )

        # Group by section
        sections = {}
        for doc, meta, emb in zip(results["documents"], results["metadatas"], results["embeddings"]):
            section = meta.get("section", "Unknown")
            if section not in sections:
                sections[section] = {"docs": [], "embeddings": []}
            sections[section]["docs"].append(doc)
            sections[section]["embeddings"].append(emb)

        # Calculate section embeddings (average)
        section_embeddings = {}
        for section, data in sections.items():
            if data["embeddings"]:
                avg = [sum(e[i] for e in data["embeddings"]) / len(data["embeddings"])
                       for i in range(len(data["embeddings"][0]))]
                section_embeddings[section] = avg

        # Check consistency between key sections
        consistency_pairs = [
            ("Abstract", "Conclusion"),
            ("Abstract", "Results"),
            ("Methods", "Results"),
            ("Introduction", "Discussion")
        ]

        pair_scores = []
        notes = []

        for sec1, sec2 in consistency_pairs:
            # Try variations
            s1 = section_embeddings.get(sec1) or section_embeddings.get(sec1 + "s")
            s2 = section_embeddings.get(sec2) or section_embeddings.get(sec2 + "s")

            if s1 and s2:
                sim = self._cosine_similarity(s1, s2)
                pair_scores.append(sim)
                notes.append(f"{sec1} ↔ {sec2}: {sim:.2f}")

        if not pair_scores:
            return ValidationResult(
                item_validated=paper_title,
                validation_type="consistency",
                validation_notes=["Not enough sections to check consistency"]
            )

        avg_consistency = sum(pair_scores) / len(pair_scores)

        return ValidationResult(
            item_validated=paper_title,
            validation_type="consistency",
            overall_confidence=avg_consistency * 100,
            consistency_score=avg_consistency * 100,
            coverage_score=len(sections) / 5 * 100,  # Normalize by expected sections
            similarity_score=max(pair_scores) * 100,
            validation_notes=notes
        )

    def cross_validate_papers(
        self,
        paper_titles: list[str]
    ) -> list[ValidationResult]:
        """
        Cross-validate findings across multiple papers.

        Returns validation results for each paper based on how
        well its findings are supported by other papers.
        """
        results = []

        for title in paper_titles:
            # Get paper content
            paper_data = self.vector_store.collection.get(
                where={"paper_title": title},
                include=["documents"]
            )

            if not paper_data["ids"]:
                continue

            # Use first few chunks as representative content
            representative = " ".join(paper_data["documents"][:3])

            # Search other papers
            search_results = self.vector_store.search(representative[:1000], top_k=20)

            # Filter out self-references
            other_papers = [r for r in search_results
                           if r.metadata.get("paper_title") != title]

            # Calculate support from other papers
            if other_papers:
                similarities = [r.relevance_score / 100 for r in other_papers]
                unique_supporters = set(r.metadata.get("paper_title") for r in other_papers
                                       if r.relevance_score > 50)

                result = ValidationResult(
                    item_validated=title,
                    validation_type="cross-validation",
                    overall_confidence=sum(similarities[:5]) / 5 * 100,
                    similarity_score=max(similarities) * 100,
                    coverage_score=len(unique_supporters) / len(paper_titles) * 100,
                    consistency_score=sum(similarities) / len(similarities) * 100,
                    supporting_papers=[
                        {"title": r.metadata.get("paper_title"), "similarity": r.relevance_score / 100}
                        for r in other_papers[:5]
                    ]
                )
            else:
                result = ValidationResult(
                    item_validated=title,
                    validation_type="cross-validation",
                    validation_notes=["No cross-references found"]
                )

            results.append(result)

        return results

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _calculate_coverage(self, summary: str, original: str) -> float:
        """Calculate what fraction of key concepts in original appear in summary."""
        import re

        # Extract significant words (4+ chars, not stopwords)
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'which', 'their', 'more'}

        def extract_words(text):
            words = set(re.findall(r'\b[a-z]{4,}\b', text.lower()))
            return words - stopwords

        original_words = extract_words(original)
        summary_words = extract_words(summary)

        if not original_words:
            return 0.0

        # How many original key words appear in summary
        overlap = len(original_words & summary_words)
        return overlap / len(original_words)


def create_validator(disease_domain: str) -> ResearchValidator:
    """Create a validator instance."""
    return ResearchValidator(disease_domain=disease_domain)
