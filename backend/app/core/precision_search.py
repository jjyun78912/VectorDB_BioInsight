"""
Precision Search Engine with Medical Vocabulary Support.

Implements lexical-first search strategy with field-aware ranking
to eliminate noise and improve disease-specific search precision.

Key Features:
- MeSH term anchoring for disease queries
- Field-aware scoring: Title > Abstract > Full text
- Lexical (BM25) first, embeddings for reranking only
- Explicit search diagnostics
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import re

from .vector_store import BioVectorStore, SearchResult, create_vector_store
from .medical_vocabulary import get_medical_vocabulary, MedicalVocabulary


class MatchField(Enum):
    """Where the match was found - determines ranking weight."""
    MESH = "mesh"           # MeSH term match (highest)
    TITLE = "title"         # Title match
    ABSTRACT = "abstract"   # Abstract match
    FULL_TEXT = "full_text" # Full text only match (lowest)
    NONE = "none"           # No match


# Field weights for scoring
FIELD_WEIGHTS = {
    MatchField.MESH: 1.0,
    MatchField.TITLE: 0.9,
    MatchField.ABSTRACT: 0.7,
    MatchField.FULL_TEXT: 0.3,
    MatchField.NONE: 0.0,
}


@dataclass
class MatchDiagnostic:
    """Explains why a result was matched."""
    field: MatchField
    matched_terms: list[str] = field(default_factory=list)
    disease_score: float = 0.0
    modifier_score: float = 0.0
    field_weight: float = 0.0
    explanation: str = ""


@dataclass
class PrecisionSearchResult:
    """Search result with diagnostics and field-aware scoring."""
    rank: int
    content: str
    paper_title: str
    section: str
    doi: str
    year: str
    pmid: str

    # Scoring details
    final_score: float
    disease_relevance: float
    field_match: MatchField

    # Diagnostics
    diagnostic: MatchDiagnostic

    # Original metadata
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchDiagnostics:
    """Overall search diagnostics."""
    query: str
    detected_disease: Optional[str]
    mesh_term: Optional[str]
    search_terms: list[str]
    modifiers: list[str]
    total_candidates: int
    filtered_results: int
    strategy_used: str
    explanation: str


class PrecisionSearch:
    """
    Precision search engine with medical vocabulary support.

    Strategy:
    1. Extract disease term from query using MeSH vocabulary
    2. Use lexical search (BM25) to get candidates
    3. Score candidates by field-aware disease relevance
    4. Apply embeddings only for final reranking within relevant results
    """

    def __init__(
        self,
        vector_store: BioVectorStore | None = None,
        disease_domain: str | None = None,
        min_disease_score: float = 0.3,
        require_title_abstract_match: bool = True
    ):
        """
        Initialize precision search.

        Args:
            vector_store: Vector store to search
            disease_domain: Disease domain for collection
            min_disease_score: Minimum disease relevance score (0-1)
            require_title_abstract_match: If True, filter out full-text-only matches
        """
        self.vector_store = vector_store or create_vector_store(disease_domain=disease_domain)
        self.vocabulary = get_medical_vocabulary()
        self.min_disease_score = min_disease_score
        self.require_title_abstract_match = require_title_abstract_match

    def search(
        self,
        query: str,
        top_k: int = 10,
        section_filter: str | None = None,
        include_diagnostics: bool = True
    ) -> tuple[list[PrecisionSearchResult], SearchDiagnostics]:
        """
        Perform precision search with disease vocabulary anchoring.

        Args:
            query: Search query
            top_k: Number of results to return
            section_filter: Optional section filter
            include_diagnostics: Include detailed diagnostics

        Returns:
            Tuple of (results, diagnostics)
        """
        # Step 1: Extract disease term and modifiers from query
        disease_key, remaining_query = self.vocabulary.extract_disease_from_query(query)

        # Build structured search query
        if disease_key:
            search_query = self.vocabulary.build_search_query(
                disease_key,
                modifiers=remaining_query.split() if remaining_query else None
            )
            mesh_term = search_query.get("primary_terms", [None])[0]
            search_terms = (
                search_query.get("primary_terms", []) +
                search_query.get("synonyms", []) +
                search_query.get("abbreviations", [])
            )
            modifiers = search_query.get("modifiers", [])
        else:
            mesh_term = None
            search_terms = query.split()
            modifiers = []

        # Step 2: Get candidates using lexical search (BM25)
        # Fetch more candidates for filtering
        candidates = self._get_lexical_candidates(
            query=query,
            top_k=top_k * 5,  # Get more for filtering
            section_filter=section_filter
        )

        total_candidates = len(candidates)

        # Step 3: Score candidates by disease relevance and field
        scored_results = []
        for result in candidates:
            diagnostic = self._analyze_match(
                content=result.content,
                metadata=result.metadata,
                disease_key=disease_key,
                search_terms=search_terms,
                modifiers=modifiers
            )

            # Apply field-aware scoring
            final_score = self._calculate_final_score(
                base_score=result.relevance_score,
                diagnostic=diagnostic
            )

            # Filter based on disease relevance
            if disease_key and diagnostic.disease_score < self.min_disease_score:
                continue

            # Optionally filter full-text-only matches
            if (self.require_title_abstract_match and
                diagnostic.field == MatchField.FULL_TEXT):
                continue

            scored_results.append(PrecisionSearchResult(
                rank=0,  # Will be set after sorting
                content=result.content,
                paper_title=result.metadata.get("paper_title", "Unknown"),
                section=result.metadata.get("section", "Unknown"),
                doi=result.metadata.get("doi", ""),
                year=result.metadata.get("year", ""),
                pmid=result.metadata.get("pmid", ""),
                final_score=final_score,
                disease_relevance=diagnostic.disease_score,
                field_match=diagnostic.field,
                diagnostic=diagnostic,
                metadata=result.metadata
            ))

        # Step 4: Sort by final score and assign ranks
        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        for i, result in enumerate(scored_results[:top_k], 1):
            result.rank = i

        final_results = scored_results[:top_k]

        # Build diagnostics
        diagnostics = SearchDiagnostics(
            query=query,
            detected_disease=disease_key,
            mesh_term=mesh_term,
            search_terms=search_terms,
            modifiers=modifiers,
            total_candidates=total_candidates,
            filtered_results=len(final_results),
            strategy_used="lexical_first_field_aware",
            explanation=self._build_explanation(
                disease_key=disease_key,
                mesh_term=mesh_term,
                total_candidates=total_candidates,
                filtered=len(final_results)
            )
        )

        return final_results, diagnostics

    def _get_lexical_candidates(
        self,
        query: str,
        top_k: int,
        section_filter: str | None = None
    ) -> list[SearchResult]:
        """Get candidates using BM25/lexical search."""
        where = {"section": section_filter} if section_filter else None

        # Use hybrid search but rely primarily on BM25 component
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            where=where,
            use_hybrid=True
        )

        return results

    def _analyze_match(
        self,
        content: str,
        metadata: dict,
        disease_key: Optional[str],
        search_terms: list[str],
        modifiers: list[str]
    ) -> MatchDiagnostic:
        """Analyze where and how the match occurred."""
        matched_terms = []
        field = MatchField.NONE
        disease_score = 0.0
        modifier_score = 0.0

        # Get paper title and determine best match location
        paper_title = metadata.get("paper_title", "").lower()
        section = metadata.get("section", "").lower()
        content_lower = content.lower()

        # Check if this is abstract section
        is_abstract = "abstract" in section

        # Check disease term matches using vocabulary
        if disease_key:
            # Use vocabulary's match scoring
            match_result = self.vocabulary.match_score(paper_title, disease_key)
            title_match = match_result["has_primary"] or match_result["score"] > 0.3

            if title_match:
                field = MatchField.TITLE
                disease_score = 0.9
                matched_terms.extend([m["term"] for m in match_result["matches"]])
            else:
                # Check abstract
                if is_abstract:
                    abstract_match = self.vocabulary.match_score(content, disease_key)
                    if abstract_match["has_primary"] or abstract_match["score"] > 0.3:
                        field = MatchField.ABSTRACT
                        disease_score = 0.7
                        matched_terms.extend([m["term"] for m in abstract_match["matches"]])

                # If not found in abstract, check full text
                if field == MatchField.NONE:
                    text_match = self.vocabulary.match_score(content, disease_key)
                    if text_match["score"] > 0.2:
                        field = MatchField.FULL_TEXT
                        disease_score = 0.3
                        matched_terms.extend([m["term"] for m in text_match["matches"]])
        else:
            # No disease key - use search terms directly
            for term in search_terms:
                term_lower = term.lower()
                if term_lower in paper_title:
                    field = MatchField.TITLE
                    disease_score = max(disease_score, 0.8)
                    matched_terms.append(term)
                elif term_lower in content_lower and is_abstract:
                    if field.value not in ["title"]:
                        field = MatchField.ABSTRACT
                    disease_score = max(disease_score, 0.6)
                    matched_terms.append(term)
                elif term_lower in content_lower:
                    if field == MatchField.NONE:
                        field = MatchField.FULL_TEXT
                    disease_score = max(disease_score, 0.3)
                    matched_terms.append(term)

        # Check modifier matches
        for modifier in modifiers:
            modifier_lower = modifier.lower()
            if modifier_lower in content_lower:
                modifier_score += 0.2
                matched_terms.append(modifier)

        modifier_score = min(1.0, modifier_score)

        # Build explanation
        field_weight = FIELD_WEIGHTS[field]
        explanation = self._build_match_explanation(
            field=field,
            matched_terms=matched_terms,
            disease_score=disease_score,
            disease_key=disease_key
        )

        return MatchDiagnostic(
            field=field,
            matched_terms=list(set(matched_terms)),
            disease_score=disease_score,
            modifier_score=modifier_score,
            field_weight=field_weight,
            explanation=explanation
        )

    def _calculate_final_score(
        self,
        base_score: float,
        diagnostic: MatchDiagnostic
    ) -> float:
        """Calculate final score combining base score with field-aware weighting."""
        # Normalize base score to 0-1
        normalized_base = base_score / 100.0

        # Combine scores
        # Disease relevance is most important (40%)
        # Field weight determines location quality (30%)
        # Base semantic similarity (20%)
        # Modifier matches (10%)
        final = (
            diagnostic.disease_score * 0.40 +
            diagnostic.field_weight * 0.30 +
            normalized_base * 0.20 +
            diagnostic.modifier_score * 0.10
        )

        return final * 100  # Return as percentage

    def _build_match_explanation(
        self,
        field: MatchField,
        matched_terms: list[str],
        disease_score: float,
        disease_key: Optional[str]
    ) -> str:
        """Build human-readable match explanation."""
        if field == MatchField.NONE:
            return "No disease term match found"

        terms_str = ", ".join(matched_terms[:5])  # Limit to 5 terms
        if len(matched_terms) > 5:
            terms_str += f" (+{len(matched_terms) - 5} more)"

        field_desc = {
            MatchField.MESH: "MeSH term match",
            MatchField.TITLE: "Title match",
            MatchField.ABSTRACT: "Abstract match",
            MatchField.FULL_TEXT: "Full text match only"
        }

        relevance = "high" if disease_score > 0.7 else "medium" if disease_score > 0.4 else "low"

        return f"{field_desc[field]} ({relevance} relevance): {terms_str}"

    def _build_explanation(
        self,
        disease_key: Optional[str],
        mesh_term: Optional[str],
        total_candidates: int,
        filtered: int
    ) -> str:
        """Build overall search explanation."""
        parts = []

        if disease_key:
            parts.append(f"Detected disease: '{disease_key}'")
            if mesh_term:
                parts.append(f"Using MeSH term: '{mesh_term}'")
        else:
            parts.append("No specific disease term detected - using keyword search")

        parts.append(f"Found {total_candidates} candidates, returned {filtered} after filtering")

        if self.require_title_abstract_match:
            parts.append("Filtered: requiring title/abstract match (excluded full-text-only matches)")

        return " | ".join(parts)

    def get_supported_diseases(self) -> list[dict]:
        """Get list of supported diseases with their MeSH info."""
        diseases = []
        for key, mesh in self.vocabulary.mesh_terms.items():
            diseases.append({
                "key": key,
                "mesh_term": mesh.primary,
                "mesh_id": mesh.mesh_id,
                "synonyms": mesh.synonyms,
                "abbreviations": mesh.abbreviations
            })
        return diseases


def search_with_precision(
    query: str,
    disease_domain: str | None = None,
    top_k: int = 10
) -> tuple[list[PrecisionSearchResult], SearchDiagnostics]:
    """
    Convenience function for precision search.

    Args:
        query: Search query
        disease_domain: Disease domain to search
        top_k: Number of results

    Returns:
        Tuple of (results, diagnostics)
    """
    searcher = PrecisionSearch(disease_domain=disease_domain)
    return searcher.search(query, top_k=top_k)
