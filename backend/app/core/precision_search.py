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
        require_title_abstract_match: bool = False
    ):
        """
        Initialize precision search.

        Args:
            vector_store: Vector store to search
            disease_domain: Disease domain for collection
            min_disease_score: Minimum disease relevance score (0-1)
            require_title_abstract_match: If True, filter out full-text-only matches (default: False for more results)
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

        # Check if query is disease-name-only (no modifiers)
        # If so, return ALL papers from the collection
        is_disease_only_query = disease_key and not remaining_query.strip()

        if is_disease_only_query:
            return self._list_all_papers(disease_key, top_k, section_filter)

        # Check if query has specific keywords (e.g., gene names, pathways)
        # If so, prioritize papers containing those keywords
        keywords = remaining_query.split() if remaining_query else []
        if disease_key and keywords:
            return self._search_with_keywords(disease_key, keywords, top_k, section_filter)

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

    def _search_with_keywords(
        self,
        disease_key: str,
        keywords: list[str],
        top_k: int,
        section_filter: str | None = None
    ) -> tuple[list[PrecisionSearchResult], SearchDiagnostics]:
        """
        Search for papers containing specific keywords within the disease collection.

        When user searches "breast cancer BRCA1" or "유방암 BRCA1",
        prioritize papers that contain both the disease AND the keyword(s).

        Scoring:
        - Papers with keyword in title: 100 points
        - Papers with keyword in abstract: 80 points
        - Papers with keyword in full text: 60 points
        - Multiple keywords matched: additional 20 points each
        """
        # Get all documents from the collection
        all_docs = self.vector_store.collection.get(include=['metadatas', 'documents'])

        # Score papers by keyword match
        papers_scores: dict[str, dict] = {}

        for i, meta in enumerate(all_docs['metadatas']):
            pmid = meta.get('pmid', '')
            if not pmid:
                continue

            # Apply section filter if specified
            if section_filter and meta.get('section', '').lower() != section_filter.lower():
                continue

            paper_title = meta.get('paper_title', '').lower()
            section = meta.get('section', '').lower()
            content = all_docs['documents'][i].lower() if all_docs['documents'] else ''
            is_abstract = 'abstract' in section

            # Initialize paper entry
            if pmid not in papers_scores:
                papers_scores[pmid] = {
                    'pmid': pmid,
                    'paper_title': meta.get('paper_title', 'Unknown'),
                    'year': meta.get('year', ''),
                    'doi': meta.get('doi', ''),
                    'section': meta.get('section', 'Unknown'),
                    'content': all_docs['documents'][i][:500] if all_docs['documents'] else '',
                    'metadata': meta,
                    'score': 0,
                    'matched_keywords': [],
                    'match_locations': []
                }

            # Score by keyword matches - check ALL chunks for this paper
            # We need to track which keywords have been matched for this paper
            for keyword in keywords:
                kw_lower = keyword.lower()

                # Skip if this keyword already matched for this paper
                if keyword in papers_scores[pmid]['matched_keywords']:
                    continue

                # Check title match (highest priority)
                if kw_lower in paper_title:
                    papers_scores[pmid]['score'] += 100
                    papers_scores[pmid]['matched_keywords'].append(keyword)
                    if 'title' not in papers_scores[pmid]['match_locations']:
                        papers_scores[pmid]['match_locations'].append('title')

                # Check abstract match
                elif is_abstract and kw_lower in content:
                    papers_scores[pmid]['score'] += 80
                    papers_scores[pmid]['matched_keywords'].append(keyword)
                    if 'abstract' not in papers_scores[pmid]['match_locations']:
                        papers_scores[pmid]['match_locations'].append('abstract')

                # Check full text match
                elif kw_lower in content:
                    papers_scores[pmid]['score'] += 60
                    papers_scores[pmid]['matched_keywords'].append(keyword)
                    if 'full_text' not in papers_scores[pmid]['match_locations']:
                        papers_scores[pmid]['match_locations'].append('full_text')

            # Bonus for matching multiple keywords
            n_matched = len(papers_scores[pmid]['matched_keywords'])
            if n_matched > 1:
                papers_scores[pmid]['score'] += (n_matched - 1) * 20

        # Filter papers that match ALL keywords (AND logic, not OR)
        # A paper must contain ALL keywords to be considered relevant
        matched_papers = [
            p for p in papers_scores.values()
            if len(p['matched_keywords']) == len(keywords)  # Must match ALL keywords
        ]

        sorted_papers = sorted(
            matched_papers,
            key=lambda x: (x['score'], x.get('year', '0000')),
            reverse=True
        )

        # Limit to top_k
        limited_papers = sorted_papers[:top_k]

        # Build results
        results = []
        for i, paper in enumerate(limited_papers, 1):
            match_field = MatchField.FULL_TEXT
            if 'title' in paper['match_locations']:
                match_field = MatchField.TITLE
            elif 'abstract' in paper['match_locations']:
                match_field = MatchField.ABSTRACT

            diagnostic = MatchDiagnostic(
                field=match_field,
                matched_terms=paper['matched_keywords'],
                disease_score=1.0,
                modifier_score=len(paper['matched_keywords']) / len(keywords),
                field_weight=FIELD_WEIGHTS[match_field],
                explanation=f"Keywords [{', '.join(paper['matched_keywords'])}] found in {', '.join(set(paper['match_locations']))}"
            )

            results.append(PrecisionSearchResult(
                rank=i,
                content=paper['content'],
                paper_title=paper['paper_title'],
                section=paper['section'],
                doi=paper['doi'],
                year=paper['year'],
                pmid=paper['pmid'],
                final_score=paper['score'],
                disease_relevance=1.0,
                field_match=match_field,
                diagnostic=diagnostic,
                metadata=paper['metadata']
            ))

        # Build diagnostics
        mesh_info = self.vocabulary.mesh_terms.get(disease_key)
        mesh_term = mesh_info.primary if mesh_info else None

        total_papers = len([p for p in papers_scores.values()])

        diagnostics = SearchDiagnostics(
            query=f"{disease_key} {' '.join(keywords)}",
            detected_disease=disease_key,
            mesh_term=mesh_term,
            search_terms=[disease_key] + keywords,
            modifiers=keywords,
            total_candidates=total_papers,
            filtered_results=len(results),
            strategy_used="keyword_search",
            explanation=f"Found {len(matched_papers)} papers matching keywords [{', '.join(keywords)}] in {disease_key} collection (showing top {len(results)})"
        )

        return results, diagnostics

    def _list_all_papers(
        self,
        disease_key: str,
        top_k: int,
        section_filter: str | None = None
    ) -> tuple[list[PrecisionSearchResult], SearchDiagnostics]:
        """
        List all papers in the collection for disease-name-only queries.

        When user searches only "breast cancer" without additional keywords,
        return ALL papers in the breast_cancer collection, aggregated by PMID.
        """
        # Get all documents from the collection
        all_docs = self.vector_store.collection.get(include=['metadatas', 'documents'])

        # Aggregate by PMID to get unique papers
        papers_by_pmid: dict[str, dict] = {}

        for i, meta in enumerate(all_docs['metadatas']):
            pmid = meta.get('pmid', '')
            if not pmid:
                continue

            # Apply section filter if specified
            if section_filter and meta.get('section', '').lower() != section_filter.lower():
                continue

            if pmid not in papers_by_pmid:
                papers_by_pmid[pmid] = {
                    'pmid': pmid,
                    'paper_title': meta.get('paper_title', 'Unknown'),
                    'year': meta.get('year', ''),
                    'doi': meta.get('doi', ''),
                    'section': meta.get('section', 'Unknown'),
                    'content': all_docs['documents'][i][:500] if all_docs['documents'] else '',
                    'metadata': meta
                }

        # Sort by year (newest first), then by title
        sorted_papers = sorted(
            papers_by_pmid.values(),
            key=lambda x: (x.get('year', '0000'), x.get('paper_title', '')),
            reverse=True
        )

        # Limit to top_k
        limited_papers = sorted_papers[:top_k]

        # Build results
        results = []
        for i, paper in enumerate(limited_papers, 1):
            diagnostic = MatchDiagnostic(
                field=MatchField.TITLE,
                matched_terms=[disease_key],
                disease_score=1.0,
                field_weight=1.0,
                explanation=f"Paper from {disease_key} collection"
            )

            results.append(PrecisionSearchResult(
                rank=i,
                content=paper['content'],
                paper_title=paper['paper_title'],
                section=paper['section'],
                doi=paper['doi'],
                year=paper['year'],
                pmid=paper['pmid'],
                final_score=100.0,  # All papers in collection are relevant
                disease_relevance=1.0,
                field_match=MatchField.TITLE,
                diagnostic=diagnostic,
                metadata=paper['metadata']
            ))

        # Build diagnostics
        mesh_info = self.vocabulary.mesh_terms.get(disease_key)
        mesh_term = mesh_info.primary if mesh_info else None

        diagnostics = SearchDiagnostics(
            query=disease_key,
            detected_disease=disease_key,
            mesh_term=mesh_term,
            search_terms=[disease_key],
            modifiers=[],
            total_candidates=len(papers_by_pmid),
            filtered_results=len(results),
            strategy_used="list_all_papers",
            explanation=f"Listing all {len(papers_by_pmid)} papers in {disease_key} collection (showing top {len(results)})"
        )

        return results, diagnostics


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
