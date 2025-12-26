"""
Similarity Search Interface for Bio Papers.

Provides semantic search capabilities with filtering and ranking.
"""
from dataclasses import dataclass
from typing import Optional

from .vector_store import BioVectorStore, SearchResult, create_vector_store
from .embeddings import get_embedder
from .config import TOP_K_RESULTS


@dataclass
class EnrichedSearchResult:
    """Search result with additional context."""
    rank: int
    content: str
    paper_title: str
    section: str
    parent_section: Optional[str]
    doi: str
    year: str
    keywords: list[str]
    relevance_score: float
    chunk_index: int


class BioPaperSearch:
    """
    Semantic search interface for biomedical papers.

    Features:
    - Full-text semantic search
    - Section-filtered search
    - Paper-specific search
    - Keyword filtering
    - Results ranking
    """

    def __init__(
        self,
        disease_domain: str,
        vector_store: BioVectorStore | None = None
    ):
        """
        Initialize the search interface.

        Args:
            disease_domain: Disease domain to search (e.g., "pheochromocytoma")
            vector_store: Optional pre-initialized vector store
        """
        self.disease_domain = disease_domain
        self.vector_store = vector_store or create_vector_store(disease_domain=disease_domain)
        self.embedder = get_embedder()

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        section_filter: str | None = None,
        year_filter: str | None = None,
        min_score: float = 0.0
    ) -> list[EnrichedSearchResult]:
        """
        Perform semantic search.

        Args:
            query: Natural language search query
            top_k: Maximum number of results
            section_filter: Filter by section (e.g., "Methods", "Results")
            year_filter: Filter by publication year
            min_score: Minimum relevance score (0-1)

        Returns:
            List of enriched search results
        """
        # Build where clause
        where = {}
        if section_filter:
            where["section"] = section_filter
        if year_filter:
            where["year"] = year_filter

        # Execute search
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            where=where if where else None
        )

        # Filter by minimum score and enrich
        enriched = []
        for rank, result in enumerate(results, 1):
            if result.relevance_score >= min_score:
                enriched.append(self._enrich_result(result, rank))

        return enriched

    def search_methods(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS
    ) -> list[EnrichedSearchResult]:
        """Search specifically in Methods/Materials sections."""
        return self.search(query, top_k, section_filter="Methods") + \
               self.search(query, top_k, section_filter="Materials and Methods")

    def search_results_discussion(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS
    ) -> list[EnrichedSearchResult]:
        """Search in Results and Discussion sections."""
        results = self.search(query, top_k, section_filter="Results")
        results += self.search(query, top_k, section_filter="Discussion")
        # Re-rank by score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]

    def find_similar_chunks(
        self,
        text: str,
        top_k: int = TOP_K_RESULTS,
        exclude_same_paper: str | None = None
    ) -> list[EnrichedSearchResult]:
        """
        Find chunks similar to given text.
        Useful for finding related content across papers.

        Args:
            text: Reference text to find similar content for
            top_k: Number of results
            exclude_same_paper: Paper title to exclude from results

        Returns:
            Similar content from other papers
        """
        results = self.search(text, top_k=top_k * 2)  # Get extra to allow filtering

        if exclude_same_paper:
            results = [r for r in results if r.paper_title != exclude_same_paper]

        return results[:top_k]

    def get_paper_sections(
        self,
        paper_title: str
    ) -> dict[str, list[str]]:
        """
        Get all sections from a specific paper.

        Returns:
            Dict mapping section names to list of chunk contents
        """
        results = self.vector_store.collection.get(
            where={"paper_title": paper_title},
            include=["documents", "metadatas"]
        )

        sections = {}
        for doc, meta in zip(results["documents"], results["metadatas"]):
            section = meta.get("section", "Unknown")
            if section not in sections:
                sections[section] = []
            sections[section].append(doc)

        return sections

    def _enrich_result(self, result: SearchResult, rank: int) -> EnrichedSearchResult:
        """Convert SearchResult to EnrichedSearchResult."""
        keywords = result.metadata.get("keywords", "")
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]

        return EnrichedSearchResult(
            rank=rank,
            content=result.content,
            paper_title=result.metadata.get("paper_title", "Unknown"),
            section=result.metadata.get("section", "Unknown"),
            parent_section=result.metadata.get("parent_section"),
            doi=result.metadata.get("doi", ""),
            year=result.metadata.get("year", ""),
            keywords=keywords,
            relevance_score=result.relevance_score,
            chunk_index=result.metadata.get("chunk_index", 0)
        )

    def format_results(
        self,
        results: list[EnrichedSearchResult],
        show_content: bool = True,
        max_content_length: int = 300
    ) -> str:
        """Format results for display."""
        if not results:
            return "No results found."

        output = []
        for r in results:
            lines = [
                f"[{r.rank}] Score: {r.relevance_score:.3f}",
                f"    Paper: {r.paper_title}",
                f"    Section: {r.section}",
            ]
            if r.parent_section:
                lines.append(f"    Subsection of: {r.parent_section}")
            if r.doi:
                lines.append(f"    DOI: {r.doi}")
            if r.year:
                lines.append(f"    Year: {r.year}")
            if show_content:
                content = r.content[:max_content_length]
                if len(r.content) > max_content_length:
                    content += "..."
                lines.append(f"    Content: {content}")
            output.append("\n".join(lines))

        return "\n\n".join(output)


def search_pheochromocytoma(
    query: str,
    top_k: int = TOP_K_RESULTS
) -> list[EnrichedSearchResult]:
    """Convenience function to search Pheochromocytoma papers."""
    searcher = BioPaperSearch(disease_domain="pheochromocytoma")
    return searcher.search(query, top_k)


def create_searcher(disease_domain: str) -> BioPaperSearch:
    """Create a search interface for any disease domain."""
    return BioPaperSearch(disease_domain=disease_domain)
