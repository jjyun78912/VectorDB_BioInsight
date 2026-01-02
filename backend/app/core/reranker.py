"""
Cross-encoder based Re-ranker for improving search result relevance.

Re-ranking uses a cross-encoder model that processes query-document pairs together,
allowing for more accurate relevance scoring than bi-encoder (embedding) approaches.

Supported models:
- cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, general purpose)
- cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
- BAAI/bge-reranker-base (good for scientific text)
- BAAI/bge-reranker-large (best quality, slower)
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class RerankedResult:
    """A re-ranked search result with cross-encoder score."""
    content: str
    metadata: dict
    original_score: float  # Original retrieval score
    rerank_score: float    # Cross-encoder score
    final_score: float     # Combined score


class CrossEncoderReranker:
    """
    Cross-encoder based re-ranker for search results.

    Uses a cross-encoder model to compute query-document relevance scores,
    which are more accurate than embedding similarity for ranking.
    """

    # Available models with their characteristics
    AVAILABLE_MODELS = {
        "ms-marco-mini": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "description": "Fast, general purpose (22M params)",
            "speed": "fast"
        },
        "ms-marco-medium": {
            "name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "description": "Balanced speed/quality (33M params)",
            "speed": "medium"
        },
        "bge-base": {
            "name": "BAAI/bge-reranker-base",
            "description": "Good for scientific text (278M params)",
            "speed": "medium"
        },
        "bge-large": {
            "name": "BAAI/bge-reranker-large",
            "description": "Best quality, slower (560M params)",
            "speed": "slow"
        }
    }

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        original_weight: float = 0.3,
        rerank_weight: float = 0.7
    ):
        """
        Initialize the cross-encoder re-ranker.

        Args:
            model_name: Cross-encoder model name or alias
            device: Device to use (cuda, mps, cpu). Auto-detected if None.
            batch_size: Batch size for processing
            original_weight: Weight for original retrieval score (0-1)
            rerank_weight: Weight for re-rank score (0-1)
        """
        # Resolve model alias
        if model_name in self.AVAILABLE_MODELS:
            model_name = self.AVAILABLE_MODELS[model_name]["name"]

        self.model_name = model_name
        self.batch_size = batch_size
        self.original_weight = original_weight
        self.rerank_weight = rerank_weight

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Lazy load model
        self._model = None

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            print(f"Loading re-ranker model: {self.model_name} on {self.device}")
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def compute_scores(
        self,
        query: str,
        documents: list[str],
        show_progress: bool = False
    ) -> list[float]:
        """
        Compute cross-encoder relevance scores for query-document pairs.

        Args:
            query: Search query
            documents: List of document texts
            show_progress: Whether to show progress bar

        Returns:
            List of relevance scores (higher is better)
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=show_progress
        )

        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)

    def rerank(
        self,
        query: str,
        results: list[dict],
        content_key: str = "content",
        score_key: str = "relevance_score",
        top_k: Optional[int] = None
    ) -> list[RerankedResult]:
        """
        Re-rank search results using cross-encoder scores.

        Args:
            query: Search query
            results: List of search results (dicts with content and score)
            content_key: Key for document content in result dict
            score_key: Key for original score in result dict
            top_k: Number of results to return (None = all)

        Returns:
            List of RerankedResult objects sorted by final score
        """
        if not results:
            return []

        # Extract documents
        documents = [r.get(content_key, "") for r in results]
        original_scores = [r.get(score_key, 0.0) for r in results]

        # Compute cross-encoder scores
        rerank_scores = self.compute_scores(query, documents)

        # Normalize rerank scores to 0-100 range
        min_score = min(rerank_scores) if rerank_scores else 0
        max_score = max(rerank_scores) if rerank_scores else 1
        score_range = max_score - min_score if max_score > min_score else 1

        normalized_rerank = [
            ((s - min_score) / score_range) * 100
            for s in rerank_scores
        ]

        # Combine scores
        reranked = []
        for i, result in enumerate(results):
            final_score = (
                self.original_weight * original_scores[i] +
                self.rerank_weight * normalized_rerank[i]
            )

            reranked.append(RerankedResult(
                content=result.get(content_key, ""),
                metadata=result.get("metadata", {}),
                original_score=original_scores[i],
                rerank_score=normalized_rerank[i],
                final_score=final_score
            ))

        # Sort by final score (descending)
        reranked.sort(key=lambda x: x.final_score, reverse=True)

        # Apply top_k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def rerank_search_results(
        self,
        query: str,
        search_results: list,
        top_k: Optional[int] = None
    ) -> list:
        """
        Re-rank SearchResult objects from BioVectorStore.

        Args:
            query: Search query
            search_results: List of SearchResult objects
            top_k: Number of results to return

        Returns:
            List of SearchResult objects with updated relevance_score
        """
        if not search_results:
            return []

        # Extract documents and scores
        documents = [r.content for r in search_results]
        original_scores = [r.relevance_score for r in search_results]

        # Compute cross-encoder scores
        rerank_scores = self.compute_scores(query, documents)

        # Normalize rerank scores to 0-100 range
        min_score = min(rerank_scores) if rerank_scores else 0
        max_score = max(rerank_scores) if rerank_scores else 1
        score_range = max_score - min_score if max_score > min_score else 1

        normalized_rerank = [
            ((s - min_score) / score_range) * 100
            for s in rerank_scores
        ]

        # Create result pairs with combined scores
        scored_results = []
        for i, result in enumerate(search_results):
            final_score = (
                self.original_weight * original_scores[i] +
                self.rerank_weight * normalized_rerank[i]
            )
            scored_results.append((result, final_score, normalized_rerank[i]))

        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k
        if top_k is not None:
            scored_results = scored_results[:top_k]

        # Update relevance scores and return
        from .vector_store import SearchResult

        reranked_results = []
        for result, final_score, rerank_score in scored_results:
            # Create new SearchResult with updated score
            reranked_results.append(SearchResult(
                content=result.content,
                metadata={
                    **result.metadata,
                    "original_score": result.relevance_score,
                    "rerank_score": rerank_score
                },
                distance=result.distance,
                relevance_score=final_score
            ))

        return reranked_results


# Global reranker instance (lazy loaded)
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: Optional[str] = None,
    force_new: bool = False
) -> CrossEncoderReranker:
    """
    Get or create the global reranker instance.

    Args:
        model_name: Model to use (uses config default if None)
        force_new: Force creation of new instance

    Returns:
        CrossEncoderReranker instance
    """
    global _reranker

    if _reranker is None or force_new:
        from .config import (
            RERANKER_MODEL,
            RERANKER_ORIGINAL_WEIGHT,
            RERANKER_WEIGHT
        )

        _reranker = CrossEncoderReranker(
            model_name=model_name or RERANKER_MODEL,
            original_weight=RERANKER_ORIGINAL_WEIGHT,
            rerank_weight=RERANKER_WEIGHT
        )

    return _reranker
