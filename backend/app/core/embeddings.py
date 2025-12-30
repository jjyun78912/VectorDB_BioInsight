"""
Hybrid Embedding Module for Bio Papers.

Combines Dense (PubMedBERT) + Sparse (BM25) for optimal retrieval.
- Dense: Semantic similarity via PubMedBERT embeddings
- Sparse: Keyword matching via BM25 algorithm
"""
from typing import Optional
import re
import pickle
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from .config import EMBEDDING_MODEL, CHROMA_DIR


class PubMedBertEmbedder:
    """
    Embedding generator using PubMedBERT via Sentence-Transformers.

    Supported models:
    - pritamdeka/S-PubMedBert-MS-MARCO (recommended for retrieval)
    - NeuML/pubmedbert-base-embeddings
    - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32
    ):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model name for embeddings
            device: Device to run model on (auto-detected if None)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            print(f"Using device: {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            List of floats (embedding vector)
        """
        embedding = self.model.encode(
            text,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=False,
            show_progress_bar=show_progress
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.
        Some models use different encoding for queries vs documents.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        # For S-PubMedBert-MS-MARCO, query prefix can help
        if "MS-MARCO" in self.model_name:
            query = f"query: {query}"
        return self.embed_text(query)


# Singleton instance for convenience
_embedder: Optional[PubMedBertEmbedder] = None


def get_embedder(model_name: str = EMBEDDING_MODEL) -> PubMedBertEmbedder:
    """Get or create the singleton embedder instance."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = PubMedBertEmbedder(model_name=model_name)
    return _embedder


def embed_text(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    return get_embedder().embed_text(text)


def embed_texts(texts: list[str], show_progress: bool = True) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedder().embed_texts(texts, show_progress=show_progress)


def embed_query(query: str) -> list[float]:
    """Convenience function to embed a search query."""
    return get_embedder().embed_query(query)


# =============================================================================
# BM25 Sparse Search
# =============================================================================

class BM25Index:
    """
    BM25 sparse index for keyword-based retrieval.
    Complements dense embeddings for hybrid search.
    """

    def __init__(self, collection_name: str = "default"):
        self.collection_name = collection_name
        self.index_path = CHROMA_DIR / f"bm25_{collection_name}.pkl"
        self.bm25: Optional[BM25Okapi] = None
        self.documents: list[str] = []
        self.doc_ids: list[str] = []

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 (simple word tokenization)."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 2]
        return tokens

    def build_index(self, documents: list[str], doc_ids: list[str] = None):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
            doc_ids: Optional list of document IDs
        """
        self.documents = documents
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(len(documents))]

        # Tokenize all documents
        tokenized_docs = [self._tokenize(doc) for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"Built BM25 index with {len(documents)} documents")

    def save(self):
        """Save index to disk."""
        if self.bm25 is None:
            return

        data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "doc_ids": self.doc_ids
        }
        with open(self.index_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved BM25 index to {self.index_path}")

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.doc_ids = data["doc_ids"]
            print(f"Loaded BM25 index with {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if score > 0
                results.append((self.doc_ids[idx], float(scores[idx])))

        return results


# =============================================================================
# Hybrid Search (Dense + Sparse)
# =============================================================================

class HybridSearcher:
    """
    Hybrid search combining Dense (PubMedBERT) and Sparse (BM25) retrieval.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings.
    """

    def __init__(
        self,
        collection_name: str = "default",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid searcher.

        Args:
            collection_name: Name for the BM25 index
            dense_weight: Weight for dense (semantic) results (0-1)
            sparse_weight: Weight for sparse (BM25) results (0-1)
            rrf_k: RRF parameter (higher = smoother fusion)
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

        # Dense embedder
        self.dense_embedder = get_embedder()

        # Sparse BM25 index
        self.bm25_index = BM25Index(collection_name)

    def build_sparse_index(self, documents: list[str], doc_ids: list[str] = None):
        """Build the BM25 index for sparse search."""
        self.bm25_index.build_index(documents, doc_ids)
        self.bm25_index.save()

    def load_sparse_index(self) -> bool:
        """Load existing BM25 index."""
        return self.bm25_index.load()

    def get_dense_embedding(self, text: str) -> list[float]:
        """Get dense embedding for text."""
        return self.dense_embedder.embed_text(text)

    def get_query_embedding(self, query: str) -> list[float]:
        """Get dense embedding for query."""
        return self.dense_embedder.embed_query(query)

    def search_sparse(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Search using BM25 only."""
        return self.bm25_index.search(query, top_k)

    def fuse_results(
        self,
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Fuse dense and sparse results using Reciprocal Rank Fusion (RRF).

        Args:
            dense_results: List of (doc_id, score) from dense search
            sparse_results: List of (doc_id, score) from sparse search
            top_k: Number of final results

        Returns:
            List of (doc_id, fused_score) tuples
        """
        scores = {}

        # Add dense scores with RRF
        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Add sparse scores with RRF
        for rank, (doc_id, _) in enumerate(sparse_results):
            rrf_score = self.sparse_weight / (self.rrf_k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + rrf_score

        # Sort by fused score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:top_k]


# Singleton hybrid searcher
_hybrid_searcher: Optional[HybridSearcher] = None


def get_hybrid_searcher(
    collection_name: str = "default",
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4
) -> HybridSearcher:
    """Get or create hybrid searcher instance."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher(
            collection_name=collection_name,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
    return _hybrid_searcher
