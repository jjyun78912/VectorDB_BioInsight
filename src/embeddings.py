"""
PubMedBERT Embedding Module for Bio Papers.

Uses Sentence-Transformers with PubMedBERT for domain-specific embeddings.
PubMedBERT is pre-trained on PubMed abstracts and PMC full-text articles,
making it ideal for biomedical text.
"""
from typing import Optional
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import EMBEDDING_MODEL


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
