"""
Core module - Business logic for BioInsight RAG system.

This module contains:
- config: Configuration settings
- embeddings: PubMedBERT embeddings and BM25 indexing
- vector_store: ChromaDB vector store with hybrid search
- text_splitter: Bio-paper aware text chunking
- pdf_parser: PDF parsing with section detection
- indexer: Document indexing pipeline
- rag_pipeline: Retrieval-Augmented Generation
- precision_search: MeSH vocabulary-aware precision search
- medical_vocabulary: MeSH term mapping
- paper_agent: Per-paper chat agent
- summarizer: Paper summarization
- search: Search utilities
- recommender: Paper recommendations
- corpus_builder: PubMed corpus building
- web_crawler_agent: Real-time paper fetching
- playwright_crawler: Browser-based crawling
"""

from .config import (
    BASE_DIR,
    DATA_DIR,
    PAPERS_DIR,
    CHROMA_DIR,
    GEMINI_API_KEY,
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BIO_PAPER_SECTIONS,
    METHODS_SUBSECTIONS,
    EXCLUDE_SECTIONS,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    ENABLE_HYBRID_SEARCH,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
)

from .embeddings import (
    PubMedBertEmbedder,
    get_embedder,
    embed_text,
    embed_texts,
    embed_query,
    BM25Index,
    HybridSearcher,
    get_hybrid_searcher,
)

from .vector_store import (
    SearchResult,
    BioVectorStore,
    create_vector_store,
)

from .text_splitter import (
    TextChunk,
    BioPaperSplitter,
)

__all__ = [
    # Config
    "BASE_DIR",
    "DATA_DIR",
    "PAPERS_DIR",
    "CHROMA_DIR",
    "GEMINI_API_KEY",
    "EMBEDDING_MODEL",
    "COLLECTION_NAME",
    # Embeddings
    "PubMedBertEmbedder",
    "get_embedder",
    "embed_text",
    "embed_texts",
    "embed_query",
    "BM25Index",
    "HybridSearcher",
    # Vector Store
    "SearchResult",
    "BioVectorStore",
    "create_vector_store",
    # Text Splitter
    "TextChunk",
    "BioPaperSplitter",
]
