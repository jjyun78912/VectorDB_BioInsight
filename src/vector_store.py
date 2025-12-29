"""
ChromaDB Vector Store for Bio Papers with Hybrid Search.

Features:
- Persistent storage for paper embeddings
- Rich metadata support (section, DOI, keywords, etc.)
- Disease-domain specific collections
- Hybrid search (Dense + Sparse BM25)
- Similarity search with metadata filtering
"""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from .config import CHROMA_DIR, COLLECTION_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from .embeddings import PubMedBertEmbedder, get_embedder, BM25Index, HybridSearcher
from .text_splitter import TextChunk


@dataclass
class SearchResult:
    """A search result with content and metadata."""
    content: str
    metadata: dict
    distance: float  # Lower is better (L2 distance)
    relevance_score: float  # Higher is better (0-100%, intuitive percentage)


class BioVectorStore:
    """
    ChromaDB-based vector store for biomedical papers with Hybrid Search.

    Supports:
    - Multiple disease-specific collections
    - Rich metadata filtering
    - Persistent storage
    - Batch operations
    - Hybrid search (Dense PubMedBERT + Sparse BM25)
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | Path = CHROMA_DIR,
        embedder: PubMedBertEmbedder | None = None,
        disease_domain: str | None = None,
        enable_hybrid: bool = True,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedder: Embedding model (uses default PubMedBERT if None)
            disease_domain: Optional disease domain for collection naming
            enable_hybrid: Enable hybrid search (Dense + BM25)
            dense_weight: Weight for dense search results (0-1)
            sparse_weight: Weight for sparse/BM25 results (0-1)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or get_embedder()

        # Add disease domain to collection name if specified
        if disease_domain:
            collection_name = f"{collection_name}_{disease_domain.lower().replace(' ', '_')}"
        self.collection_name = collection_name
        self.disease_domain = disease_domain

        # Hybrid search settings
        self.enable_hybrid = enable_hybrid
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Initialize BM25 index for hybrid search
        self._bm25_index: Optional[BM25Index] = None
        if enable_hybrid:
            self._bm25_index = BM25Index(collection_name)
            self._bm25_index.load()  # Try to load existing index

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": f"Bio papers vector store",
                "disease_domain": disease_domain or "general",
                "embedding_model": self.embedder.model_name,
                "embedding_dimension": self.embedder.embedding_dimension,
                "hybrid_enabled": str(enable_hybrid),
            }
        )

    @property
    def collection(self):
        """Get the ChromaDB collection."""
        return self._collection

    @property
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self._collection.count()

    def add_chunks(
        self,
        chunks: list[TextChunk],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> int:
        """
        Add text chunks to the vector store.

        Args:
            chunks: List of TextChunk objects
            batch_size: Batch size for adding documents
            show_progress: Whether to show progress bar

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        total_added = 0
        all_documents = []
        all_ids = []

        iterator = range(0, len(chunks), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Adding to VectorDB")

        for i in iterator:
            batch = chunks[i:i + batch_size]

            # Prepare batch data
            ids = [f"chunk_{self.count + i + j}" for j in range(len(batch))]
            documents = [chunk.content for chunk in batch]
            metadatas = [self._prepare_metadata(chunk.metadata) for chunk in batch]

            # Generate embeddings
            embeddings = self.embedder.embed_texts(documents, show_progress=False)

            # Add to collection
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            # Collect for BM25 index
            all_documents.extend(documents)
            all_ids.extend(ids)

            total_added += len(batch)

        # Build/update BM25 index for hybrid search
        if self.enable_hybrid and self._bm25_index and all_documents:
            self._rebuild_bm25_index()

        return total_added

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from all documents in collection."""
        if not self._bm25_index:
            return

        # Get all documents from ChromaDB
        results = self._collection.get(include=["documents"], limit=50000)

        if results["ids"] and results["documents"]:
            self._bm25_index.build_index(
                documents=results["documents"],
                doc_ids=results["ids"]
            )
            self._bm25_index.save()
            print(f"Rebuilt BM25 index with {len(results['ids'])} documents")

    def _prepare_metadata(self, metadata: dict) -> dict:
        """
        Prepare metadata for ChromaDB storage.
        ChromaDB only supports str, int, float, bool types.
        """
        prepared = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                prepared[key] = ", ".join(str(v) for v in value)
            else:
                prepared[key] = str(value)
        return prepared

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        where: dict | None = None,
        where_document: dict | None = None,
        include_embeddings: bool = False,
        use_hybrid: bool = True
    ) -> list[SearchResult]:
        """
        Search for similar documents using hybrid search (Dense + BM25).

        Args:
            query: Search query text
            top_k: Number of results to return
            where: Metadata filter (e.g., {"section": "Methods"})
            where_document: Document content filter
            include_embeddings: Whether to include embeddings in results
            use_hybrid: Use hybrid search if enabled (default True)

        Returns:
            List of SearchResult objects
        """
        # Use hybrid search if enabled and no filters
        if use_hybrid and self.enable_hybrid and self._bm25_index and where is None:
            return self._hybrid_search(query, top_k)

        # Fall back to dense-only search
        return self._dense_search(query, top_k, where, where_document, include_embeddings)

    def _dense_search(
        self,
        query: str,
        top_k: int,
        where: dict | None = None,
        where_document: dict | None = None,
        include_embeddings: bool = False
    ) -> list[SearchResult]:
        """Dense (embedding-based) search only."""
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Build include list
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        # Query the collection
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=include
        )

        # Convert to SearchResult objects
        return self._process_dense_results(results)

    def _hybrid_search(self, query: str, top_k: int) -> list[SearchResult]:
        """
        Hybrid search combining Dense (ChromaDB) and Sparse (BM25) results.
        Uses Reciprocal Rank Fusion (RRF) to combine rankings.
        """
        # 1. Dense search (get more results for fusion)
        query_embedding = self.embedder.embed_query(query)
        dense_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )

        # 2. Sparse BM25 search
        sparse_results = self._bm25_index.search(query, top_k=top_k * 2)

        # 3. Fuse results using RRF
        rrf_k = 60  # RRF parameter

        # Build dense rankings: doc_id -> (rank, distance, content, metadata)
        dense_data = {}
        if dense_results["ids"] and dense_results["ids"][0]:
            for rank, doc_id in enumerate(dense_results["ids"][0]):
                dense_data[doc_id] = {
                    "rank": rank,
                    "distance": dense_results["distances"][0][rank],
                    "content": dense_results["documents"][0][rank],
                    "metadata": dense_results["metadatas"][0][rank] if dense_results["metadatas"] else {}
                }

        # Build sparse rankings: doc_id -> rank
        sparse_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(sparse_results)}

        # Calculate RRF scores
        rrf_scores = {}
        all_doc_ids = set(dense_data.keys()) | set(sparse_ranks.keys())

        for doc_id in all_doc_ids:
            score = 0.0

            # Dense contribution
            if doc_id in dense_data:
                rank = dense_data[doc_id]["rank"]
                score += self.dense_weight / (rrf_k + rank + 1)

            # Sparse contribution
            if doc_id in sparse_ranks:
                rank = sparse_ranks[doc_id]
                score += self.sparse_weight / (rrf_k + rank + 1)

            rrf_scores[doc_id] = score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # Build final results
        search_results = []
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

        for doc_id in sorted_ids:
            # Get document data
            if doc_id in dense_data:
                data = dense_data[doc_id]
                content = data["content"]
                metadata = data["metadata"]
                distance = data["distance"]
            else:
                # Need to fetch from ChromaDB
                fetch_result = self._collection.get(ids=[doc_id], include=["documents", "metadatas"])
                if fetch_result["documents"]:
                    content = fetch_result["documents"][0]
                    metadata = fetch_result["metadatas"][0] if fetch_result["metadatas"] else {}
                    distance = 1.5  # Default distance for BM25-only matches
                else:
                    continue

            # Convert RRF score to percentage (0-100)
            relevance_score = (rrf_scores[doc_id] / max_rrf) * 100 if max_rrf > 0 else 50.0

            search_results.append(SearchResult(
                content=content,
                metadata=metadata,
                distance=distance,
                relevance_score=max(0, min(100, relevance_score))
            ))

        return search_results

    def _process_dense_results(self, results: dict) -> list[SearchResult]:
        """Process ChromaDB dense search results into SearchResult objects."""
        search_results = []
        if results["ids"] and results["ids"][0]:
            # Get min/max distances for normalization
            distances = results["distances"][0] if results["distances"] else []
            max_distance = max(distances) if distances else 1.0
            min_distance = min(distances) if distances else 0.0

            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0

                # Convert L2 distance to intuitive percentage score (0-100%)
                if max_distance > min_distance:
                    normalized = (max_distance - distance) / (max_distance - min_distance)
                    relevance_score = normalized * 100
                else:
                    relevance_score = 80.0

                abs_relevance = max(0, min(100, (1 - distance / 3.0) * 100))
                final_score = (relevance_score * 0.6) + (abs_relevance * 0.4)

                search_results.append(SearchResult(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    distance=distance,
                    relevance_score=max(0, min(100, final_score))
                ))

        return search_results

    def search_by_section(
        self,
        query: str,
        section: str,
        top_k: int = TOP_K_RESULTS
    ) -> list[SearchResult]:
        """Search within a specific section."""
        return self.search(query, top_k=top_k, where={"section": section})

    def search_by_paper(
        self,
        query: str,
        paper_title: str,
        top_k: int = TOP_K_RESULTS
    ) -> list[SearchResult]:
        """Search within a specific paper."""
        return self.search(query, top_k=top_k, where={"paper_title": paper_title})

    def get_all_papers(self) -> list[dict]:
        """Get list of all indexed papers with their metadata."""
        # Get all documents (limited to prevent memory issues)
        results = self._collection.get(
            include=["metadatas"],
            limit=10000
        )

        # Extract unique papers
        papers = {}
        for metadata in results["metadatas"]:
            title = metadata.get("paper_title", "Unknown")
            if title not in papers:
                papers[title] = {
                    "title": title,
                    "doi": metadata.get("doi", ""),
                    "year": metadata.get("year", ""),
                    "keywords": metadata.get("keywords", ""),
                    "source_file": metadata.get("source_file", ""),
                }

        return list(papers.values())

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        papers = self.get_all_papers()

        # Count chunks per section
        results = self._collection.get(include=["metadatas"], limit=10000)
        section_counts = {}
        for metadata in results["metadatas"]:
            section = metadata.get("section", "Unknown")
            section_counts[section] = section_counts.get(section, 0) + 1

        return {
            "collection_name": self.collection_name,
            "disease_domain": self.disease_domain,
            "total_chunks": self.count,
            "total_papers": len(papers),
            "chunks_by_section": section_counts,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dimension,
        }

    def delete_paper(self, paper_title: str) -> int:
        """Delete all chunks from a specific paper."""
        # Get IDs of chunks from this paper
        results = self._collection.get(
            where={"paper_title": paper_title},
            include=["metadatas"]
        )

        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def reset(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={
                "description": f"Bio papers vector store",
                "disease_domain": self.disease_domain or "general",
                "embedding_model": self.embedder.model_name,
            }
        )


def create_vector_store(
    disease_domain: str | None = None,
    collection_name: str = COLLECTION_NAME
) -> BioVectorStore:
    """Convenience function to create a vector store."""
    return BioVectorStore(
        collection_name=collection_name,
        disease_domain=disease_domain
    )
