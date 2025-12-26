"""
ChromaDB Vector Store for Bio Papers.

Features:
- Persistent storage for paper embeddings
- Rich metadata support (section, DOI, keywords, etc.)
- Disease-domain specific collections
- Similarity search with metadata filtering
"""
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from .config import CHROMA_DIR, COLLECTION_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD
from .embeddings import PubMedBertEmbedder, get_embedder
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
    ChromaDB-based vector store for biomedical papers.

    Supports:
    - Multiple disease-specific collections
    - Rich metadata filtering
    - Persistent storage
    - Batch operations
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str | Path = CHROMA_DIR,
        embedder: PubMedBertEmbedder | None = None,
        disease_domain: str | None = None
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedder: Embedding model (uses default PubMedBERT if None)
            disease_domain: Optional disease domain for collection naming
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or get_embedder()

        # Add disease domain to collection name if specified
        if disease_domain:
            collection_name = f"{collection_name}_{disease_domain.lower().replace(' ', '_')}"
        self.collection_name = collection_name
        self.disease_domain = disease_domain

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

            total_added += len(batch)

        return total_added

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
        include_embeddings: bool = False
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            where: Metadata filter (e.g., {"section": "Methods"})
            where_document: Document content filter
            include_embeddings: Whether to include embeddings in results

        Returns:
            List of SearchResult objects
        """
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
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert L2 distance to intuitive percentage score (0-100%)
                # L2 distance typically ranges 0-100 in 768-dim space
                # Lower distance = higher relevance
                relevance_score = max(0, min(100, (1 - distance / 100) * 100))

                search_results.append(SearchResult(
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    distance=distance,
                    relevance_score=relevance_score
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
