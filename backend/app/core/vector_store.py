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

from .config import (
    CHROMA_DIR, COLLECTION_NAME, TOP_K_RESULTS, SIMILARITY_THRESHOLD,
    ENABLE_RERANKER, RERANKER_TOP_K
)
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
        sparse_weight: float = 0.4,
        enable_reranker: bool = ENABLE_RERANKER
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
            enable_reranker: Enable cross-encoder re-ranking for improved relevance
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

        # Re-ranker settings
        self.enable_reranker = enable_reranker
        self._reranker = None  # Lazy loaded

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

    @property
    def reranker(self):
        """Lazy load the cross-encoder reranker."""
        if self._reranker is None and self.enable_reranker:
            from .reranker import get_reranker
            self._reranker = get_reranker()
        return self._reranker

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
        use_hybrid: bool = True,
        use_reranker: bool = True
    ) -> list[SearchResult]:
        """
        Search for similar documents using hybrid search (Dense + BM25) with optional re-ranking.

        Args:
            query: Search query text
            top_k: Number of results to return
            where: Metadata filter (e.g., {"section": "Methods"})
            where_document: Document content filter
            include_embeddings: Whether to include embeddings in results
            use_hybrid: Use hybrid search if enabled (default True)
            use_reranker: Apply cross-encoder re-ranking if enabled (default True)

        Returns:
            List of SearchResult objects
        """
        # Determine how many candidates to fetch for reranking
        fetch_k = top_k
        if use_reranker and self.enable_reranker and self.reranker:
            fetch_k = max(top_k, RERANKER_TOP_K)

        # Use hybrid search if enabled and no filters
        if use_hybrid and self.enable_hybrid and self._bm25_index and where is None:
            results = self._hybrid_search(query, fetch_k)
        else:
            # Fall back to dense-only search
            results = self._dense_search(query, fetch_k, where, where_document, include_embeddings)

        # Apply re-ranking if enabled
        if use_reranker and self.enable_reranker and self.reranker and results:
            results = self.reranker.rerank_search_results(query, results, top_k=top_k)

        return results[:top_k]

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


    def get_paper_embeddings(self, pmid: str) -> Optional[list[float]]:
        """
        Get the average embedding for a paper by PMID.

        Args:
            pmid: Paper PMID

        Returns:
            Average embedding vector or None if not found
        """
        import numpy as np

        # Get all chunks for this paper
        results = self._collection.get(
            where={"pmid": pmid},
            include=["embeddings"]
        )

        if not results["ids"] or len(results["ids"]) == 0:
            return None

        if results["embeddings"] is None or len(results["embeddings"]) == 0:
            return None

        # Average all chunk embeddings
        embeddings = np.array(results["embeddings"])
        avg_embedding = np.mean(embeddings, axis=0)

        return avg_embedding.tolist()

    def find_similar_papers(
        self,
        source_pmid: str,
        top_k: int = 5,
        include_coordinates: bool = True
    ) -> dict:
        """
        Find papers similar to a given paper using embedding similarity.
        Returns similarity scores and optional 2D coordinates for visualization.

        Args:
            source_pmid: PMID of the source paper
            top_k: Number of similar papers to return
            include_coordinates: Whether to include 2D visualization coordinates

        Returns:
            Dict with source paper info, similar papers, and optional coordinates
        """
        import numpy as np

        # Get source paper embedding
        source_embedding = self.get_paper_embeddings(source_pmid)
        if source_embedding is None:
            return {"error": f"Paper {source_pmid} not found in vector store"}

        source_embedding = np.array(source_embedding)

        # Get all unique papers in the collection
        all_results = self._collection.get(
            include=["metadatas", "embeddings"],
            limit=10000
        )

        # Aggregate embeddings by paper
        paper_embeddings = {}
        paper_metadata = {}

        for i, metadata in enumerate(all_results["metadatas"]):
            pmid = metadata.get("pmid", "")
            if not pmid or pmid == source_pmid:
                continue

            if pmid not in paper_embeddings:
                paper_embeddings[pmid] = []
                paper_metadata[pmid] = {
                    "pmid": pmid,
                    "title": metadata.get("paper_title", "Unknown"),
                    "year": metadata.get("year"),
                    "doi": metadata.get("doi"),
                    "keywords": metadata.get("keywords", "")
                }

            if all_results["embeddings"] is not None and len(all_results["embeddings"]) > i:
                paper_embeddings[pmid].append(all_results["embeddings"][i])

        # Calculate average embeddings and similarities
        similarities = []
        all_embeddings = [source_embedding]  # Start with source for coordinates
        all_pmids = [source_pmid]

        for pmid, embeddings in paper_embeddings.items():
            avg_emb = np.mean(embeddings, axis=0)
            all_embeddings.append(avg_emb)
            all_pmids.append(pmid)

            # Cosine similarity
            similarity = np.dot(source_embedding, avg_emb) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(avg_emb)
            )

            similarities.append({
                **paper_metadata[pmid],
                "similarity_score": float(similarity) * 100,
                "embedding": avg_emb.tolist() if include_coordinates else None
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_similar = similarities[:top_k]

        result = {
            "source_pmid": source_pmid,
            "similar_papers": top_similar,
            "total": len(top_similar)
        }

        # Add 2D coordinates for visualization using t-SNE
        if include_coordinates and len(all_embeddings) > 2:
            try:
                from sklearn.manifold import TSNE

                embeddings_array = np.array(all_embeddings)

                # Use t-SNE for 2D projection
                perplexity = min(30, len(all_embeddings) - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                coords_2d = tsne.fit_transform(embeddings_array)

                # Normalize coordinates to [-1, 1] range for visualization
                coords_2d = (coords_2d - coords_2d.mean(axis=0)) / (coords_2d.std(axis=0) + 1e-8)

                # Source paper is at index 0
                result["visualization"] = {
                    "source": {
                        "pmid": source_pmid,
                        "x": float(coords_2d[0][0]),
                        "y": float(coords_2d[0][1])
                    },
                    "papers": []
                }

                # Add coordinates for similar papers
                for i, paper in enumerate(top_similar):
                    paper_idx = all_pmids.index(paper["pmid"])
                    result["visualization"]["papers"].append({
                        "pmid": paper["pmid"],
                        "title": paper["title"],
                        "x": float(coords_2d[paper_idx][0]),
                        "y": float(coords_2d[paper_idx][1]),
                        "similarity_score": paper["similarity_score"]
                    })

            except ImportError:
                # sklearn not available, skip coordinates
                pass
            except Exception as e:
                print(f"Error calculating coordinates: {e}")

        # Clean up embeddings from response
        for paper in result["similar_papers"]:
            paper.pop("embedding", None)

        return result


def create_vector_store(
    disease_domain: str | None = None,
    collection_name: str = COLLECTION_NAME
) -> BioVectorStore:
    """Convenience function to create a vector store."""
    return BioVectorStore(
        collection_name=collection_name,
        disease_domain=disease_domain
    )
