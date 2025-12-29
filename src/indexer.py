"""
Bio Paper Indexing Pipeline.

Complete pipeline: PDF → Parse → Split → Embed → Store
"""
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from .pdf_parser import BioPaperParser, PaperMetadata, PaperSection
from .text_splitter import BioPaperSplitter, TextChunk
from .embeddings import PubMedBertEmbedder, get_embedder
from .vector_store import BioVectorStore, create_vector_store
from .config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class BioPaperIndexer:
    """
    Complete indexing pipeline for biomedical papers.

    Pipeline:
    1. PDF Parsing (PyMuPDF)
    2. Section-aware Text Splitting
    3. PubMedBERT Embedding
    4. ChromaDB Storage with Metadata
    """

    def __init__(
        self,
        disease_domain: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        embedder: PubMedBertEmbedder | None = None
    ):
        """
        Initialize the indexer.

        Args:
            disease_domain: Disease domain for the collection (e.g., "pheochromocytoma")
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            embedder: Optional custom embedder
        """
        self.disease_domain = disease_domain
        self.parser = BioPaperParser()
        self.splitter = BioPaperSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = embedder or get_embedder()
        self.vector_store = create_vector_store(disease_domain=disease_domain)

        print(f"Initialized indexer for domain: {disease_domain}")
        print(f"Collection: {self.vector_store.collection_name}")
        print(f"Embedding model: {self.embedder.model_name}")

    def index_pdf(
        self,
        pdf_path: str | Path,
        additional_metadata: dict | None = None
    ) -> dict:
        """
        Index a single PDF paper.

        Args:
            pdf_path: Path to the PDF file
            additional_metadata: Extra metadata to add to all chunks

        Returns:
            Indexing statistics
        """
        pdf_path = Path(pdf_path)
        print(f"\nIndexing: {pdf_path.name}")

        # Step 1: Parse PDF
        print("  [1/4] Parsing PDF...")
        try:
            metadata, sections = self.parser.parse_pdf(pdf_path)
        except Exception as e:
            return {"error": f"Failed to parse PDF: {e}", "file": str(pdf_path)}

        # Add disease domain to metadata
        if additional_metadata:
            for key, value in additional_metadata.items():
                setattr(metadata, key, value)

        # Step 2: Split into chunks
        print("  [2/4] Splitting into chunks...")
        chunks = self.splitter.split_paper(metadata, sections)

        # Add disease domain to chunk metadata
        for chunk in chunks:
            chunk.metadata["disease_domain"] = self.disease_domain

        print(f"       → {len(chunks)} chunks created")

        # If no chunks created, return early with error
        if not chunks:
            return {
                "error": "No text content could be extracted from PDF. The file may be image-based or corrupted.",
                "file": str(pdf_path),
                "title": metadata.title,
                "chunks": 0
            }

        # Step 3 & 4: Embed and store (handled by vector_store)
        print("  [3/4] Generating embeddings...")
        print("  [4/4] Storing in VectorDB...")
        added = self.vector_store.add_chunks(chunks, show_progress=False)

        result = {
            "file": str(pdf_path),
            "title": metadata.title,
            "doi": metadata.doi,
            "sections_found": [s.name for s in sections],
            "chunks_created": len(chunks),
            "chunks_added": added,
            "success": True
        }

        print(f"  ✓ Successfully indexed: {metadata.title[:50]}...")
        return result

    def index_directory(
        self,
        directory: str | Path = PAPERS_DIR,
        recursive: bool = True
    ) -> list[dict]:
        """
        Index all PDF files in a directory.

        Args:
            directory: Directory containing PDF files
            recursive: Whether to search subdirectories

        Returns:
            List of indexing results
        """
        directory = Path(directory)

        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return []

        print(f"Found {len(pdf_files)} PDF files")
        print("=" * 60)

        results = []
        for pdf_path in tqdm(pdf_files, desc="Indexing papers"):
            result = self.index_pdf(pdf_path)
            results.append(result)

        # Summary
        successful = sum(1 for r in results if r.get("success", False))
        total_chunks = sum(r.get("chunks_added", 0) for r in results)

        print("\n" + "=" * 60)
        print(f"Indexing complete!")
        print(f"  Papers indexed: {successful}/{len(pdf_files)}")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Collection: {self.vector_store.collection_name}")

        return results

    def get_stats(self) -> dict:
        """Get indexing statistics."""
        return self.vector_store.get_collection_stats()

    def search(
        self,
        query: str,
        top_k: int = 5,
        section: str | None = None
    ):
        """Search the indexed papers."""
        if section:
            return self.vector_store.search_by_section(query, section, top_k)
        return self.vector_store.search(query, top_k)


def create_pheochromocytoma_indexer() -> BioPaperIndexer:
    """Create an indexer for Pheochromocytoma papers."""
    return BioPaperIndexer(disease_domain="pheochromocytoma")


def create_indexer(disease_domain: str) -> BioPaperIndexer:
    """Create an indexer for any disease domain."""
    return BioPaperIndexer(disease_domain=disease_domain)
