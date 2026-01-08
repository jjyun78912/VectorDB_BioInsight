"""Configuration settings for the BioInsight RAG system."""
import os
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(name: str = "bioinsight") -> logging.Logger:
    """
    Configure and return a logger for the application.

    Args:
        name: Logger name (usually module name)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))

    logger.addHandler(console_handler)

    # Optionally add file handler
    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


# Create default logger
logger = setup_logging("bioinsight")

# Paths - Navigate from backend/app/core/ to project root
BASE_DIR = Path(__file__).parent.parent.parent.parent  # backend/app/core -> backend/app -> backend -> root
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys - Gemini API Key (same as Google API Key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")

# API Keys - Anthropic Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Embedding Model - PubMedBERT variants
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "pritamdeka/S-PubMedBert-MS-MARCO"  # Fine-tuned for semantic search
)

# Alternative PubMedBERT models:
# - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# - "NeuML/pubmedbert-base-embeddings"
# - "pritamdeka/S-PubMedBert-MS-MARCO" (recommended for retrieval)

# Gemini Model - 2.5 Pro for better reasoning
GEMINI_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-pro")

# Claude Model - Default to Claude 3.5 Sonnet
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

# LLM Provider Selection (claude or gemini)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")

# Text Splitting Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Bio Paper Sections (ordered)
BIO_PAPER_SECTIONS = [
    "Abstract",
    "Introduction",
    "Background",
    "Materials and Methods",
    "Methods",
    "Experimental Procedures",
    "Results",
    "Discussion",
    "Conclusion",
    "Conclusions",
    "Acknowledgments",
    "References",
    "Supplementary",
]

# Sub-sections commonly found in Methods
METHODS_SUBSECTIONS = [
    "RNA extraction",
    "DNA extraction",
    "Library preparation",
    "RNA-seq processing",
    "Sequencing",
    "Differential expression analysis",
    "Statistical analysis",
    "Cell culture",
    "Western blot",
    "PCR",
    "qRT-PCR",
    "Immunohistochemistry",
    "Flow cytometry",
    "Animal experiments",
    "Clinical samples",
    "Data availability",
]

# Sections to exclude from embedding
EXCLUDE_SECTIONS = [
    "References",
    "Acknowledgments",
    "Author Contributions",
    "Competing Interests",
    "Funding",
]

# ChromaDB Collection Name
COLLECTION_NAME = "bio_papers"

# Retrieval Settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# Hybrid Search Settings
ENABLE_HYBRID_SEARCH = True  # Enable Dense + BM25 hybrid search
DENSE_WEIGHT = 0.6  # Weight for dense (semantic) search (0-1)
SPARSE_WEIGHT = 0.4  # Weight for sparse (BM25 keyword) search (0-1)

# Re-ranker Settings (Cross-encoder for improved relevance)
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, good quality
)
# Alternative reranker models:
# - "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast, 22M params)
# - "cross-encoder/ms-marco-MiniLM-L-12-v2" (balanced, 33M params)
# - "BAAI/bge-reranker-base" (scientific text, 278M params)
# - "BAAI/bge-reranker-large" (best quality, 560M params)

RERANKER_ORIGINAL_WEIGHT = float(os.getenv("RERANKER_ORIGINAL_WEIGHT", "0.3"))
RERANKER_WEIGHT = float(os.getenv("RERANKER_RERANK_WEIGHT", "0.7"))
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "20"))  # Candidates to rerank
