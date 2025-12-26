"""Configuration settings for the BioInsight RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys - Gemini API Key (same as Google API Key)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")

# Embedding Model - PubMedBERT variants
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "pritamdeka/S-PubMedBert-MS-MARCO"  # Fine-tuned for semantic search
)

# Alternative PubMedBERT models:
# - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
# - "NeuML/pubmedbert-base-embeddings"
# - "pritamdeka/S-PubMedBert-MS-MARCO" (recommended for retrieval)

# Gemini Model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

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
