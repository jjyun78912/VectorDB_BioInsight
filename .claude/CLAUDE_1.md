# CLAUDE.md - BioInsight AI Development Guide

## Project Overview

BioInsight AI is an AI-powered integrated research platform for bio/healthcare researchers. This document provides guidelines for AI assistants working on this project.

---

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **R Integration**: rpy2 (for DESeq2)
- **ML Libraries**: scikit-learn, XGBoost, SHAP

### Frontend
- **MVP**: Streamlit
- **Production**: React + Tailwind CSS

### Database
- **Relational**: PostgreSQL (analysis results, user data)
- **Vector DB**: ChromaDB (paper embeddings)
- **File Storage**: AWS S3 / GCP Cloud Storage

### AI/ML
- **Embeddings**: PubMedBERT, BioBERT
- **LLM**: GPT-4o, Claude, Gemini (via API)
- **RAG Framework**: LangChain

### Infrastructure
- **Cloud**: AWS / GCP
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

---

## Project Structure

```
bioinsight-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── paper.py     # Paper analysis endpoints
│   │   │   │   ├── rnaseq.py    # RNA-seq analysis endpoints
│   │   │   │   ├── ml.py        # ML prediction endpoints
│   │   │   │   └── assistant.py # AI assistant endpoints
│   │   │   └── deps.py          # Dependencies
│   │   ├── core/
│   │   │   ├── config.py        # Configuration
│   │   │   └── security.py      # Auth & security
│   │   ├── services/
│   │   │   ├── paper_service.py
│   │   │   ├── rnaseq_service.py
│   │   │   ├── ml_service.py
│   │   │   └── assistant_service.py
│   │   ├── models/              # Pydantic models
│   │   └── db/                  # Database models
│   ├── r_scripts/               # DESeq2 R scripts
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── streamlit_app/           # MVP
│   │   └── app.py
│   └── react_app/               # Production
│       ├── src/
│       └── package.json
├── ml/
│   ├── embeddings/              # Embedding models
│   ├── classifiers/             # ML classifiers
│   └── utils/
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-compose.yml
├── docs/
│   ├── PRD.md
│   └── API.md
├── .env.example
├── README.md
└── CLAUDE.md
```

---

## Development Guidelines

### Code Style

#### Python
- Follow PEP 8
- Use type hints for all functions
- Docstrings in Google style format
- Maximum line length: 100 characters

```python
def analyze_rnaseq(
    count_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    design_formula: str = "~ condition"
) -> AnalysisResult:
    """
    Perform RNA-seq differential expression analysis.
    
    Args:
        count_matrix: Gene expression count matrix (genes x samples)
        metadata: Sample metadata with condition information
        design_formula: DESeq2 design formula
        
    Returns:
        AnalysisResult containing DEG list and statistics
        
    Raises:
        ValidationError: If input data format is invalid
    """
    pass
```

#### React/TypeScript
- Use functional components with hooks
- Follow Airbnb style guide
- Use TypeScript for type safety

### Git Workflow

#### Branch Naming
- `feature/` - New features (e.g., `feature/paper-embedding`)
- `bugfix/` - Bug fixes (e.g., `bugfix/deg-calculation`)
- `hotfix/` - Urgent production fixes
- `refactor/` - Code refactoring

#### Commit Messages
Follow Conventional Commits:
```
feat: add volcano plot visualization
fix: correct p-value adjustment in DEG analysis
docs: update API documentation
refactor: optimize embedding generation
test: add unit tests for ML service
```

### API Design

RESTful API conventions:
- Use plural nouns for resources
- Version API endpoints (`/api/v1/`)
- Return appropriate HTTP status codes
- Include pagination for list endpoints

```
GET    /api/v1/papers           # List papers
POST   /api/v1/papers           # Upload paper
GET    /api/v1/papers/{id}      # Get paper details
DELETE /api/v1/papers/{id}      # Delete paper

POST   /api/v1/rnaseq/analyze   # Start analysis
GET    /api/v1/rnaseq/{id}      # Get analysis result
```

---

## Core Modules

### 1. Paper Analysis Module (Vector DB) ✅ IMPLEMENTED

**Purpose**: Process uploaded PDFs, generate embeddings, provide summaries and Q&A

**Key Components**:
- PDF Parser (PyMuPDF/pdfplumber)
- Text Chunker (LangChain)
- Embedding Generator (PubMedBERT)
- Vector Store (ChromaDB)
- LLM Summarizer

**Flow**:
```
PDF Upload → Parse → Chunk → Embed → Store → Summarize
```

#### Implementation Status (2024-12-24)

**Location**: `src/` directory

| Component | File | Status |
|-----------|------|--------|
| PDF Parser | `src/pdf_parser.py` | ✅ Complete |
| Text Splitter | `src/text_splitter.py` | ✅ Complete |
| Embeddings | `src/embeddings.py` | ✅ Complete |
| Vector Store | `src/vector_store.py` | ✅ Complete |
| Indexer | `src/indexer.py` | ✅ Complete |
| Search | `src/search.py` | ✅ Complete |
| CLI | `main.py` | ✅ Complete |

**Key Features Implemented**:

1. **Font-based Section Detection**
   - Detects section headers by font size/bold
   - Handles variations: "Methods", "METHODS", "Materials and Methods"
   - Extracts subsections from Methods (Statistical Analysis, Patients, etc.)

2. **Section-aware Chunking**
   - RecursiveCharacterTextSplitter with bio-specific separators
   - Preserves section context in metadata
   - Chunk size: 1000, Overlap: 200

3. **PubMedBERT Embeddings**
   - Model: `pritamdeka/S-PubMedBert-MS-MARCO`
   - 768-dimensional vectors
   - Optimized for biomedical retrieval

4. **ChromaDB with Rich Metadata**
   ```python
   metadata = {
       "paper_title": str,
       "doi": str,
       "year": str,
       "section": str,        # Abstract, Methods, Results, etc.
       "parent_section": str, # For subsections
       "disease_domain": str, # e.g., "pheochromocytoma"
       "keywords": str,
       "chunk_index": int,
       "source_file": str
   }
   ```

5. **Disease-Domain Collections**
   - Supports multiple disease domains
   - Collection naming: `bio_papers_{domain}`
   - Current: `bio_papers_pheochromocytoma`

**Usage**:

```bash
# Index papers
python main.py index -d pheochromocytoma -p ./data/papers/

# Search
python main.py search -d pheochromocytoma -q "RET mutation" -k 5

# Section-filtered search
python main.py search -d pheochromocytoma -q "RNA seq" -s Methods

# Stats
python main.py stats -d pheochromocytoma
```

**Python API**:

```python
from src.indexer import create_indexer
from src.search import create_searcher

# Indexing
indexer = create_indexer(disease_domain="pheochromocytoma")
indexer.index_pdf("./paper.pdf")

# Searching
searcher = create_searcher(disease_domain="pheochromocytoma")
results = searcher.search("catecholamine synthesis", top_k=5)
results = searcher.search_methods("RNA extraction protocol")
```

**Current Stats** (Pheochromocytoma Collection):
- Papers: 5
- Chunks: 521
- Sections: 8 types (Methods, Background, Conclusion, Abstract, Results, Discussion, Patients, Treatment)

### 2. RNA-seq Analysis Module

**Purpose**: Automated differential expression analysis with visualizations

**Key Components**:
- Data Validator
- DESeq2 Wrapper (rpy2)
- Normalizer (TPM/FPKM/VST)
- Batch Corrector (ComBat)
- Visualizer (matplotlib/plotly)
- Pathway Analyzer (clusterProfiler)

**Flow**:
```
Upload → Validate → Normalize → DESeq2 → Visualize → Pathway → Report
```

### 3. ML Prediction Module

**Purpose**: Build and evaluate classification models from gene signatures

**Key Components**:
- Feature Selector
- Model Trainer (XGBoost, RF, SVM)
- Cross Validator
- SHAP Explainer

**Flow**:
```
Select Features → Train → Validate → Explain → Export
```

### 4. AI Research Assistant

**Purpose**: Integrated Q&A combining paper knowledge, DEG results, and pathway information

**Key Components**:
- Context Builder
- RAG Pipeline
- Response Generator
- Citation Linker

---

## Environment Setup

### Prerequisites
- Python 3.11+
- R 4.3+ (with DESeq2, clusterProfiler)
- Node.js 18+ (for React)
- Docker & Docker Compose
- PostgreSQL 15+

### Local Development

```bash
# Clone repository
git clone https://github.com/org/bioinsight-ai.git
cd bioinsight-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Install R packages
Rscript -e "install.packages(c('DESeq2', 'apeglm', 'clusterProfiler'))"

# Set environment variables
cp .env.example .env
# Edit .env with your configurations

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

### Docker Development

```bash
docker-compose up --build
```

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=app --cov-report=html

# Specific module
pytest tests/test_rnaseq.py

# Integration tests
pytest tests/integration/ -v
```

### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: API endpoint testing
- **E2E Tests**: Full workflow testing

---

## Key Technical Decisions

### Why rpy2 for DESeq2?
DESeq2 is the gold standard for RNA-seq analysis. Rather than reimplementing in Python, we use rpy2 to call R functions directly, ensuring statistical accuracy and reproducibility.

### Why ChromaDB for Embeddings?
ChromaDB offers:
- Simple Python API
- Persistent storage
- Efficient similarity search
- Good integration with LangChain

### Why Multiple LLM Support?
Different LLMs have different strengths. Supporting GPT-4o, Claude, and Gemini allows:
- Fallback options
- Cost optimization
- Performance comparison

---

## Common Tasks

### Adding a New API Endpoint

1. Create route in `backend/app/api/routes/`
2. Add service logic in `backend/app/services/`
3. Define Pydantic models in `backend/app/models/`
4. Register route in `backend/app/api/routes/__init__.py`
5. Write tests in `backend/tests/`

### Adding a New Visualization

1. Add visualization function in `backend/app/services/visualizer.py`
2. Return plot as base64 or save to storage
3. Create frontend component to display

### Updating Embedding Model

1. Update model in `ml/embeddings/`
2. Re-embed existing papers (migration script)
3. Update ChromaDB collection

---

## Troubleshooting

### rpy2 Installation Issues
```bash
# Ensure R is in PATH
export R_HOME=/usr/lib/R

# Install with specific R
pip install rpy2 --install-option="--r-home=/usr/lib/R"
```

### DESeq2 Memory Issues
For large datasets, increase R memory limit:
```r
options(java.parameters = "-Xmx8g")
```

### ChromaDB Performance
For large collections, consider:
- Batch insertions
- Index optimization
- Pagination for queries

---

## Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [DESeq2 Vignette](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)
- [LangChain Docs](https://python.langchain.com/)
- [ChromaDB Docs](https://docs.trychroma.com/)

### Papers
- DESeq2: Love et al., 2014
- PubMedBERT: Gu et al., 2021
- SHAP: Lundberg & Lee, 2017

---

## Contact

For questions about this project:
- Technical Lead: [Name]
- Product Owner: [Name]
- Repository: [URL]
