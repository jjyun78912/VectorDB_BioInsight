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
- **Production**: React + Vite + Tailwind CSS ✅ IMPLEMENTED
- **3D Visualization**: react-force-graph-3d (Galaxy View)

### Database
- **Relational**: PostgreSQL (analysis results, user data)
- **Vector DB**: ChromaDB (paper embeddings)
- **File Storage**: AWS S3 / GCP Cloud Storage

### AI/ML
- **Embeddings**: PubMedBERT, BioBERT
- **LLM**: Gemini 2.0 Flash (Primary), GPT-4o, Claude (via API)
- **RAG Framework**: LangChain
- **Real-time APIs**: PubMed E-utilities, CrossRef, Semantic Scholar

### Infrastructure
- **Cloud**: AWS / GCP
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

---

## Disease Domains

Supported disease domains for paper collection and analysis:

| Key | Name | Korean | Status |
|-----|------|--------|--------|
| `pancreatic_cancer` | Pancreatic Cancer | 췌장암 | ✅ |
| `blood_cancer` | Blood Cancer | 혈액암 | ✅ |
| `glioblastoma` | Glioblastoma | 교모세포종 | ✅ |
| `alzheimer` | Alzheimer's Disease | 알츠하이머 | ✅ |
| `pcos` | Polycystic Ovary Syndrome | 다낭성난소증후군 | ✅ |
| `pheochromocytoma` | Pheochromocytoma | 갈색세포종 | ✅ |
| `lung_cancer` | Lung Cancer | 폐암 | ✅ NEW |
| `breast_cancer` | Breast Cancer | 유방암 | ✅ NEW |
| `colorectal_cancer` | Colorectal Cancer | 대장암 | ✅ NEW |
| `liver_cancer` | Liver Cancer | 간암 | ✅ NEW |
| `rnaseq_transcriptomics` | RNA-seq & Transcriptomics | RNA-seq 전사체학 | ✅ NEW |

---

## RNA-seq Analysis Stack

### Python Packages

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Data Collection** | GEOparse | 2.0.4 | GEO dataset download |
| | pysradb | 2.5.1 | SRA metadata query |
| **Preprocessing** | scanpy | 1.11.5 | Single-cell analysis |
| | anndata | 0.12.7 | Data structures |
| | scvi-tools | 1.4.1 | Deep learning single-cell |
| **Statistics** | rpy2 | 3.6.4 | R integration |
| **GRN Inference** | arboreto | 0.1.6 | GENIE3 implementation |
| | pyscenic | 0.12.1 | SCENIC pipeline |
| **Network Analysis** | networkx | 3.6.1 | Graph algorithms |
| | python-igraph | 1.0.0 | High-performance graphs |
| **Functional Analysis** | gseapy | 1.1.11 | GSEA in Python |
| | goatools | 1.5.2 | GO analysis |
| **ML/Stats** | scikit-learn | 1.8.0 | Machine learning |
| | scipy | 1.16.3 | Scientific computing |

### R Packages (Bioconductor)

| Package | Purpose |
|---------|---------|
| DESeq2 | Bulk RNA-seq differential expression |
| edgeR | Alternative DEG analysis |
| limma | Microarray/RNA-seq analysis |
| clusterProfiler | GO/KEGG pathway enrichment |

---

## Project Structure

```
VectorDB_BioInsight/
├── backend/
│   └── app/
│       ├── main.py                    # FastAPI entry point
│       ├── api/routes/
│       │   ├── paper.py               # Paper analysis endpoints
│       │   ├── search.py              # Vector search endpoints
│       │   ├── chat.py                # AI chat endpoints (Gemini)
│       │   ├── crawler.py             # Web crawler endpoints
│       │   ├── news.py                # BIO Research Daily endpoints
│       │   └── graph.py               # Knowledge graph endpoints
│       └── core/                      # Core Python modules ✅
│           ├── config.py              # Configuration (API keys)
│           ├── pdf_parser.py          # PDF text extraction
│           ├── text_splitter.py       # Bio-aware text chunking
│           ├── embeddings.py          # PubMedBERT embeddings
│           ├── vector_store.py        # ChromaDB operations
│           ├── indexer.py             # Paper indexing
│           ├── search.py              # Semantic search
│           ├── reranker.py            # Cross-encoder reranking
│           ├── rag_pipeline.py        # RAG pipeline
│           ├── summarizer.py          # AI summarization
│           ├── translator.py          # Korean ↔ English
│           └── web_crawler_agent.py   # PubMed/CrossRef crawler ✅
├── frontend/
│   └── react_app/                     # Production React app ✅
│       ├── App.tsx                    # Main app component
│       ├── components/
│       │   ├── Hero.tsx               # Search + PubMedResults modal
│       │   ├── KnowledgeGraph.tsx     # 3D Galaxy visualization
│       │   ├── TrendingPapers.tsx     # Trending papers section
│       │   ├── BioResearchDaily.tsx   # BIO 연구 데일리
│       │   └── Glow.tsx               # UI effects
│       ├── services/
│       │   └── client.ts              # API client
│       └── package.json
├── scripts/
│   └── pubmed_collector.py            # PubMed paper collection script
├── data/
│   └── papers/                        # Disease domain folders
│       └── {domain}/                  # JSON paper files
├── chroma_db/                         # Vector database storage
├── docs/
│   ├── PRD.md
│   ├── API.md
│   └── EMBEDDING_RAG_ANALYSIS.md      # RAG architecture analysis
├── .claude/
│   ├── CLAUDE.md                      # This file
│   ├── PRD.md
│   └── agents/
│       └── rnaseq-cancer-analyst.md   # RNA-seq analysis agent
├── main.py                            # CLI entry point
├── requirements.txt
└── .env                               # API keys (GEMINI_API_KEY, etc.)
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
- Return appropriate HTTP status codes
- Include pagination for list endpoints

**Current API Endpoints**:

```
# Search & Papers
GET    /api/search?query=...              # Vector search
GET    /api/search/papers?query=...       # Paper-level search
GET    /api/search/similar/{pmid}         # Similar papers (local DB)

# Crawler (Real-time)
GET    /api/crawler/search?q=...          # PubMed live search
GET    /api/crawler/trending/{category}   # Trending papers
GET    /api/crawler/similar/{pmid}        # Similar papers (PubMed elink)
POST   /api/crawler/fetch/doi             # Fetch by DOI
POST   /api/crawler/fetch/url             # Fetch by URL

# Chat (Gemini AI)
POST   /api/chat/ask-abstract             # Q&A on paper abstract
POST   /api/chat/summarize-abstract       # Summarize abstract
POST   /api/chat/analyze                  # Analyze paper

# Graph
GET    /api/graph/                        # Knowledge graph data
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

#### Implementation Status (2025-01-03)

**Location**: `backend/app/core/` directory

| Component | File | Status |
|-----------|------|--------|
| PDF Parser | `backend/app/core/pdf_parser.py` | ✅ Complete |
| Text Splitter | `backend/app/core/text_splitter.py` | ✅ Complete |
| Embeddings | `backend/app/core/embeddings.py` | ✅ Complete |
| Vector Store | `backend/app/core/vector_store.py` | ✅ Complete |
| Indexer | `backend/app/core/indexer.py` | ✅ Complete |
| Search | `backend/app/core/search.py` | ✅ Complete |
| Reranker | `backend/app/core/reranker.py` | ✅ Complete |
| RAG Pipeline | `backend/app/core/rag_pipeline.py` | ✅ Complete |
| CLI | `main.py` | ✅ Complete |

> 📖 **See also**: `docs/EMBEDDING_RAG_ANALYSIS.md` for detailed RAG architecture analysis

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
from backend.app.core.indexer import create_indexer
from backend.app.core.search import create_searcher

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

### 1.5. Web Crawler Agent ✅ IMPLEMENTED (2024-12-29)

**Purpose**: Real-time paper fetching from PubMed, CrossRef, Semantic Scholar

**Location**: `backend/app/core/web_crawler_agent.py`, `backend/app/api/routes/crawler.py`

**Key Features**:

| Feature | Endpoint | Description |
|---------|----------|-------------|
| PubMed Search | `GET /api/crawler/search?q=...` | Real-time hybrid search (latest + high-impact) |
| Trending Papers | `GET /api/crawler/trending/{category}` | 8 categories, major journals filter |
| Similar Papers | `GET /api/crawler/similar/{pmid}` | PubMed elink API for related articles |
| DOI Fetch | `POST /api/crawler/fetch/doi` | CrossRef + Semantic Scholar enrichment |
| URL Import | `POST /api/crawler/fetch/url` | Supports DOI, PubMed, PMC URLs |

**Trending Categories**:
- oncology, immunotherapy, gene_therapy, neurology
- infectious_disease, ai_medicine, genomics, drug_discovery

**Major Journals Filter** (High-Impact):
- Nature, Science, Cell, NEJM, Lancet, JAMA
- Nature Medicine, Nature Genetics, Cell Metabolism
- PNAS, JCO, Blood, Gut, etc.

### 1.6. React Frontend ✅ IMPLEMENTED (2024-12-29)

**Location**: `frontend/react_app/`

**Key Components**:

| Component | File | Description |
|-----------|------|-------------|
| Hero | `components/Hero.tsx` | Main search interface with 3 modes |
| PubMedResults | `components/Hero.tsx` | Split-view modal with paper list + detail |
| KnowledgeGraph | `components/KnowledgeGraph.tsx` | 3D Galaxy visualization |
| TrendingPapers | `components/TrendingPapers.tsx` | Real-time trending papers |

**Search Modes**:
1. **Local DB** - Search indexed papers in ChromaDB
2. **PubMed Live** - Real-time PubMed search with hybrid ranking
3. **DOI/URL** - Import specific paper by identifier

**Galaxy Visualization**:
- 3-layer graph: Source Paper → Similar Papers (15+) → Keywords (20+)
- Color-coded by similarity score (green→yellow→orange→red)
- Interactive 3D navigation with zoom-to-node

**AI Chat Features**:
- Ask questions about paper abstract
- Summarize paper content
- Perplexity-style inline citations

### 1.7. BIO 연구 데일리 ✅ IMPLEMENTED (2025-01-02)

**Purpose**: AI-powered daily research news digest for bio/healthcare researchers

**Location**: `frontend/react_app/src/pages/BioResearchDaily.tsx`, `backend/app/api/routes/news.py`

**Key Features**:

| Feature | Description |
|---------|-------------|
| Multi-source News | PubMed, bioRxiv, medRxiv, Nature News |
| AI Summarization | Gemini-powered Korean summaries |
| Category Filtering | Oncology, Immunotherapy, Genomics, etc. |
| i18n Support | Korean/English interface |
| Real-time Updates | Daily automated collection |

**API Endpoints**:
```
GET  /api/news/daily              # Get daily news digest
GET  /api/news/trending           # Trending research topics
POST /api/news/generate           # Generate AI summary
```

### 1.8. RNA-seq Cancer Analyst Agent ✅ CONFIGURED (2025-01-03)

**Purpose**: Specialized Claude agent for RNA-seq cancer research analysis

**Location**: `.claude/agents/rnaseq-cancer-analyst.md`

**Capabilities**:

| Feature | Tools/Methods |
|---------|---------------|
| Data Collection | GEO, TCGA, SRA via GEOparse, pysradb |
| Preprocessing | Bulk: DESeq2 normalization, Single-cell: scanpy, scvi-tools |
| Differential Expression | DESeq2, edgeR, Wilcoxon rank-sum |
| GRN Inference | GRNformer, GENIE3, SCENIC |
| Network Analysis | Hub gene detection, centrality metrics |
| Functional Analysis | GO, KEGG, GSEA via gseapy, clusterProfiler |
| Validation | DisGeNET, COSMIC, OMIM queries |

**Usage**: Invoke via Claude Code with `subagent_type='rnaseq-cancer-analyst'`

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

1. Update model in `backend/app/core/embeddings.py`
2. Re-embed existing papers (migration script)
3. Update ChromaDB collection
4. See `docs/EMBEDDING_RAG_ANALYSIS.md` for model comparison

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
