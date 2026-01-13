# CLAUDE.md - BioInsight AI Development Guide

## Project Overview

**BioInsight AI**는 바이오·헬스케어 연구자를 위한 AI 기반 통합 연구 지원 플랫폼입니다.

> **"연구자의 발견을 가속화하되, 판단은 연구자에게"**

---

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                          🧬 BioInsight AI Platform Architecture                          │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                 FRONTEND LAYER                                     │  │
│  │                           (React + Vite + Tailwind CSS)                            │  │
│  │                                                                                    │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │   │    Hero      │  │  Knowledge   │  │   Daily      │  │   RNA-seq    │          │  │
│  │   │   Search     │  │    Graph     │  │  Briefing    │  │   Upload     │          │  │
│  │   │  Component   │  │     3D       │  │    View      │  │    Modal     │          │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  │                                                                                    │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │   │  Trending    │  │   Paper      │  │  Pipeline    │  │    Chat      │          │  │
│  │   │   Papers     │  │   Agent      │  │  Progress    │  │  Interface   │          │  │
│  │   │    View      │  │    Chat      │  │    SSE       │  │    (RAG)     │          │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                  API LAYER                                         │  │
│  │                              (FastAPI + REST)                                      │  │
│  │                                                                                    │  │
│  │   /api/paper/*     /api/chat/*      /api/crawler/*    /api/rnaseq/*               │  │
│  │   /api/search/*    /api/graph/*     /api/briefing/*   /api/trends/*               │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                               │
│         ┌────────────────────────────────┼────────────────────────────────┐              │
│         │                                │                                │              │
│         ▼                                ▼                                ▼              │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────┐  │
│  │                     │  │                             │  │                         │  │
│  │   CORE FEATURES     │  │    ANALYSIS MODULES         │  │    DATA LAYER           │  │
│  │                     │  │                             │  │                         │  │
│  │  • Paper RAG        │  │  • Bulk RNA-seq (2-Step)    │  │  • ChromaDB (Vector)    │  │
│  │  • Real-time Search │  │  • Single-cell (1-Step)     │  │  • PostgreSQL           │  │
│  │  • Knowledge Graph  │  │  • ML Prediction            │  │  • File Storage         │  │
│  │  • Daily Briefing   │  │  • RAG Interpretation       │  │                         │  │
│  │  • Trends/Citations │  │                             │  │                         │  │
│  │                     │  │                             │  │                         │  │
│  └─────────────────────┘  └─────────────────────────────┘  └─────────────────────────┘  │
│                                          │                                               │
│                                          ▼                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              EXTERNAL SERVICES                                     │  │
│  │                                                                                    │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │   │   Claude     │  │   PubMed     │  │   COSMIC     │  │    TCGA      │          │  │
│  │   │    API       │  │  E-utilities │  │   OncoKB     │  │   GDC API    │          │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  │                                                                                    │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │   │   bioRxiv    │  │   CrossRef   │  │  Semantic    │  │   Enrichr    │          │  │
│  │   │   medRxiv    │  │     DOI      │  │   Scholar    │  │   (GO/KEGG)  │          │  │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 핵심 철학

### 4대 핵심 가치

| 가치 | 설명 | 적용 |
|------|------|------|
| **정보 접근성** | 흩어진 논문/데이터를 한 곳에서, 언어 장벽 해소 | Paper RAG, 실시간 검색 |
| **분석 자동화** | 반복적 분석 작업 대행, 표준 파이프라인 | 6-Agent Pipeline, ML 예측 |
| **맥락적 해석** | 기존 지식과 연결, 근거 기반 해석 (PMID 인용) | RAG 해석, TCGA 비교 |
| **불확실성 투명성** | 한계/주의사항 명시, 과도한 확신 방지 | Guardrail, 검증 제안 |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18+ / Vite / TypeScript / Tailwind CSS |
| **Backend** | Python 3.11+ / FastAPI / Uvicorn |
| **AI/LLM** | Claude API / PubMedBERT / LangChain |
| **Vector DB** | ChromaDB |
| **Database** | PostgreSQL |
| **RNA-seq** | DESeq2 (R) / Scanpy / CatBoost / SHAP |
| **Visualization** | Plotly / D3.js / 3d-force-graph / NetworkX |

---

## Core Features

### 1. Paper RAG System

```
PDF Upload → Text Splitter → PubMedBERT Embedding → ChromaDB → Semantic Search → Claude API → Answer + Citations
```

**API Endpoints**:
- `POST /api/paper/upload` - PDF 업로드
- `POST /api/paper/analyze` - 논문 분석
- `POST /api/chat/ask` - Q&A

### 2. Real-time Search & Web Crawler

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Local DB   │     │  PubMed     │     │  DOI/URL    │
│   Search    │     │   Live      │     │   Import    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       └──────────────────┬┴─────────────────┘
                         │
                   Web Crawler Agent
                 (PubMed, bioRxiv, CrossRef, Semantic Scholar)
```

### 3. Knowledge Graph (3D)

- react-force-graph-3d 기반 Galaxy View
- 논문/유전자/질병/Pathway 노드
- 인용 관계, 연관성 시각화

### 4. Daily Briefing

- 평일 오전 6시 자동 생성 (launchd)
- Multi-source: FDA, ClinicalTrials, bioRxiv, PubMed
- HTML 뉴스레터 형식

---

## RNA-seq Analysis Architecture

### Data Type Detection & Routing

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                     🧬 RNA-seq Analysis - Unified Entry Point                            │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              INPUT DETECTION                                    │   │
│   │                                                                                 │   │
│   │   📊 Matrix Shape Analysis:                                                     │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │   genes × samples (rows × cols)                                         │  │   │
│   │   │   ────────────────────────────                                          │  │   │
│   │   │   • 20,000 × 6~100    →  Bulk RNA-seq                                   │  │   │
│   │   │   • 20,000 × 1,000+   →  Single-cell RNA-seq                            │  │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   │   🔬 Sample Count Check:                                                        │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│   │   │   samples >= 6  →  DESeq2 통계 분석 가능                                │  │   │
│   │   │   samples < 6   →  Pre-computed DEG (Fold Change only)                  │  │   │
│   │   └─────────────────────────────────────────────────────────────────────────┘  │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                     ┌────────────────────┴────────────────────┐                         │
│                     │                                         │                          │
│                     ▼                                         ▼                          │
│   ┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐    │
│   │                                     │   │                                     │    │
│   │      🧪 BULK RNA-seq                │   │      🔬 SINGLE-CELL RNA-seq         │    │
│   │         (2-Step Process)            │   │         (1-Step Process)            │    │
│   │                                     │   │                                     │    │
│   └─────────────────────────────────────┘   └─────────────────────────────────────┘    │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Bulk RNA-seq Pipeline (2-Step Process)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                       🧪 BULK RNA-seq Pipeline (2-Step)                                  │
│                                                                                          │
│   ══════════════════════════════════════════════════════════════════════════════════    │
│                              STEP 1: STATISTICAL DEG ANALYSIS                            │
│   ══════════════════════════════════════════════════════════════════════════════════    │
│                                                                                          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                               │
│   │ count_      │     │ metadata    │     │ config      │                               │
│   │ matrix.csv  │     │ .csv        │     │ .json       │                               │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                               │
│          └───────────────────┴───────────────────┘                                       │
│                              │                                                           │
│                              ▼                                                           │
│                    ┌─────────────────┐                                                   │
│                    │   AGENT 1: DEG  │                                                   │
│                    │                 │                                                   │
│                    │  DESeq2 → apeglm Shrinkage → Filtering                             │
│                    │                 │                                                   │
│                    └────────┬────────┘                                                   │
│                             │                                                            │
│                             ▼                                                            │
│            ┌────────────────┴────────────────┐                                           │
│            │                                 │                                           │
│   deg_significant.csv              normalized_counts.csv                                 │
│                                                                                          │
│   ══════════════════════════════════════════════════════════════════════════════════    │
│                           STEP 2: INTERPRETATION & REPORT                                │
│   ══════════════════════════════════════════════════════════════════════════════════    │
│                                                                                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│   │  AGENT 2    │  │  AGENT 3    │  │  AGENT 4    │  │  AGENT 5    │                    │
│   │  Network    │  │  Pathway    │  │  DB Valid   │  │  Visual     │                    │
│   │             │  │             │  │             │  │             │                    │
│   │ • Spearman  │  │ • GO/KEGG   │  │ • COSMIC    │  │ • Volcano   │                    │
│   │ • Hub genes │  │ • Enrichr   │  │ • OncoKB    │  │ • Heatmap   │                    │
│   │ • NetworkX  │  │ • gseapy    │  │ • Scoring   │  │ • Network   │                    │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                    │
│          └────────────────┴────────────────┴────────────────┘                            │
│                                    │                                                     │
│                                    ▼                                                     │
│                          ┌─────────────────┐                                             │
│                          │    AGENT 6      │                                             │
│                          │    Report       │                                             │
│                          │                 │                                             │
│                          │  + ML Predict   │◄──── CatBoost + SHAP                       │
│                          │  + RAG Interp   │◄──── Claude API + Vector                   │
│                          │                 │                                             │
│                          └────────┬────────┘                                             │
│                                   │                                                      │
│                                   ▼                                                      │
│                           ┌──────────────┐                                               │
│                           │ report.html  │                                               │
│                           │ (Interactive)│                                               │
│                           └──────────────┘                                               │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Single-cell RNA-seq Pipeline (1-Step Process)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                      🔬 SINGLE-CELL RNA-seq Pipeline (1-Step)                            │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                 │   │
│   │                    UNIFIED SCANPY/SEURAT PIPELINE                               │   │
│   │                                                                                 │   │
│   │   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐                │   │
│   │   │    QC     │──▶│  Normal   │──▶│   HVG     │──▶│   PCA     │                │   │
│   │   │ Filtering │   │  ization  │   │ Selection │   │           │                │   │
│   │   └───────────┘   └───────────┘   └───────────┘   └───────────┘                │   │
│   │         │                                               │                       │   │
│   │         │   ┌───────────────────────────────────────────┘                       │   │
│   │         │   │                                                                   │   │
│   │         ▼   ▼                                                                   │   │
│   │   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐                │   │
│   │   │  Batch    │──▶│   UMAP    │──▶│ Clustering│──▶│ Cell Type │                │   │
│   │   │Correction │   │  t-SNE    │   │ (Leiden)  │   │Annotation │                │   │
│   │   │(Harmony)  │   │           │   │           │   │(CellTypist)│               │   │
│   │   └───────────┘   └───────────┘   └───────────┘   └───────────┘                │   │
│   │                                                         │                       │   │
│   │                                                         ▼                       │   │
│   │   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐                │   │
│   │   │  Pseudo   │◀──│Trajectory │◀──│   DEG     │◀──│  Marker   │                │   │
│   │   │   bulk    │   │(optional) │   │  (Wilcox) │   │   Genes   │                │   │
│   │   └───────────┘   └───────────┘   └───────────┘   └───────────┘                │   │
│   │                                                                                 │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
│   Output: AnnData (.h5ad), Cluster Markers, Cell Composition, UMAP/Violin/Dot plots     │
│                                                                                          │
│   ✅ Complete in 1 Step (QC → Analysis → Report in single pipeline)                      │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Bulk vs Single-cell Comparison

| 구분 | Bulk RNA-seq (2-Step) | Single-cell RNA-seq (1-Step) |
|------|----------------------|------------------------------|
| **입력** | genes × samples matrix | cells × genes matrix |
| **샘플 수** | 6~100 samples | 1,000~100,000+ cells |
| **Step 1** | DESeq2 통계 분석 | - |
| **Step 2** | Network/Pathway/Validation/Report | QC → Clustering → Annotation → Report |
| **주요 도구** | DESeq2, NetworkX, Enrichr, CatBoost | Scanpy, Seurat, CellTypist, Harmony |
| **출력** | DEG list + HTML Report | Cell clusters + Markers + Report |

---

## ML & RAG Modules

### ML Prediction Module

```
TCGA Data → Preprocess (TPM→log2→std) → Optuna HPO → CatBoost → SHAP Explainer
                                                           │
                                                           ▼
User Data → Gene ID Mapping → CatBoost Predict → SHAP Waterfall → Output
                                                                    │
                                                                    ▼
                                              prediction_label, probability, top_genes
                                              ⚠️ "예측은 진단이 아니며, 참고용입니다"
```

**Pre-trained Models**:
| Cancer Type | Samples | AUC | Status |
|-------------|---------|-----|--------|
| Breast (BRCA) | 1,222 | 0.998 | ✅ |
| Pan-Cancer | 11,000+ | 0.95+ | ✅ |
| Pancreatic | TBD | TBD | 📋 |

### RAG Interpretation Module

```
Gene Symbol + Cancer Type + Direction
              │
              ▼
    PubMedBERT Embedding → ChromaDB Search → Top-k chunks
                                                   │
                                                   ▼
                                            Claude API
                                     (Literature-backed interpretation)
                                                   │
                                                   ▼
                              "BRCA1은 DNA 손상 복구에 중요한 역할..." [PMID: 12345678]
```

---

## Project Structure

```
VectorDB_BioInsight/
├── backend/
│   └── app/
│       ├── main.py                      # FastAPI entry point
│       ├── api/routes/
│       │   ├── paper.py                 # Paper analysis
│       │   ├── chat.py                  # AI chat
│       │   ├── crawler.py               # Web crawler
│       │   ├── rnaseq.py                # RNA-seq API (SSE)
│       │   └── briefing.py              # Daily Briefing
│       └── core/
│           ├── pdf_parser.py            # PDF extraction
│           ├── text_splitter.py         # Bio-aware chunking
│           ├── embeddings.py            # PubMedBERT
│           ├── vector_store.py          # ChromaDB
│           └── web_crawler_agent.py     # Multi-source crawler
│
├── frontend/react_app/
│   ├── components/
│   │   ├── Hero.tsx                     # Search interface
│   │   ├── KnowledgeGraph.tsx           # 3D visualization
│   │   ├── RNAseqUploadModal.tsx        # RNA-seq upload
│   │   ├── PipelineProgress.tsx         # SSE progress
│   │   └── DailyBriefing.tsx            # News digest
│   └── services/client.ts               # API client
│
├── rnaseq_pipeline/
│   ├── orchestrator.py                  # Pipeline controller
│   ├── agents/
│   │   ├── agent1_deg.py                # DESeq2 analysis
│   │   ├── agent2_network.py            # Network/Hub genes
│   │   ├── agent3_pathway.py            # GO/KEGG enrichment
│   │   ├── agent4_validation.py         # DB validation
│   │   ├── agent5_visualization.py      # Plots generation
│   │   └── agent6_report.py             # HTML report
│   ├── ml/
│   │   ├── trainer.py                   # CatBoost training
│   │   ├── predictor.py                 # Prediction service
│   │   └── pancancer_classifier.py      # Pan-cancer model
│   └── rag/
│       └── gene_interpreter.py          # RAG interpretation
│
├── bio-daily-briefing/
│   └── src/
│       ├── aggregator.py                # Multi-source aggregation
│       ├── scheduler.py                 # Automation (launchd)
│       └── sources/                     # FDA, Trials, bioRxiv
│
├── models/rnaseq/                       # Pre-trained models
│   ├── breast/                          # TCGA-BRCA (AUC 0.998)
│   └── pancancer/                       # Multi-cancer classifier
│
├── chroma_db/                           # Vector database
├── data/papers/                         # Collected papers
└── docs/
    ├── UNIFIED_ARCHITECTURE.md          # Full architecture
    └── RNASEQ_ARCHITECTURE.md           # RNA-seq details
```

---

## API Endpoints

### Core Features (✅ 구현됨)

```
# Paper Analysis
POST   /api/paper/upload
POST   /api/paper/analyze
POST   /api/chat/ask

# Real-time Search
GET    /api/crawler/search?q=...
GET    /api/crawler/trending/{category}
POST   /api/crawler/fetch/doi

# Knowledge Graph
GET    /api/graph/
GET    /api/graph/node/{id}

# Daily Briefing
GET    /api/briefing/latest
GET    /api/briefing/archive
```

### RNA-seq Analysis (✅ 구현됨)

```
# File Upload & Analysis
POST   /api/rnaseq/upload                # count_matrix.csv, metadata.csv
POST   /api/rnaseq/start/{job_id}        # Start pipeline
GET    /api/rnaseq/progress/{job_id}     # SSE streaming
GET    /api/rnaseq/status/{job_id}       # Status check

# Results
GET    /api/rnaseq/report/{job_id}       # HTML report
GET    /api/rnaseq/download/{job_id}     # Download results
```

---

## Code Patterns & Guidelines

### 출력 언어 (한국어 기본)

```python
# ✅ Correct
interpretation = "KRAS 상향 발현은 췌장암에서 흔히 관찰됩니다."

# ❌ Wrong
interpretation = "KRAS upregulation is commonly observed in pancreatic cancer."
```

### 불확실성 명시 (필수)

```python
# ✅ Correct
result = {
    "prediction": "췌장암 확률 87%",
    "warning": "⚠️ 예측이며 진단이 아닙니다",
    "suggested_validations": ["조직검사 확인"]
}

# ❌ Wrong
result = {"prediction": "이 환자는 췌장암입니다"}
```

### PMID 인용 (필수)

```python
# ✅ Correct
interpretation = "KRAS는 췌장암의 90%에서 변이가 관찰됨 [PMID: 29625050]"

# ❌ Wrong
interpretation = "KRAS는 췌장암의 주요 원인이다"  # 출처 없음
```

---

## DO / DON'T Checklist

| DO ✅ | DON'T ❌ |
|-------|---------|
| 출처/근거 명시 (PMID, URL) | 출처 없는 주장 |
| 한계/주의사항 안내 | 불확실성 숨김 |
| 한국어 우선 출력 | 영문 전용 |
| ML 예측 = "참고용" 명시 | "진단"으로 표현 |
| 검증 실험 제안 | 검증 없이 결론 |

---

## Development Roadmap

### Completed (✅)

| Feature | Description |
|---------|-------------|
| Paper RAG | PDF 분석, 임베딩, Q&A |
| Web Crawler | PubMed, bioRxiv 검색 |
| Knowledge Graph | 3D 시각화 |
| Daily Briefing | AI 뉴스 다이제스트 |
| RNA-seq 6-Agent | DEG, Network, Pathway, Validation, Viz, Report |
| ML Prediction | CatBoost + SHAP (AUC 0.998) |
| RAG Interpretation | Claude + PubMedBERT |
| Web UI | SSE Progress Streaming |

### Planned (📋)

| Feature | Description |
|---------|-------------|
| Single-cell Pipeline | Scanpy/Seurat integration |
| Cell Type Annotation | CellTypist |
| Proteomics | 단백질 분석 |
| Genomics | 변이 분석 |
| Drug Discovery | 약물 탐색 |

---

## Important Notes

### 리포트 생성 시 주의사항

```python
# ⚠️ 리포트 생성 시 CSV 파일 필요
# report_data.json만 있으면 "No data" 표시됨

# ✅ 올바른 폴더 구조
output_dir/
├── deg_significant.csv      # 필수
├── hub_genes.csv            # 필수
├── pathway_summary.csv      # 필수
├── integrated_gene_table.csv # 필수
└── figures/
    ├── volcano_plot.png
    └── network_3d_interactive.html
```

### Daily Briefing 데이터 형식

```python
# newsletter_generator는 list 형식을 기대함
# scheduler.py에서 반드시 변환 필요!

# ❌ Wrong (dict 형식)
clinical_trials = {"phase3_results": [...], "new_trials": [...]}

# ✅ Correct (list 형식)
clinical_trials = [
    {"type": "phase3_completed", "title": "...", "description": "..."}
]
```

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- [DESeq2 Vignette](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)
- [Scanpy Documentation](https://scanpy.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

*Last Updated: 2026-01-13*
