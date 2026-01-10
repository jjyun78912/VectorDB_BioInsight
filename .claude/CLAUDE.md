# CLAUDE.md - BioInsight AI Development Guide

## Project Overview

**BioInsight AI**는 바이오·헬스케어 연구자를 위한 AI 기반 통합 연구 지원 플랫폼입니다.

---

## 핵심 철학

> **"연구자의 발견을 가속화하되, 판단은 연구자에게"**

BioInsight AI는 연구자가 더 빠르게 정보를 찾고, 더 깊이 분석하고, 더 넓은 맥락에서 해석할 수 있도록 돕습니다. 그러나 최종 판단과 결론은 항상 연구자의 몫입니다.

### 4대 핵심 가치

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  [1] 정보 접근성 향상 (Accessibility)                       │
│      • 흩어진 논문/데이터를 한 곳에서                       │
│      • 언어 장벽 해소 (한국어 ↔ 영어)                       │
│      • 복잡한 정보의 구조화된 요약                          │
│                                                             │
│  [2] 분석 자동화 (Automation)                               │
│      • 반복적 분석 작업 대행                                │
│      • 표준 파이프라인 제공                                 │
│      • 연구자는 해석에 집중                                 │
│                                                             │
│  [3] 맥락적 해석 지원 (Contextualization)                   │
│      • 기존 지식과 연결                                     │
│      • 근거 기반 해석 (PMID 인용)                           │
│      • 유사 연구와 비교                                     │
│                                                             │
│  [4] 불확실성 투명성 (Transparency)                         │
│      • 한계와 주의사항 명시                                 │
│      • 과도한 확신 방지                                     │
│      • 검증 방법 제안                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 기능별 철학 적용

| 기능 | 접근성 | 자동화 | 맥락화 | 투명성 |
|------|--------|--------|--------|--------|
| **Paper RAG** | 논문 검색/요약 | 임베딩 자동화 | Q&A로 맥락 제공 | 출처 명시 |
| **Real-time Search** | 다중 소스 통합 | 실시간 수집 | 트렌드 파악 | 소스 표시 |
| **Daily Briefing** | 뉴스 큐레이션 | 자동 다이제스트 | 분야별 정리 | 원문 링크 |
| **RNA-seq** | 분석 진입장벽 낮춤 | 6-Agent 파이프라인 | RAG 해석, TCGA 비교 | Guardrail |
| **ML 예측** | 즉시 예측 | 사전학습 모델 | SHAP 설명 | "참고용" 명시 |

---

## 플랫폼 구성

```
┌─────────────────────────────────────────────────────────────┐
│  BioInsight AI - 바이오/헬스케어 통합 연구 플랫폼           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CORE FEATURES (핵심 기능)                          │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  [1] Paper RAG        - 논문 분석 & Q&A       ✅    │   │
│  │  [2] Real-time Search - PubMed/bioRxiv 검색   ✅    │   │
│  │  [3] Knowledge Graph  - 3D 시각화             ✅    │   │
│  │  [4] Daily Briefing   - AI 연구 뉴스          ✅    │   │
│  │  [5] Trends/Citations - 트렌드/인용 분석      ✅    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ANALYSIS MODULES (분석 모듈)                       │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  [A] RNA-seq Pipeline                               │   │
│  │      • 6-Agent (DEG→Network→Pathway→Valid→Viz→Report) ✅ │
│  │      • ML 예측 (CatBoost + SHAP)              ✅    │   │
│  │      • RAG 해석 (Claude + Vector)             ✅    │   │
│  │  [B] Proteomics       - 단백질 분석           📋    │   │
│  │  [C] Genomics         - 변이 분석             📋    │   │
│  │  [D] Drug Discovery   - 약물 탐색             📋    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

✅ = 완료  📋 = 예정
```

---

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **AI/LLM**: Claude API, Gemini API

### Frontend
- **Framework**: React + Vite + Tailwind CSS
- **Visualization**: react-force-graph-3d (Knowledge Graph)

### Database & Storage
- **Relational**: PostgreSQL
- **Vector DB**: ChromaDB
- **Embeddings**: PubMedBERT, BioBERT

### Analysis-specific
- **RNA-seq**: DESeq2 (R), CatBoost, SHAP, GRNFormer
- **RAG**: LangChain
- **Network Visualization**: 3d-force-graph (Three.js), NetworkX, Plotly

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
│       │   ├── graph.py               # Knowledge graph endpoints
│       │   ├── briefing.py            # Daily Briefing API
│       │   ├── trends.py              # Trending topics API
│       │   ├── dynamic_trends.py      # Dynamic trend analysis
│       │   ├── citations.py           # Citation management
│       │   └── research_gaps.py       # Research gap analysis
│       └── core/
│           ├── config.py              # Configuration
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
│           └── web_crawler_agent.py   # PubMed/CrossRef crawler
├── frontend/
│   └── react_app/
│       ├── App.tsx
│       ├── components/
│       │   ├── Hero.tsx               # Search + PubMedResults
│       │   ├── KnowledgeGraph.tsx     # 3D Galaxy visualization
│       │   ├── TrendingPapers.tsx     # Trending papers
│       │   ├── BioResearchDaily.tsx   # BIO 연구 데일리
│       │   ├── DailyBriefing.tsx      # Daily Briefing
│       │   └── Glow.tsx               # UI effects
│       ├── services/
│       │   └── client.ts              # API client
│       └── package.json
├── bio-daily-briefing/                # Daily Briefing automation
│   ├── src/
│   │   ├── aggregator.py              # Multi-source aggregation
│   │   ├── newsletter_v2.py           # Newsletter generator v2
│   │   ├── trend_analyzer.py          # Trend analysis
│   │   ├── prioritizer.py             # Content prioritization
│   │   ├── ai_summarizer.py           # AI-based summarization
│   │   ├── pubmed_fetcher.py          # PubMed data fetcher
│   │   ├── scheduler.py               # Scheduled tasks
│   │   ├── config/
│   │   │   └── hot_topics.py          # Hot topic configuration
│   │   └── sources/
│   │       ├── biorxiv_fetcher.py
│   │       ├── clinicaltrials_fetcher.py
│   │       └── fda_fetcher.py
│   └── test_v2.py
├── rnaseq_pipeline/                   # RNA-seq 6-Agent Pipeline ✅
│   ├── orchestrator.py                # Pipeline orchestrator
│   ├── agents/
│   │   ├── agent1_deg.py              # DESeq2 DEG analysis
│   │   ├── agent2_network.py          # Network/Hub gene detection
│   │   ├── agent3_pathway.py          # GO/KEGG enrichment
│   │   ├── agent4_validation.py       # DB validation (DisGeNET, OMIM)
│   │   ├── agent5_visualization.py    # Volcano, Heatmap, 2D/3D Network plots
│   │   └── agent6_report.py           # HTML report generation
│   ├── utils/
│   │   └── base_agent.py              # Base agent class
│   └── tests/
│       └── test_pipeline.py
├── chroma_db/                         # Vector DB storage
├── data/
│   └── papers/                        # 수집된 논문
├── scripts/
│   ├── pubmed_collector.py            # PubMed paper collector
│   ├── cli.py                         # CLI interface
│   ├── verify_indexing.py             # Indexing verification
│   └── test_api.py                    # API testing
├── docs/
│   ├── PRD.md
│   └── API.md
├── .claude/
│   └── CLAUDE.md
├── requirements.txt
└── .env

# ═══════════════════════════════════════════════════════════════
#  📋 예정 (미구현)
# ═══════════════════════════════════════════════════════════════
# rnaseq_pipeline/
#   ├── ml/                            # ML 예측 모듈
#   │   ├── predictor.py               # CatBoost
#   │   ├── explainer.py               # SHAP
#   │   └── tcga_comparator.py         # TCGA 비교
├── rag/                               # RAG 해석 모듈 ✅
│   ├── __init__.py
│   └── gene_interpreter.py          # Claude API + Vector Search
#   └── guardrail/                     # 불확실성 명시
#       └── validator.py
# models/                              # 사전 학습 ML 모델
#   └── rnaseq/
#       ├── pancreatic_cancer/
#       ├── lung_cancer/
#       └── breast_cancer/
```

---

## Core Features (핵심 기능)

### 1. Paper RAG ✅

**Purpose**: 논문 PDF 업로드 → 임베딩 → 요약 → Q&A

**Location**: `backend/app/core/`

| Component | File | Description |
|-----------|------|-------------|
| PDF Parser | `pdf_parser.py` | PDF 텍스트 추출 |
| Text Splitter | `text_splitter.py` | Bio-aware 청킹 |
| Embeddings | `embeddings.py` | PubMedBERT 임베딩 |
| Vector Store | `vector_store.py` | ChromaDB 저장/검색 |
| RAG Pipeline | `rag_pipeline.py` | 검색 + LLM 생성 |

**API Endpoints**:
```
POST   /api/paper/upload         # PDF 업로드
POST   /api/paper/analyze        # 논문 분석
POST   /api/chat/ask             # Q&A
```

---

### 2. Real-time Search (Web Crawler) ✅

**Purpose**: PubMed, bioRxiv, Semantic Scholar 실시간 검색

**Location**: `backend/app/core/web_crawler_agent.py`

**Features**:
- PubMed E-utilities API
- CrossRef DOI 검색
- Semantic Scholar 유사 논문 추천
- 트렌딩 논문 수집

**API Endpoints**:
```
GET    /api/crawler/search?q=...           # 실시간 검색
GET    /api/crawler/trending/{category}    # 트렌딩 논문
GET    /api/crawler/similar/{pmid}         # 유사 논문
POST   /api/crawler/fetch/doi              # DOI로 가져오기
```

---

### 3. Knowledge Graph ✅

**Purpose**: 논문/유전자/질병 관계 3D 시각화

**Location**: `frontend/react_app/components/KnowledgeGraph.tsx`

**Features**:
- react-force-graph-3d 기반
- 논문 간 인용 관계
- 유전자-질병 연결
- 실시간 인터랙션

**API Endpoints**:
```
GET    /api/graph/                # 그래프 데이터
GET    /api/graph/node/{id}       # 노드 상세
```

---

### 4. Daily Briefing ✅

**Purpose**: AI 기반 연구 뉴스 다이제스트 (평일 오전 6시 자동 생성)

**Location**: `bio-daily-briefing/`

**Architecture**:
```
┌─────────────────────────────────────────────────────────────┐
│  Daily Briefing - Multi-Source Aggregation Pipeline        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1] FDA Fetcher        → 규제/승인 뉴스 (72시간)          │
│      • Drug approvals, Safety warnings, Recalls            │
│                                                             │
│  [2] ClinicalTrials     → 임상시험 업데이트 (30일)         │
│      • Phase 3 results, New trials, Terminated             │
│                                                             │
│  [3] bioRxiv/medRxiv    → 프리프린트 (3일)                 │
│      • Hot keywords 필터링, Top preprints                  │
│                                                             │
│  [4] PubMed Fetcher     → Peer-reviewed 논문 (2일)         │
│      • 키워드별 검색 (GLP-1, CAR-T, CRISPR 등)             │
│      • High-impact journals 필터링                         │
│                                                             │
│  [5] NewsAggregator     → 통합 및 우선순위 결정            │
│      • Headline 선정 (FDA > Trials > Preprints)            │
│      • 카테고리별 정리                                     │
│                                                             │
│  [6] Newsletter Generator → HTML/JSON 생성                 │
│      • 신문 스타일 레이아웃                                │
│      • PDF 다운로드 지원                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Files**:
| File | Purpose |
|------|---------|
| `src/scheduler.py` | 스케줄러 + 데이터 변환 (⚠️ list 형식 필수) |
| `src/aggregator.py` | 멀티소스 통합 |
| `src/newsletter_generator.py` | HTML 생성 (list 형식 입력) |
| `src/sources/fda_fetcher.py` | FDA 뉴스 수집 |
| `src/sources/clinicaltrials_fetcher.py` | 임상시험 수집 |
| `src/sources/biorxiv_fetcher.py` | 프리프린트 수집 |
| `src/pubmed_fetcher.py` | PubMed 논문 수집 |

**⚠️ 주의사항 (데이터 형식)**:
```python
# newsletter_generator는 list 형식을 기대함
# aggregator는 dict 형식을 반환함
# scheduler.py에서 반드시 변환 필요!

# ❌ Wrong (dict 형식 - newsletter_generator에서 에러)
clinical_trials = {"phase3_results": [...], "new_trials": [...]}

# ✅ Correct (list 형식)
clinical_trials = [
    {"type": "phase3_completed", "title": "...", "description": "..."},
    {"type": "new_trial", "title": "...", "description": "..."}
]
```

**Automation** (launchd):
- 평일 (월~금) 오전 6시 자동 실행
- plist: `~/Library/LaunchAgents/com.bioinsight.daily-briefing.plist`
- 로그: `bio-daily-briefing/output/scheduler.log`

**API Endpoints**:
```
GET    /api/briefing/latest       # 최신 브리핑
GET    /api/briefing/date/{date}  # 특정 날짜 브리핑
GET    /api/briefing/html/latest  # HTML 형식
GET    /api/briefing/archive      # 전체 목록
GET    /api/briefing/trends/summary  # 트렌드 요약
```

---

## Analysis Modules (분석 모듈)

### Module A: RNA-seq Pipeline

**Purpose**: RNA-seq 데이터 분석 + ML 예측 + RAG 해석

**Location**: `rnaseq_pipeline/`

**Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│  RNA-seq Analysis Pipeline                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STAGE 1: COMPUTE (6-Agent)                     ✅ 완료    │
│  ─────────────────────────────────────────────────────────  │
│  [Agent 1] DESeq2 → DEG 분석                    ✅         │
│  [Agent 2] Network → Hub gene 탐지              ✅         │
│  [Agent 3] Pathway → GO/KEGG enrichment         ✅         │
│  [Agent 4] DB 검증 (DisGeNET, OMIM, COSMIC)     ✅         │
│  [Agent 5] 시각화 (Volcano, Heatmap, 2D/3D Network) ✅      │
│  [Agent 6] HTML 리포트                          ✅         │
│                                                             │
│  STAGE 2: PREDICT + INTERPRET                   📋 예정    │
│  ─────────────────────────────────────────────────────────  │
│  [ML] CatBoost + SHAP (샘플 분류)               ✅         │
│  [ML] GRNFormer (유전자 교란 예측)              📋         │
│  [RAG] 논문 기반 해석 (Claude + Vector)         ✅         │
│  [Guardrail] 불확실성 명시                      📋         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**현재 구현 상태**:

| Component | Status | Location |
|-----------|--------|----------|
| 6-Agent Pipeline | ✅ 완료 | `rnaseq_pipeline/agents/` |
| Orchestrator | ✅ 완료 | `rnaseq_pipeline/orchestrator.py` |
| ML 예측 (CatBoost) | ✅ 완료 | `rnaseq_pipeline/ml/` |
| RAG 해석 | ✅ 완료 | `rnaseq_pipeline/rag/gene_interpreter.py` |
| Guardrail | 📋 예정 | 미구현 |
| Pre-trained Models | ✅ 완료 | `models/rnaseq/breast/` |
| 3D Network Viz | ✅ 완료 | `agent5_visualization.py` |

**RAG Gene Selection Logic** (Hub-First):

```python
# agent4_validation.py - RAG 유전자 선택 로직
# 목적: 네트워크 분석에서 도출된 Hub gene을 문헌 기반으로 검증

# 1. Hub gene 우선 선택 (hub_score 내림차순)
hub_genes_df = integrated_df[integrated_df['is_hub'] == True]
    .sort_values('hub_score', ascending=False)

# 2. 남은 자리에 DB-matched 유전자 추가
db_matched_non_hub = integrated_df[
    (integrated_df['db_matched'] == True) &
    (integrated_df['is_hub'] == False)
]

# 결과: Hub gene 100% 포함 (이전: 65% → 현재: 100%)
```

**Network Visualization Types**:

| Type | Library | Features |
|------|---------|----------|
| Galaxy 2D | Matplotlib | 어두운 우주 배경, glow 효과, 전체 라벨 |
| Obsidian 3D | 3d-force-graph + Three.js | 파티클 애니메이션, 회전/줌, 클릭 포커스 |

**ML Components**:

| Component | Purpose | Hardware | Status |
|-----------|---------|----------|--------|
| CatBoost + SHAP | 샘플 분류, 중요 유전자 | CPU | ✅ 완료 |
| GRNFormer | 유전자 교란 예측 | GPU (온디맨드) | 📋 예정 |

**Pre-trained Models** (TCGA 기반):

| Cancer Type | TCGA Code | Status |
|-------------|-----------|--------|
| Breast Cancer | BRCA | ✅ 완료 (AUC 0.998) |
| Pancreatic Cancer | PAAD | 📋 예정 |
| Lung Cancer | LUAD/LUSC | 📋 예정 |
| Multi-cancer | ALL | 📋 예정 |

**API Endpoints** (예정):
```
# 📋 예정 - 현재 미구현
POST   /api/rnaseq/analyze           # 전체 파이프라인
POST   /api/rnaseq/predict           # ML 예측만
POST   /api/rnaseq/perturbation      # GRNFormer
POST   /api/rnaseq/interpret         # RAG 해석만
GET    /api/rnaseq/genes/{symbol}    # 유전자 조회
GET    /api/rnaseq/reports/{id}      # 리포트 조회
```

**Gene Status Card 출력**:

```
═══════════════════════════════════════════════════════════════
  Gene Status Card: KRAS
═══════════════════════════════════════════════════════════════

  EXPRESSION STATUS (DESeq2)
  Direction: ↑ Upregulated (4.2x)
  P-value: 2.3e-09 (adjusted: 5.1e-07)

  ML PREDICTION (CatBoost + SHAP)
  샘플 분류: 췌장암 확률 87%
  KRAS 기여도 (SHAP): +0.45 (1위)
  TCGA 비교: 패턴 일치 ✅

  DISEASE ASSOCIATIONS (DB)
  1. Pancreatic Cancer  Score: 0.95 [COSMIC]
  2. Lung Adenocarcinoma  Score: 0.89 [TCGA]

  RAG INTERPRETATION (논문 기반)
  • KRAS 변이는 췌장암의 90% 이상에서 발견됨 [PMID: 29625050]

  LIMITATIONS (Guardrail)
  ⚠️ ML 예측은 진단이 아니며, 참고용입니다.
  ⚠️ SHAP 순위는 "분류 기여도"입니다.

  SUGGESTED VALIDATIONS
  • KRAS 변이 확인 (Sanger sequencing)
  • 단백질 수준 확인 (Western blot)
```

---

### Module B: Proteomics 📋 (예정)

**Purpose**: 단백질 발현 분석, 상호작용 네트워크

**계획**:
- MS 데이터 분석
- Protein-Protein Interaction (PPI) 네트워크
- Pathway enrichment

---

### Module C: Genomics 📋 (예정)

**Purpose**: 유전체 변이 분석

**계획**:
- VCF 파일 분석
- 변이 주석 (ANNOVAR, VEP)
- 임상적 의미 해석

---

### Module D: Drug Discovery 📋 (예정)

**Purpose**: 약물-타겟 탐색, 리포지셔닝

**계획**:
- Drug-Target 데이터베이스 연동
- 분자 도킹 시뮬레이션
- ADMET 예측

---

## Disease Domains

| Key | Name | Korean | Status |
|-----|------|--------|--------|
| `pancreatic_cancer` | Pancreatic Cancer | 췌장암 | ✅ |
| `blood_cancer` | Blood Cancer | 혈액암 | ✅ |
| `glioblastoma` | Glioblastoma | 교모세포종 | ✅ |
| `alzheimer` | Alzheimer's Disease | 알츠하이머 | ✅ |
| `pcos` | Polycystic Ovary Syndrome | 다낭성난소증후군 | ✅ |
| `pheochromocytoma` | Pheochromocytoma | 갈색세포종 | ✅ |
| `lung_cancer` | Lung Cancer | 폐암 | ✅ |
| `breast_cancer` | Breast Cancer | 유방암 | ✅ |
| `colorectal_cancer` | Colorectal Cancer | 대장암 | ✅ |
| `liver_cancer` | Liver Cancer | 간암 | ✅ |
| `rnaseq_transcriptomics` | RNA-seq & Transcriptomics | RNA-seq 전사체학 | ✅ |

---

## Data Sources

### 논문/지식 (Core Features)

| Source | Purpose | Volume |
|--------|---------|--------|
| PubMed | 논문 메타데이터, 초록 | 3,500만+ |
| bioRxiv | 프리프린트 | 25만+ |
| Semantic Scholar | 인용 관계, 유사 논문 | - |
| CrossRef | DOI 메타데이터 | - |

### 유전자/질병 (Analysis Modules)

| Source | Purpose | Volume |
|--------|---------|--------|
| TCGA | 암 RNA-seq | ~11,000명 |
| GEO | 공개 발현 데이터 | 수만 데이터셋 |
| GTEx | 정상 조직 발현 | ~17,000명 |
| DisGeNET | 유전자-질병 연관 | 100만+ |
| OMIM | 유전 질환 | 16,000+ |
| COSMIC | 암 체세포 변이 | - |

---

## API Endpoints (전체)

```
# ═══════════════════════════════════════════════════════════════
#  CORE FEATURES ✅ 구현됨
# ═══════════════════════════════════════════════════════════════

# Paper Analysis
POST   /api/paper/upload
POST   /api/paper/analyze
GET    /api/search?query=...
GET    /api/search/papers?query=...

# Real-time Search (Crawler)
GET    /api/crawler/search?q=...
GET    /api/crawler/trending/{category}
GET    /api/crawler/similar/{pmid}
POST   /api/crawler/fetch/doi
POST   /api/crawler/fetch/url

# AI Chat
POST   /api/chat/ask
POST   /api/chat/ask-abstract
POST   /api/chat/summarize-abstract
POST   /api/chat/analyze

# Knowledge Graph
GET    /api/graph/
GET    /api/graph/node/{id}

# Daily Briefing
GET    /api/briefing/today
GET    /api/briefing/history

# Trends & Analysis
GET    /api/trends/...
GET    /api/dynamic-trends/...
GET    /api/citations/...
GET    /api/research-gaps/...

# ═══════════════════════════════════════════════════════════════
#  ANALYSIS MODULES 📋 예정 (API 미구현)
# ═══════════════════════════════════════════════════════════════

# RNA-seq Pipeline (예정)
# POST   /api/rnaseq/analyze
# POST   /api/rnaseq/predict
# POST   /api/rnaseq/perturbation
# POST   /api/rnaseq/interpret
# GET    /api/rnaseq/genes/{symbol}
# GET    /api/rnaseq/reports/{id}

# Proteomics (예정)
# POST   /api/proteomics/analyze

# Genomics (예정)
# POST   /api/genomics/analyze

# Drug Discovery (예정)
# POST   /api/drug/search
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
    "confidence": "high",
    "warning": "⚠️ 예측이며 진단이 아닙니다",
    "limitations": ["샘플 수 제한", "단일 시점"],
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

### 전체 플랫폼 공통

| DO ✅ | DON'T ❌ |
|-------|---------|
| 출처/근거 명시 (PMID, URL) | 출처 없는 주장 |
| 한계/주의사항 안내 | 불확실성 숨김 |
| 한국어 우선 출력 | 영문 전용 |
| 후속 조치 제안 | 제안 없이 종료 |
| 원문 링크 제공 | 요약만 제공 |

### Paper RAG / 검색

| DO ✅ | DON'T ❌ |
|-------|---------|
| 관련 논문 PMID 인용 | 출처 없는 정보 |
| 검색 결과 개수 명시 | "많은 논문이 있다" |
| 원문 확인 권장 | AI 요약만 신뢰 유도 |

### 분석 모듈 (RNA-seq 등)

| DO ✅ | DON'T ❌ |
|-------|---------|
| 결과를 "상태"로 제시 | 확정적 결론 |
| ML 예측 = "참고용" 명시 | "진단"으로 표현 |
| 통계적 한계 설명 | 샘플 수 무시 |
| 검증 실험 제안 | 검증 없이 결론 |

### Daily Briefing

| DO ✅ | DON'T ❌ |
|-------|---------|
| 원문 소스 링크 제공 | 요약만 제공 |
| 날짜/시점 명시 | 시점 없는 정보 |
| 분야/카테고리 구분 | 무분별한 나열 |

---

## Development Roadmap

### Core Features

| Phase | Status | Description |
|-------|--------|-------------|
| Paper RAG | ✅ Done | PDF 분석, 임베딩, Q&A |
| Web Crawler | ✅ Done | PubMed, bioRxiv 검색 |
| Knowledge Graph | ✅ Done | 3D 시각화 |
| Daily Briefing | ✅ Done | AI 뉴스 다이제스트 |
| Trends & Citations | ✅ Done | 트렌드 분석, 인용 관리 |

### Analysis Modules

| Phase | Status | Description |
|-------|--------|-------------|
| RNA-seq: 6-Agent Pipeline | ✅ Done | DEG, Network, Pathway, Validation, Viz, Report |
| RNA-seq: ML (CatBoost + SHAP) | ✅ Done | TCGA-BRCA 분류기 (AUC 0.998) |
| RNA-seq: RAG 해석 | ✅ Done | Claude API + PubMedBERT Vector Search |
| RNA-seq: API 통합 | 📋 Planned | FastAPI 엔드포인트 |
| RNA-seq: Guardrail | 📋 Planned | 불확실성 명시 |
| RNA-seq: GRNFormer | 📋 Planned | 유전자 교란 예측 |
| Proteomics | 📋 Planned | 단백질 분석 |
| Genomics | 📋 Planned | 변이 분석 |
| Drug Discovery | 📋 Planned | 약물 탐색 |

---

## Environment Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- R 4.3+ (for DESeq2)
- PostgreSQL 15+

### Installation

```bash
# Clone repository
git clone https://github.com/org/bioinsight-ai.git
cd bioinsight-ai

# Backend setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd frontend/react_app
npm install

# Environment variables
cp .env.example .env

# Start servers
uvicorn backend.app.main:app --reload  # Backend
npm run dev                              # Frontend
```

---

## Testing

```bash
# All tests
pytest

# Specific modules
pytest tests/test_paper_rag.py -v
pytest tests/test_crawler.py -v
pytest tests/test_rnaseq_pipeline.py -v
```

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- [DESeq2 Vignette](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [3d-force-graph](https://github.com/vasturiano/3d-force-graph) - 3D Network Visualization
- [Obsidian 3D Graph](https://github.com/AlexW00/obsidian-3d-graph) - UI/UX Reference

---

## Project History

전체 프로젝트 히스토리는 `docs/PROJECT_HISTORY.md` 참조.

### 최근 업데이트 (2026-01-11)

- **RAG 유전자 선택 로직 수정**: Hub gene 우선 선택 (65% → 100%)
- **Network 시각화 개선**: ENSG ID → 유전자 이름 표시
- **Galaxy 2D Network**: 어두운 우주 배경, glow 효과
- **Obsidian 3D Network**: 파티클 애니메이션, 인터랙티브 컨트롤
- **DESeq2 컬럼 처리**: apeglm shrinkage 5컬럼 동적 대응