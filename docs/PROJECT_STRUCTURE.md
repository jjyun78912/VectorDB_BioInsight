# 🧬 BioInsight AI - 프로젝트 구조

## 📊 프로젝트 개요

| 항목 | 수치 |
|------|------|
| **Python 파일** | 227개 |
| **TypeScript 파일** | 115개 |
| **주요 디렉토리** | 15개 |
| **API 엔드포인트** | 12개 라우트 |
| **테스트 파일** | 19개 |

---

## 📁 디렉토리 구조

```
VectorDB_BioInsight/
│
├── 📂 backend/                    # FastAPI 백엔드 서버
│   └── app/
│       ├── api/routes/            # API 엔드포인트
│       ├── core/                  # 핵심 비즈니스 로직
│       ├── db/                    # 데이터베이스 연결
│       ├── models/                # Pydantic 모델
│       ├── services/              # 서비스 레이어
│       └── utils/                 # 유틸리티 함수
│
├── 📂 frontend/                   # 프론트엔드 애플리케이션
│   ├── react_app/                 # React + TypeScript 앱
│   │   ├── src/
│   │   │   ├── components/        # UI 컴포넌트
│   │   │   ├── hooks/             # 커스텀 훅
│   │   │   ├── services/          # API 클라이언트
│   │   │   └── i18n/              # 다국어 지원
│   │   └── dist/                  # 빌드 출력
│   └── streamlit_app/             # Streamlit 대시보드 (레거시)
│
├── 📂 rnaseq_pipeline/            # RNA-seq 분석 파이프라인
│   ├── agents/                    # 6-Agent 파이프라인
│   │   ├── singlecell/            # Single-cell 전용 에이전트
│   │   └── *.py                   # Bulk RNA-seq 에이전트
│   ├── ml/                        # 머신러닝 모듈
│   ├── rag/                       # RAG 해석 모듈
│   ├── reports/                   # 리포트 생성기
│   ├── external_apis/             # 외부 API 클라이언트
│   └── utils/                     # 공통 유틸리티
│
├── 📂 bio-daily-briefing/         # 일일 브리핑 시스템
│   ├── src/
│   │   ├── sources/               # 데이터 소스 (FDA, PubMed 등)
│   │   └── aggregator.py          # 뉴스 집계
│   └── templates/                 # HTML 템플릿
│
├── 📂 models/                     # 학습된 ML 모델
│   └── rnaseq/
│       ├── pancancer/             # Pan-Cancer 17종 분류기
│       └── breast/                # 유방암 특화 모델
│
├── 📂 data/                       # 데이터 저장소
│   ├── chroma_db/                 # ChromaDB 벡터 저장소
│   ├── papers/                    # 수집된 논문
│   ├── tcga/                      # TCGA 데이터
│   └── rnaseq_uploads/            # 사용자 업로드 데이터
│
├── 📂 scripts/                    # 유틸리티 스크립트
├── 📂 tests/                      # 테스트 코드
├── 📂 docs/                       # 문서
│
├── 📄 .env.example                # 환경변수 템플릿
├── 📄 requirements.txt            # Python 의존성
├── 📄 docker-compose.yml          # Docker 설정
└── 📄 Dockerfile                  # 컨테이너 이미지
```

---

## 🔧 Backend 구조

### API 라우트 (`backend/app/api/routes/`)

| 파일 | 경로 | 설명 |
|------|------|------|
| `paper.py` | `/api/paper/*` | 논문 업로드, 분석, 질의응답 |
| `chat.py` | `/api/chat/*` | AI 채팅, RAG 기반 Q&A |
| `search.py` | `/api/search/*` | 벡터 검색, 정밀 검색 |
| `crawler.py` | `/api/crawler/*` | PubMed, bioRxiv 실시간 검색 |
| `rnaseq.py` | `/api/rnaseq/*` | RNA-seq 분석 API (SSE) |
| `graph.py` | `/api/graph/*` | Knowledge Graph 데이터 |
| `briefing.py` | `/api/briefing/*` | 일일 브리핑 |
| `trends.py` | `/api/trends/*` | 연구 트렌드 분석 |
| `citations.py` | `/api/citations/*` | 인용 정보 |

### Core 모듈 (`backend/app/core/`)

| 파일 | 설명 |
|------|------|
| `vector_store.py` | ChromaDB 벡터 저장소 관리 |
| `embeddings.py` | PubMedBERT 임베딩 생성 |
| `text_splitter.py` | 바이오 도메인 인식 텍스트 분할 |
| `pdf_parser.py` | PDF 논문 파싱 |
| `paper_agent.py` | 논문 분석 에이전트 |
| `paper_insights.py` | 논문 인사이트 추출 |
| `llm_helper.py` | LLM API 래퍼 (OpenAI, Claude, Gemini) |
| `search.py` | 하이브리드 검색 엔진 |
| `precision_search.py` | 고정밀 검색 |
| `reranker.py` | 검색 결과 리랭킹 |

---

## 🧬 RNA-seq Pipeline 구조

### 6-Agent Bulk RNA-seq 파이프라인

```
입력 데이터 → Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → Agent 6 → 리포트
              (DEG)    (Network)  (Pathway)  (Valid)   (Viz)     (Report)
```

| Agent | 파일 | 역할 |
|-------|------|------|
| **Agent 1** | `agent1_deg.py` | DESeq2 차등발현 분석 |
| **Agent 2** | `agent2_network.py` | 유전자 네트워크, Hub 유전자 식별 |
| **Agent 3** | `agent3_pathway.py` | GO/KEGG 경로 분석 (Enrichr) |
| **Agent 4** | `agent4_validation.py` | COSMIC, OncoKB DB 검증 |
| **Agent 5** | `agent5_visualization.py` | Volcano, Heatmap, Network 시각화 |
| **Agent 6** | `agent6_report.py` | HTML 리포트 생성 |

### Single-cell 파이프라인 (`agents/singlecell/`)

| Agent | 파일 | 역할 |
|-------|------|------|
| **Agent 1** | `agent1_qc.py` | QC 필터링, 정규화 |
| **Agent 2** | `agent2_cluster.py` | 클러스터링, Cell type annotation |
| **Agent 3** | `agent3_pathway.py` | 경로 분석 |
| **Agent 4** | `agent4_trajectory.py` | Pseudotime 궤적 분석 |
| **Agent 5** | `agent5_cnv_ml.py` | CNV 추론, ML 분류 |
| **Agent 6** | `agent6_report.py` | 리포트 생성 |

### ML 모듈 (`ml/`)

| 파일 | 설명 |
|------|------|
| `pancancer_classifier.py` | Pan-Cancer 17종 분류기 (CatBoost) |
| `trainer.py` | 모델 학습 파이프라인 |
| `predictor.py` | 예측 서비스 |
| `explainer.py` | SHAP 설명 생성 |
| `preprocessor.py` | 데이터 전처리 |
| `tcga_downloader.py` | TCGA 데이터 다운로드 |

### RAG 모듈 (`rag/`)

| 파일 | 설명 |
|------|------|
| `gene_interpreter.py` | 유전자 문헌 기반 해석 |
| `paper_recommender.py` | 관련 논문 추천 |
| `dgidb_client.py` | 약물-유전자 상호작용 조회 |
| `enhanced_interpreter.py` | 향상된 RAG 해석기 |

### External APIs (`external_apis/`)

| 파일 | API | 용도 |
|------|-----|------|
| `oncokb_client.py` | OncoKB | 암 유전자 정보 |
| `string_client.py` | STRING DB | 단백질 상호작용 |
| `reactome_client.py` | Reactome | 경로 데이터 |
| `kegg_client.py` | KEGG | 대사 경로 |
| `uniprot_client.py` | UniProt | 단백질 정보 |
| `civic_client.py` | CIViC | 임상 변이 해석 |

---

## 🖥️ Frontend 구조

### React 컴포넌트 (`frontend/react_app/src/components/`)

```
components/
├── layout/                    # 레이아웃 컴포넌트
│   ├── Navbar.tsx
│   ├── Footer.tsx
│   ├── FeatureSuite.tsx
│   └── CtaSection.tsx
│
├── ui/                        # 재사용 UI 컴포넌트
│   ├── Button.tsx
│   ├── Modal.tsx
│   ├── Card.tsx
│   ├── Input.tsx
│   ├── Tabs.tsx
│   ├── Badge.tsx
│   └── Spinner.tsx
│
└── features/                  # 기능별 컴포넌트
    ├── search/                # 검색 기능
    │   ├── Hero.tsx           # 메인 검색 UI
    │   ├── LocalDBResults.tsx # 로컬 DB 결과
    │   └── PubMedResults.tsx  # PubMed 결과
    │
    ├── rnaseq/                # RNA-seq 분석
    │   ├── RNAseqUploadModal.tsx
    │   ├── PipelineProgress.tsx
    │   └── GeneNetworkGraph.tsx
    │
    ├── literature/            # 논문 관리
    │   ├── LiteratureReview.tsx
    │   ├── ChatWithPDF.tsx
    │   ├── ResearchLibrary.tsx
    │   ├── PaperDetailModal.tsx
    │   ├── PaperInsightsCard.tsx
    │   ├── TrendingPapers.tsx
    │   └── ChatPanel.tsx
    │
    ├── knowledge-graph/       # 지식 그래프
    │   └── KnowledgeGraph.tsx
    │
    └── briefing/              # 일일 브리핑
        ├── DailyBriefing.tsx
        ├── HotTopics.tsx
        ├── ResearchTrends.tsx
        └── TrendAnalysis.tsx
```

### 상태 관리 및 서비스

| 디렉토리 | 설명 |
|----------|------|
| `hooks/` | 커스텀 React 훅 |
| `services/` | API 클라이언트 (`client.ts`) |
| `i18n/` | 다국어 지원 (한국어/영어) |
| `contexts/` | React Context |

---

## 📊 데이터 구조

### ChromaDB Collections (`data/chroma_db/`)

| Collection | 논문 수 | 용도 |
|------------|---------|------|
| `rnaseq_breast_cancer` | ~50편 | 유방암 RNA-seq 논문 |
| `rnaseq_lung_cancer` | ~50편 | 폐암 RNA-seq 논문 |
| `rnaseq_pancreatic_cancer` | ~50편 | 췌장암 RNA-seq 논문 |
| `rnaseq_*` (17종) | ~850편 | 전체 암종 논문 |

### ML 모델 (`models/rnaseq/`)

```
models/rnaseq/
├── pancancer/                 # Pan-Cancer 17종 분류기
│   ├── catboost_model.cbm     # CatBoost 모델
│   ├── preprocessor.joblib    # 전처리기
│   ├── feature_selector.joblib
│   └── label_encoder.joblib
│
└── breast/                    # 유방암 특화 모델
    └── catboost_model.cbm
```

### TCGA 데이터 (`data/tcga/`)

```
data/tcga/
├── BRCA/                      # 유방암 (1,222 samples)
├── LUAD/                      # 폐선암
├── LUSC/                      # 폐편평세포암
├── COAD/                      # 대장암
├── STAD/                      # 위암
├── ... (총 17개 암종)
└── pancancer/                 # 통합 데이터셋
```

---

## 🔧 스크립트 (`scripts/`)

### 데이터 수집
| 스크립트 | 설명 |
|----------|------|
| `pubmed_collector.py` | PubMed 논문 수집 |
| `collect_rnaseq_papers.py` | RNA-seq 특화 논문 수집 |
| `download_tcga_cancer_data.py` | TCGA 데이터 다운로드 |
| `collect_geo_cancer_data.py` | GEO 데이터 수집 |

### ML 학습 및 검증
| 스크립트 | 설명 |
|----------|------|
| `train_rnaseq_classifier.py` | RNA-seq 분류기 학습 |
| `train_pancancer_17types.py` | Pan-Cancer 17종 학습 |
| `evaluate_pancancer_model.py` | 모델 평가 |
| `run_shap_analysis.py` | SHAP 분석 |
| `robust_model_validation.py` | 교차 검증 |

### 유틸리티
| 스크립트 | 설명 |
|----------|------|
| `paper_citation_ranker.py` | 논문 품질 점수 계산 |
| `build_driver_database.py` | Driver 유전자 DB 구축 |
| `install-hooks.sh` | Git 보안 훅 설치 |

---

## 🧪 테스트 (`tests/`)

| 파일 | 설명 |
|------|------|
| `test_api.py` | API 엔드포인트 테스트 |
| `test_rnaseq_pipeline.py` | 파이프라인 통합 테스트 |
| `test_rnaseq_agent.py` | Agent 단위 테스트 |
| `test_singlecell_enhanced.py` | Single-cell 테스트 |
| `conftest.py` | pytest 픽스처 |

---

## 📦 주요 의존성

### Python (`requirements.txt`)
```
# Web Framework
fastapi>=0.100.0
uvicorn>=0.22.0

# AI/ML
openai>=1.0.0
anthropic>=0.18.0
catboost>=1.2.0
shap>=0.42.0

# Bioinformatics
scanpy>=1.9.0
anndata>=0.9.0
gseapy>=1.0.0

# Vector DB
chromadb>=0.4.0
sentence-transformers>=2.2.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
```

### Frontend (`package.json`)
```
# Core
react: ^19.0.0
typescript: ^5.8.0
vite: ^6.0.0

# UI
tailwindcss: ^4.0.0
framer-motion: ^11.0.0
react-force-graph-3d: ^1.24.0

# State
zustand: ^5.0.0
```

---

## 🐳 Docker 구성

### 서비스 구성 (`docker-compose.yml`)

```yaml
services:
  backend:      # FastAPI 서버 (포트 8000)
  frontend:     # React 앱 (포트 3000)
  worker:       # Celery 워커
  redis:        # 태스크 큐
```

### Dockerfile 특징
- **Multi-stage 빌드**: R 4.3 + Python 3.11
- **Non-root 유저**: 보안 강화
- **Health check**: 서비스 상태 모니터링

---

## 📝 설정 파일

| 파일 | 용도 |
|------|------|
| `.env.example` | 환경변수 템플릿 |
| `.gitignore` | Git 제외 파일 |
| `pytest.ini` | 테스트 설정 |
| `tsconfig.json` | TypeScript 설정 |
| `vite.config.ts` | Vite 빌드 설정 |
| `tailwind.config.js` | Tailwind CSS 설정 |

---

## 🔐 보안

### Git Hooks (`.git/hooks/pre-commit`)
- API 키 패턴 자동 감지
- `.env` 파일 커밋 차단
- Private key 감지

### 환경변수 관리
```bash
# 설정 방법
cp .env.example .env
# .env 파일에 실제 API 키 입력
```

---

## 📚 문서

| 문서 | 설명 |
|------|------|
| `CLAUDE.md` | 개발 가이드 (AI 어시스턴트용) |
| `docs/UNIFIED_ARCHITECTURE.md` | 전체 아키텍처 문서 |
| `docs/RNASEQ_ARCHITECTURE.md` | RNA-seq 파이프라인 상세 |
| `docs/SECURITY_KEY_ROTATION.md` | API 키 관리 안내 |
| `docs/PROJECT_STRUCTURE.md` | 프로젝트 구조 (이 문서) |

---

*Last Updated: 2026-01-26*
