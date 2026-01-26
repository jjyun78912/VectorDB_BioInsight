# BioInsight AI - 프로젝트 분석 문서

> **"연구자의 발견을 가속화하되, 판단은 연구자에게"**

---

## 1. 프로젝트 개요

### 1.1 정의

**BioInsight AI**는 바이오·헬스케어 연구자를 위한 AI 기반 통합 연구 지원 플랫폼입니다. 논문 검색/분석, RNA-seq 데이터 파이프라인, 암 유형 예측, 지식 그래프 시각화 등의 기능을 통합하여 연구자의 워크플로우를 자동화하고 가속화합니다.

### 1.2 핵심 가치

| 가치 | 설명 | 구현 기능 |
|------|------|----------|
| **정보 접근성** | 흩어진 논문/데이터를 한 곳에서, 언어 장벽 해소 | Paper RAG, 실시간 PubMed 검색, 자동 번역 |
| **분석 자동화** | 반복적 분석 작업 대행, 표준 파이프라인 제공 | 6-Agent RNA-seq Pipeline, ML 예측 |
| **맥락적 해석** | 기존 지식과 연결, 근거 기반 해석 | RAG 해석 (PMID 인용), TCGA 비교 |
| **불확실성 투명성** | 한계/주의사항 명시, 과도한 확신 방지 | Guardrail, 검증 실험 제안 |

### 1.3 대상 사용자

- 바이오인포매틱스 연구자
- 암 연구 과학자
- 의생명과학 대학원생/포닥
- 제약/바이오테크 R&D 팀

---

## 2. 기술 스택

### 2.1 애플리케이션 스택

| 계층 | 기술 | 버전/상세 |
|------|------|----------|
| **Frontend** | React + Vite + TypeScript | React 18+, Tailwind CSS |
| **Backend** | FastAPI + Uvicorn | Python 3.11+ |
| **Vector DB** | ChromaDB | PubMedBERT 임베딩 |
| **Database** | PostgreSQL | 메타데이터 저장 |
| **시각화** | Plotly, D3.js, 3d-force-graph | 인터랙티브 차트/그래프 |

### 2.2 AI/ML 스택

| 구성 요소 | 기술 | 용도 |
|----------|------|------|
| **LLM** | OpenAI (gpt-4o-mini) | Primary - 빠르고 저렴 |
| | Claude API (Anthropic) | Fallback - 복잡한 분석 |
| | Vertex AI (Google) | Secondary fallback |
| **Embedding** | PubMedBERT | 생물의학 텍스트 특화 |
| **ML** | CatBoost + Optuna | 암 유형 분류 |
| **Explainability** | SHAP | 예측 해석 |

### 2.3 RNA-seq 분석 스택

| 도구 | 용도 |
|------|------|
| DESeq2 (R) | 차등 발현 유전자 분석 |
| Scanpy | Single-cell RNA-seq 분석 |
| NetworkX | 유전자 네트워크 분석 |
| gseapy | GO/KEGG Pathway 분석 |
| Enrichr API | Enrichment 분석 |

---

## 3. 시스템 아키텍처

### 3.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                │
│              React + Vite + TypeScript + Tailwind               │
│                                                                 │
│  Hero Search │ Knowledge Graph 3D │ RNA-seq Upload │ Chat UI   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                               │
│                    FastAPI + REST + SSE                         │
│                                                                 │
│  /api/paper  /api/chat  /api/rnaseq  /api/graph  /api/briefing │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  CORE FEATURES  │ │ ANALYSIS MODULE │ │   DATA LAYER    │
│                 │ │                 │ │                 │
│ • Paper RAG     │ │ • Bulk RNA-seq  │ │ • ChromaDB      │
│ • Live Search   │ │ • Single-cell   │ │ • PostgreSQL    │
│ • Knowledge     │ │ • ML Prediction │ │ • File Storage  │
│   Graph         │ │ • RAG Interpret │ │                 │
│ • Daily Brief   │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXTERNAL SERVICES                           │
│                                                                 │
│  Claude API │ PubMed │ COSMIC │ OncoKB │ TCGA │ Enrichr │ ...  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 디렉토리 구조

```
VectorDB_BioInsight/
├── backend/app/              # FastAPI 백엔드
│   ├── main.py               # 엔트리포인트
│   ├── api/routes/           # API 라우트
│   └── core/                 # 핵심 모듈
│
├── frontend/react_app/       # React 프론트엔드
│   ├── components/           # UI 컴포넌트
│   └── services/             # API 클라이언트
│
├── rnaseq_pipeline/          # RNA-seq 분석 파이프라인
│   ├── orchestrator.py       # 파이프라인 컨트롤러
│   ├── agents/               # 6개 분석 에이전트
│   ├── ml/                   # ML 모델
│   └── rag/                  # RAG 해석 모듈
│
├── bio-daily-briefing/       # 일일 브리핑 시스템
├── models/rnaseq/            # 사전 학습 ML 모델
├── chroma_db/                # 벡터 데이터베이스
├── data/papers/              # 수집된 논문
└── deploy/gcp/               # GCP 배포 설정
```

---

## 4. 핵심 기능 분석

### 4.1 Paper Search & RAG

#### 기능 설명

논문 검색, 분석, Q&A를 위한 RAG(Retrieval-Augmented Generation) 시스템입니다.

#### 구성 요소

| 기능 | 설명 |
|------|------|
| **Local Vector Search** | ChromaDB + PubMedBERT 기반 벡터 검색 |
| **PubMed Live Search** | 실시간 PubMed API 검색 (한→영 자동 번역) |
| **Paper Explainer** | Quick (규칙 기반) + LLM (상세 분석) 2단계 |
| **Paper Insights** | Bottom Line, Study Quality, Evidence Level 평가 |

#### 성능 지표

| 지표 | 수치 |
|------|------|
| 인덱싱된 논문 수 | 850+ 편 |
| 총 청크 수 | 53,000+ 개 |
| 지원 암종 | 17종 |
| 검색 응답 시간 | < 2초 |

#### 코드 위치

- `backend/app/core/vector_store.py` - ChromaDB 관리
- `backend/app/core/embeddings.py` - PubMedBERT 임베딩
- `backend/app/api/routes/paper.py` - Paper API
- `rnaseq_pipeline/rag/gene_interpreter.py` - RAG 해석

---

### 4.2 RNA-seq 6-Agent Pipeline

#### 파이프라인 개요

Bulk RNA-seq 데이터를 자동으로 분석하는 6단계 에이전트 시스템입니다.

```
┌──────────────────────────────────────────────────────────────────┐
│                        STEP 1: DEG ANALYSIS                      │
├──────────────────────────────────────────────────────────────────┤
│  Input: count_matrix.csv + metadata.csv                          │
│                           │                                      │
│                           ▼                                      │
│              ┌─────────────────────┐                             │
│              │     Agent 1: DEG    │                             │
│              │  DESeq2 + apeglm    │                             │
│              └──────────┬──────────┘                             │
│                         │                                        │
│              Output: deg_significant.csv                         │
├──────────────────────────────────────────────────────────────────┤
│                    STEP 2: INTERPRETATION                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Agent 2  │─▶│ Agent 3  │─▶│ Agent 4  │─▶│ Agent 5  │         │
│  │ Network  │  │ Pathway  │  │ Validation│  │ Visual   │         │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│       │             │             │             │                │
│       ▼             ▼             ▼             ▼                │
│   Hub genes    GO/KEGG      COSMIC/OncoKB   Volcano/Heatmap     │
│                                                                  │
│                         │                                        │
│                         ▼                                        │
│              ┌─────────────────────┐                             │
│              │    Agent 6: Report  │                             │
│              │  + ML + RAG Interp  │                             │
│              └──────────┬──────────┘                             │
│                         │                                        │
│              Output: report.html (Interactive)                   │
└──────────────────────────────────────────────────────────────────┘
```

#### 각 에이전트 상세

| Agent | 역할 | 주요 출력 | 핵심 기술 |
|-------|------|----------|----------|
| **Agent 1** | DEG 분석 | deg_significant.csv | DESeq2, apeglm shrinkage |
| **Agent 2** | 네트워크 분석 | hub_genes.csv | Spearman 상관, NetworkX |
| **Agent 3** | Pathway 분석 | pathway_summary.csv | GO/KEGG, Enrichr API |
| **Agent 4** | DB 검증 | validation_results.json | COSMIC, OncoKB |
| **Agent 5** | 시각화 | figures/*.png | Plotly, Matplotlib |
| **Agent 6** | 리포트 생성 | report.html | Jinja2, ML, RAG |

#### 성능 벤치마크

TCGA BRCA 50 samples, 7.5K genes 기준:

| 단계 | 소요 시간 | 출력 |
|------|----------|------|
| Agent 1 (DEG) | ~30초 | 7,583 DEGs |
| Agent 2 (Network) | ~2분 | 3.38M edges, 20 hub genes |
| Agent 3 (Pathway) | ~22초 | 66 terms |
| Agent 4 (Validation) | ~2분 | 20 RAG interpretations |
| Agent 5 (Visualization) | ~2분 | 16 figures |
| Agent 6 (Report) | ~1분 | 11MB HTML |
| **Total** | **~8분** | 완전한 분석 리포트 |

#### 코드 위치

- `rnaseq_pipeline/orchestrator.py` - 파이프라인 오케스트레이터
- `rnaseq_pipeline/agents/agent1_deg.py` ~ `agent6_report.py`

---

### 4.3 ML Cancer Type Prediction

#### 모델 개요

TCGA 데이터로 학습된 Pan-Cancer 분류 모델입니다.

#### 학습 파이프라인

```
TCGA Data (11,000+ samples, 17 cancer types)
              │
              ▼
    Preprocessing (TPM → log2 → StandardScaler)
              │
              ▼
    Optuna Hyperparameter Optimization
              │
              ▼
    CatBoost Training
              │
              ▼
    SHAP Explainer Integration
              │
              ▼
    Pre-trained Model (.cbm + .joblib)
```

#### 성능 지표

| 모델 | 샘플 수 | AUC | 정확도 |
|------|---------|-----|--------|
| Breast Cancer (BRCA) | 1,222 | 0.998 | 99.8% |
| Pan-Cancer (17종) | 11,000+ | 0.988 | 98.8% |

#### 지원 암종 (17종)

| TCGA 코드 | 암종명 | TCGA 코드 | 암종명 |
|-----------|--------|-----------|--------|
| BLCA | 방광암 | LUAD | 폐선암 |
| BRCA | 유방암 | LUSC | 폐편평상피암 |
| COAD | 대장암 | OV | 난소암 |
| GBM | 교모세포종 | PAAD | 췌장암 |
| HNSC | 두경부암 | PRAD | 전립선암 |
| KIRC | 신장암 | SKCM | 피부흑색종 |
| LGG | 저등급신경교종 | STAD | 위암 |
| LIHC | 간암 | THCA | 갑상선암 |
| | | UCEC | 자궁내막암 |

#### 혼동 가능 암종 쌍

| 쌍 유형 | 암종 | 원인 |
|---------|------|------|
| 편평상피암 (SCC) | HNSC ↔ LUSC ↔ SKCM | 같은 조직학적 기원 |
| 선암 | LUAD ↔ PAAD | 선(gland) 조직 유사 |
| 소화기 | COAD ↔ STAD | GI tract 기원 |
| 부인과 | OV ↔ UCEC | Müllerian 기원 |

#### 코드 위치

- `rnaseq_pipeline/ml/pancancer_classifier.py` - 분류기
- `rnaseq_pipeline/ml/predictor.py` - 예측 서비스
- `models/rnaseq/pancancer/` - 사전 학습 모델

---

### 4.4 Knowledge Graph (3D)

#### 기능 설명

논문, 유전자, 질병, Pathway 간의 관계를 3D 그래프로 시각화합니다.

#### 구현 기술

- **라이브러리**: react-force-graph-3d
- **렌더링**: WebGL (Three.js)
- **데이터 소스**: ChromaDB 메타데이터, 인용 관계

#### 노드 유형

| 노드 타입 | 색상 | 설명 |
|----------|------|------|
| Paper | 파란색 | 논문 노드 |
| Gene | 녹색 | 유전자 노드 |
| Disease | 빨간색 | 질병 노드 |
| Pathway | 보라색 | 경로 노드 |

#### 코드 위치

- `frontend/react_app/components/KnowledgeGraph.tsx`
- `backend/app/api/routes/graph.py`

---

### 4.5 Daily Briefing

#### 기능 설명

매일 오전 6시에 자동으로 바이오/헬스케어 뉴스를 수집하여 다이제스트를 생성합니다.

#### 데이터 소스

| 소스 | 수집 내용 |
|------|----------|
| FDA | 신약 승인, 안전성 경고 |
| ClinicalTrials.gov | 임상 시험 결과 |
| bioRxiv/medRxiv | 프리프린트 논문 |
| PubMed | 최신 출판 논문 |

#### 스케줄링

- **방식**: macOS launchd
- **주기**: 평일 오전 6시
- **출력**: HTML 뉴스레터

#### 코드 위치

- `bio-daily-briefing/src/aggregator.py`
- `bio-daily-briefing/src/scheduler.py`

---

## 5. 외부 서비스 연동

### 5.1 API 연동 목록

| 서비스 | 용도 | 인증 방식 |
|--------|------|----------|
| OpenAI API | LLM 추론 (Primary) | API Key |
| Claude API | LLM 추론 (Fallback) | API Key |
| PubMed E-utilities | 논문 검색 | 무료 (Rate limit) |
| COSMIC | 암 유전자 DB 검증 | API Key |
| OncoKB | 암 유전자 임상 정보 | API Token |
| Enrichr | GO/KEGG Enrichment | 무료 |
| Semantic Scholar | 인용 정보 | 무료 (Rate limit) |
| DGIdb | 약물-유전자 상호작용 | GraphQL (무료) |

### 5.2 LLM Fallback Chain

```
1. OpenAI (gpt-4o-mini) ─── 실패 시 ───▶ 2. Claude API
                                              │
                                              ▼ 실패 시
                                        3. Vertex AI
```

---

## 6. 데이터 자산

### 6.1 Vector Database (ChromaDB)

| 컬렉션 | 논문 수 | 청크 수 | 용도 |
|--------|---------|---------|------|
| rnaseq_breast_cancer | 50 | ~3,100 | 유방암 RAG |
| rnaseq_pancreatic_cancer | 50 | ~3,100 | 췌장암 RAG |
| rnaseq_lung_cancer | 50 | ~3,100 | 폐암 RAG |
| ... (17종 암종) | 50 each | ~3,100 each | 각 암종 RAG |
| **Total** | **850+** | **53,000+** | |

### 6.2 ML Models

| 모델 | 파일 | 크기 | 용도 |
|------|------|------|------|
| Pan-Cancer Classifier | `pancancer_model.cbm` | ~50MB | 암 유형 분류 |
| Preprocessor | `preprocessor.joblib` | ~5MB | 데이터 전처리 |
| SHAP Explainer | `shap_explainer.joblib` | ~100MB | 예측 해석 |

---

## 7. API 명세

### 7.1 Core API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/paper/upload` | POST | PDF 논문 업로드 |
| `/api/paper/analyze` | POST | 논문 분석 요청 |
| `/api/chat/ask` | POST | RAG 기반 Q&A |
| `/api/crawler/search` | GET | 실시간 논문 검색 |
| `/api/graph/` | GET | 지식 그래프 데이터 |
| `/api/briefing/latest` | GET | 최신 브리핑 조회 |

### 7.2 RNA-seq API

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/rnaseq/upload` | POST | 분석 파일 업로드 |
| `/api/rnaseq/start/{job_id}` | POST | 파이프라인 시작 |
| `/api/rnaseq/progress/{job_id}` | GET (SSE) | 진행 상황 스트리밍 |
| `/api/rnaseq/status/{job_id}` | GET | 상태 조회 |
| `/api/rnaseq/report/{job_id}` | GET | HTML 리포트 조회 |
| `/api/rnaseq/download/{job_id}` | GET | 결과 다운로드 |

---

## 8. 배포

### 8.1 GCP Cloud Run 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Cloud Run (Frontend)                            │
│                 React + Nginx                                   │
│                 1 vCPU, 512MB RAM                               │
└────────────────────────────┬────────────────────────────────────┘
                             │ API 호출
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Cloud Run (Backend)                             │
│                 FastAPI + R + Python                            │
│                 4 vCPU, 8GB RAM, 60min timeout                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Secret Manager  │ │ Cloud Storage   │ │ External APIs   │
│                 │ │                 │ │                 │
│ • API Keys      │ │ • Models        │ │ • OpenAI        │
│                 │ │ • ChromaDB      │ │ • PubMed        │
│                 │ │ • Results       │ │ • COSMIC        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 8.2 예상 월간 비용

| 서비스 | 사용량 | 비용 |
|--------|--------|------|
| Cloud Run (Backend) | 하루 2시간 × 30일 | $20-30 |
| Cloud Run (Frontend) | 상시 최소 | $5-10 |
| Cloud Storage | 1GB | ~$0.5 |
| Secret Manager | 5개 시크릿 | ~$0.5 |
| **Total** | | **$25-45/월** |

### 8.3 배포 명령어

```bash
cd deploy/gcp
./deploy.sh setup      # 최초 설정
./deploy.sh upload-data  # 데이터 업로드
./deploy.sh all        # 빌드 및 배포
./deploy.sh status     # 상태 확인
```

---

## 9. 코드 품질 분석

### 9.1 강점

| 항목 | 설명 |
|------|------|
| **모듈화** | 에이전트 기반 파이프라인으로 관심사 분리 |
| **확장성** | 새 암종/분석 추가 용이 |
| **투명성** | PMID 인용, 불확실성 명시 |
| **사용자 경험** | SSE 실시간 진행 상황, 인터랙티브 리포트 |

### 9.2 개선 가능 영역

| 항목 | 현재 상태 | 개선 방안 |
|------|----------|----------|
| 테스트 커버리지 | 부분적 | pytest 테스트 확대 |
| 에러 핸들링 | 기본 수준 | 상세 에러 메시지, 재시도 로직 |
| 캐싱 | 없음 | Redis 도입으로 API 응답 캐싱 |
| 로깅 | 기본 수준 | 구조화된 로깅 (structlog) |

---

## 10. 향후 로드맵

### 완료됨 (✅)

- Paper RAG 시스템
- RNA-seq 6-Agent 파이프라인
- ML Cancer Prediction (17종)
- Knowledge Graph 3D
- Daily Briefing
- GCP 배포 구성

### 계획됨 (📋)

| 기능 | 설명 | 우선순위 |
|------|------|---------|
| Single-cell 파이프라인 | Scanpy/Seurat 통합 | 높음 |
| Cell Type Annotation | CellTypist 자동 주석 | 높음 |
| Proteomics 분석 | 단백질 발현 데이터 지원 | 중간 |
| Genomics 변이 분석 | VCF/MAF 통합 분석 | 중간 |
| Drug Discovery | 약물 재목적화 추천 | 낮음 |

---

## 11. 참고 자료

### 기술 문서

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [DESeq2 Vignette](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)
- [Scanpy Documentation](https://scanpy.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)

### 프로젝트 문서

- `docs/UNIFIED_ARCHITECTURE.md` - 전체 아키텍처 상세
- `docs/RNASEQ_ARCHITECTURE.md` - RNA-seq 파이프라인 상세
- `docs/PROJECT_STRUCTURE.md` - 디렉토리 구조
- `deploy/gcp/README.md` - GCP 배포 가이드

---

*문서 작성일: 2026-01-26*
*버전: 1.0*
