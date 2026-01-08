# BioInsight AI - Product Requirements Document (PRD)

## 1. 제품 개요

### 1.1 제품명
**BioInsight AI** - 바이오·헬스케어 연구자를 위한 AI 기반 통합 연구 지원 플랫폼

### 1.2 제품 비전
바이오 연구자들이 논문 분석, 데이터 해석, 예측 모델링을 하나의 플랫폼에서 수행할 수 있도록 하여, 연구 생산성을 혁신적으로 향상시키는 차세대 연구 지원 도구

### 1.3 핵심 가치 제안

```
┌─────────────────────────────────────────────────────────────┐
│  BioInsight AI = 연구의 모든 단계를 지원                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📚 논문 발견  →  🔬 데이터 분석  →  💡 해석 지원           │
│                                                             │
│  • Paper RAG        • RNA-seq        • RAG 해석            │
│  • 실시간 검색      • Proteomics     • ML 예측             │
│  • 트렌딩 논문      • Genomics       • Guardrail           │
│  • 일일 브리핑      • Drug Discovery                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.4 핵심 철학

> **"연구자의 발견을 가속화하되, 판단은 연구자에게"**

BioInsight AI는 연구자가 더 빠르게 정보를 찾고, 더 깊이 분석하고, 더 넓은 맥락에서 해석할 수 있도록 돕습니다. 그러나 최종 판단과 결론은 항상 연구자의 몫입니다.

**4대 핵심 가치**:

| 가치 | 설명 | 적용 예시 |
|------|------|----------|
| **정보 접근성** | 흩어진 정보를 한 곳에서, 언어 장벽 해소 | 논문 검색, 한국어 요약 |
| **분석 자동화** | 반복 작업 대행, 연구자는 해석에 집중 | RNA-seq 파이프라인 |
| **맥락적 해석** | 기존 지식과 연결, 근거 기반 해석 | RAG, PMID 인용 |
| **불확실성 투명성** | 한계 명시, 과도한 확신 방지 | Guardrail, 경고문 |

---

## 2. 타겟 사용자

### 2.1 사용자 유형

| 유형 | 특성 | 주요 니즈 | 주로 쓰는 기능 |
|------|------|----------|---------------|
| **대학원생/박사후** | 논문 작성 중 | 논문 검색, 해석 지원 | Paper RAG, 검색 |
| **Wet Lab 연구자** | 실험 데이터 보유 | 데이터 해석, 논문 근거 | RNA-seq, RAG 해석 |
| **임상 연구자** | 환자 샘플 보유 | 빠른 예측, 분류 | ML 예측 |
| **바이오인포 연구자** | 분석 전문가 | 고급 분석 도구 | 전체 파이프라인 |
| **제약사 연구원** | 약물 개발 | 타겟 발굴, 스크리닝 | Drug Discovery |

### 2.2 주요 사용 시나리오

**시나리오 1: 논문 작성 중인 대학원생**
```
"이 유전자에 대한 최신 논문 찾아줘"
→ Real-time Search + Paper RAG
→ 관련 논문 요약 + Q&A
```

**시나리오 2: DEG 결과를 해석해야 하는 연구자**
```
"RNA-seq 결과가 나왔는데 어떻게 해석해야 하지?"
→ RNA-seq Pipeline + RAG 해석
→ 논문 기반 해석 + Gene Status Cards
```

**시나리오 3: 새 환자 샘플을 분류해야 하는 임상 연구자**
```
"이 샘플이 어떤 암 유형인지 빠르게 파악하고 싶어"
→ ML 예측 (CatBoost)
→ 즉시 분류 + TCGA 비교
```

---

## 3. 기능 구성

### 3.1 플랫폼 구조

```
┌─────────────────────────────────────────────────────────────┐
│  BioInsight AI                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CORE FEATURES (핵심 기능)                          │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │  [1] Paper RAG           [2] Real-time Search      │   │
│  │      • PDF 업로드            • PubMed 검색         │   │
│  │      • 임베딩/인덱싱         • bioRxiv 검색        │   │
│  │      • 요약/Q&A              • 트렌딩 논문         │   │
│  │                                                     │   │
│  │  [3] Knowledge Graph     [4] Daily Briefing        │   │
│  │      • 3D 시각화             • AI 뉴스 다이제스트  │   │
│  │      • 관계 탐색             • 한국어/영어 지원    │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ANALYSIS MODULES (분석 모듈)                       │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │                                                     │   │
│  │  [A] RNA-seq Pipeline    [B] Proteomics            │   │
│  │      • 6-Agent 분석          • MS 데이터 분석      │   │
│  │      • ML 예측 (CatBoost)    • PPI 네트워크        │   │
│  │      • RAG 해석                                     │   │
│  │      • GRNFormer                                    │   │
│  │                                                     │   │
│  │  [C] Genomics            [D] Drug Discovery        │   │
│  │      • 변이 분석             • 타겟 발굴           │   │
│  │      • 임상적 해석           • 리포지셔닝          │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3.2 Core Features (핵심 기능) - ✅ 완료

#### 3.2.1 Paper RAG

| 항목 | 내용 |
|------|------|
| **기능** | PDF 업로드 → 임베딩 → 요약 → Q&A |
| **기술** | PubMedBERT, ChromaDB, LLM |
| **상태** | ✅ 완료 |

**사용자 플로우**:
```
PDF 업로드 → 텍스트 추출 → 청킹 → 임베딩 → 저장
                                          ↓
        사용자 질문 → 검색 → LLM 답변 생성 ← 관련 청크
```

#### 3.2.2 Real-time Search

| 항목 | 내용 |
|------|------|
| **기능** | PubMed, bioRxiv, Semantic Scholar 실시간 검색 |
| **기술** | PubMed E-utilities, CrossRef API |
| **상태** | ✅ 완료 |

**지원 소스**:
- PubMed (메타데이터, 초록)
- bioRxiv (프리프린트)
- Semantic Scholar (유사 논문, 인용)
- CrossRef (DOI 메타데이터)

#### 3.2.3 Knowledge Graph

| 항목 | 내용 |
|------|------|
| **기능** | 논문/유전자/질병 관계 3D 시각화 |
| **기술** | react-force-graph-3d |
| **상태** | ✅ 완료 |

#### 3.2.4 Daily Briefing

| 항목 | 내용 |
|------|------|
| **기능** | AI 기반 연구 뉴스 다이제스트 |
| **소스** | bioRxiv, ClinicalTrials, FDA |
| **상태** | ✅ 완료 |

---

### 3.3 Analysis Modules (분석 모듈)

#### 3.3.1 Module A: RNA-seq Pipeline

**Status**: 6-Agent 파이프라인 ✅ 완료, ML/RAG 📋 예정

**Architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│  사용자 데이터 업로드                                       │
│  (Count Matrix + Metadata)                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  6-AGENT PIPELINE                           ✅ 완료        │
├─────────────────────────────────────────────────────────────┤
│  [Agent 1] DESeq2 → DEG 분석                ✅             │
│  [Agent 2] Network → Hub gene 탐지          ✅             │
│  [Agent 3] Pathway → GO/KEGG enrichment     ✅             │
│  [Agent 4] DB 검증 (DisGeNET, OMIM, COSMIC) ✅             │
│  [Agent 5] 시각화 (Volcano, Heatmap, etc.)  ✅             │
│  [Agent 6] HTML 리포트 생성                 ✅             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  확장 모듈 (예정)                           📋             │
├─────────────────────────────────────────────────────────────┤
│  [ML] CatBoost + SHAP (샘플 분류)           📋             │
│  [ML] GRNFormer (유전자 교란 예측)          📋             │
│  [RAG] 논문 기반 해석                       📋             │
│  [Guardrail] 불확실성 명시                  📋             │
└─────────────────────────────────────────────────────────────┘
```

**현재 구현 상태**:

| Component | Status | Location |
|-----------|--------|----------|
| 6-Agent Pipeline | ✅ 완료 | `rnaseq_pipeline/agents/` |
| Orchestrator | ✅ 완료 | `rnaseq_pipeline/orchestrator.py` |
| API 통합 | 📋 예정 | 미구현 |
| ML 예측 | 📋 예정 | 미구현 |
| RAG 해석 | 📋 예정 | 미구현 |
| Guardrail | 📋 예정 | 미구현 |

**예정 ML Components**:

| Component | Purpose | Hardware | Status |
|-----------|---------|----------|--------|
| CatBoost + SHAP | 샘플 분류, 중요 유전자 | CPU | 📋 예정 |
| GRNFormer | 유전자 교란 예측 | GPU (온디맨드) | 📋 예정 |

**예정 Pre-trained Models** (TCGA 기반):

| Cancer Type | Samples | Status |
|-------------|---------|--------|
| Pancreatic (PAAD) | ~350 | 📋 예정 |
| Lung (LUAD/LUSC) | ~1,000 | 📋 예정 |
| Breast (BRCA) | ~1,200 | 📋 예정 |
| Multi-cancer | ~11,000 | 📋 예정 |

#### 3.3.2 Module B: Proteomics 📋

**Status**: 예정

**계획**:
- MS 데이터 분석
- Protein-Protein Interaction (PPI) 네트워크
- Pathway enrichment

#### 3.3.3 Module C: Genomics 📋

**Status**: 예정

**계획**:
- VCF 파일 분석
- 변이 주석 (ANNOVAR, VEP)
- 임상적 의미 해석 (ClinVar)

#### 3.3.4 Module D: Drug Discovery 📋

**Status**: 예정

**계획**:
- Drug-Target 데이터베이스 연동
- 분자 도킹 시뮬레이션
- ADMET 예측

---

## 4. 기대 효과 vs 기대하지 않는 것

### 4.1 기대하는 것 ⭕

| 효과 | 설명 |
|------|------|
| 논문 검색 시간 단축 | 실시간 검색 + AI 요약 |
| 데이터 해석 지원 | RAG 기반 논문 근거 제공 |
| 새 샘플 즉시 분류 | 사전 학습 ML 모델 |
| 연구 동향 파악 | Daily Briefing |
| 맹신 방지 | Guardrail로 한계 명시 |

### 4.2 기대하지 않는 것 ❌

| 항목 | 이유 |
|------|------|
| 진단 대체 | ML 예측은 참고용 |
| 인과관계 확정 | 통계적 연관성만 |
| 연구자 대체 | 최종 판단은 사용자 |
| 모든 분석 자동화 | 전문가 검토 필요 |

---

## 5. 기술 아키텍처

### 5.1 기술 스택

| 레이어 | 기술 |
|--------|------|
| **Backend** | Python 3.11+, FastAPI |
| **Frontend** | React + Vite + Tailwind |
| **Database** | PostgreSQL, ChromaDB |
| **AI/ML** | Claude, Gemini, CatBoost, SHAP, GRNFormer |
| **Embeddings** | PubMedBERT, BioBERT |
| **RAG** | LangChain |
| **Visualization** | react-force-graph-3d |

### 5.2 프로젝트 구조

```
VectorDB_BioInsight/
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/routes/
│       │   ├── paper.py               # ✅
│       │   ├── search.py              # ✅
│       │   ├── chat.py                # ✅
│       │   ├── crawler.py             # ✅
│       │   ├── graph.py               # ✅
│       │   ├── briefing.py            # ✅
│       │   ├── trends.py              # ✅
│       │   ├── dynamic_trends.py      # ✅
│       │   ├── citations.py           # ✅
│       │   └── research_gaps.py       # ✅
│       └── core/
│           ├── pdf_parser.py
│           ├── embeddings.py
│           ├── vector_store.py
│           ├── rag_pipeline.py
│           └── web_crawler_agent.py
├── frontend/
│   └── react_app/
│       ├── components/
│       │   ├── Hero.tsx
│       │   ├── KnowledgeGraph.tsx
│       │   ├── TrendingPapers.tsx
│       │   ├── BioResearchDaily.tsx
│       │   └── DailyBriefing.tsx
│       └── services/
├── bio-daily-briefing/
│   └── src/
│       ├── aggregator.py
│       ├── newsletter_v2.py
│       ├── trend_analyzer.py
│       ├── prioritizer.py
│       ├── ai_summarizer.py
│       └── sources/
├── rnaseq_pipeline/                   # ✅ 6-Agent 완료
│   ├── orchestrator.py
│   ├── agents/
│   │   ├── agent1_deg.py
│   │   ├── agent2_network.py
│   │   ├── agent3_pathway.py
│   │   ├── agent4_validation.py
│   │   ├── agent5_visualization.py
│   │   └── agent6_report.py
│   └── utils/
#   ├── ml/                            # 📋 예정
#   ├── rag/                           # 📋 예정
#   └── guardrail/                     # 📋 예정
├── chroma_db/
├── data/
└── scripts/
```

---

## 6. 데이터 소스

### 6.1 논문/지식

| Source | Purpose | Volume |
|--------|---------|--------|
| PubMed | 논문 메타데이터 | 3,500만+ |
| bioRxiv | 프리프린트 | 25만+ |
| Semantic Scholar | 인용 관계 | - |

### 6.2 유전자/질병

| Source | Purpose | Volume |
|--------|---------|--------|
| TCGA | 암 RNA-seq | ~11,000명 |
| GEO | 공개 발현 데이터 | 수만 |
| GTEx | 정상 조직 발현 | ~17,000명 |
| DisGeNET | 유전자-질병 연관 | 100만+ |
| OMIM | 유전 질환 | 16,000+ |

---

## 7. 개발 로드맵

### Phase 1: Core Features ✅ 완료

| 기능 | 상태 |
|------|------|
| Paper RAG | ✅ |
| Real-time Search | ✅ |
| Knowledge Graph | ✅ |
| Daily Briefing | ✅ |

### Phase 2: RNA-seq Module

| 기능 | 상태 | 비고 |
|------|------|------|
| 6-Agent Pipeline | ✅ 완료 | `rnaseq_pipeline/` |
| Orchestrator | ✅ 완료 | 파이프라인 조율 |
| API 통합 | 📋 예정 | FastAPI 엔드포인트 |
| TCGA 데이터 수집 | 📋 예정 | Pre-trained 모델용 |
| CatBoost 모델 학습 | 📋 예정 | 샘플 분류 |
| SHAP 해석 | 📋 예정 | 피처 중요도 |
| RAG 해석 | 📋 예정 | 논문 기반 |
| Guardrail | 📋 예정 | 불확실성 명시 |
| GRNFormer | 📋 예정 | 유전자 교란 |
| Gene Status Cards | 📋 예정 | 리포트 확장 |

### Phase 3: Additional Modules 📋 예정

| 모듈 | 예상 기간 |
|------|----------|
| Proteomics | 8주 |
| Genomics | 8주 |
| Drug Discovery | 12주 |

---

## 8. 성공 지표 (KPIs)

### Core Features

| 지표 | 목표 |
|------|------|
| 논문 검색 응답 시간 | < 3초 |
| Paper RAG 정확도 | > 80% |
| DAU | 100명 |
| MAU | 500명 |

### Analysis Modules

| 지표 | 목표 |
|------|------|
| RNA-seq 파이프라인 | < 10분 |
| ML 예측 AUC | > 0.90 |
| 불확실성 명시율 | 100% |

---

## 9. 리스크 및 대응

| 리스크 | 영향도 | 대응 |
|--------|--------|------|
| LLM Hallucination | 높음 | Guardrail, PMID 검증 |
| ML 과적합 | 높음 | CatBoost, Cross-validation |
| 사용자 맹신 | 높음 | 경고문, 한계 명시 |
| 데이터 편향 | 중간 | 다양한 데이터 소스 |

---

## 10. 부록

### 10.1 용어 정의

| 용어 | 정의 |
|------|------|
| **RAG** | Retrieval-Augmented Generation |
| **DEG** | Differentially Expressed Gene |
| **SHAP** | SHapley Additive exPlanations |
| **GRNFormer** | Gene Regulatory Network Transformer |
| **Guardrail** | 과도한 해석 방지 안전장치 |

### 10.2 질병 도메인

| Key | Name | Korean |
|-----|------|--------|
| `pancreatic_cancer` | Pancreatic Cancer | 췌장암 |
| `lung_cancer` | Lung Cancer | 폐암 |
| `breast_cancer` | Breast Cancer | 유방암 |
| `blood_cancer` | Blood Cancer | 혈액암 |
| `glioblastoma` | Glioblastoma | 교모세포종 |
| `alzheimer` | Alzheimer's Disease | 알츠하이머 |

### 10.3 변경 이력

| 버전 | 날짜 | 변경 사항 |
|------|------|-----------|
| 1.0 | 2024.12 | 초안 작성 |
| 2.0 | 2025.01 | RNA-seq ML/RAG 추가 |
| 3.0 | 2025.01 | 범용 플랫폼 구조 재정의 |
| 3.1 | 2025.01 | 실제 구조와 동기화 (6-Agent ✅, ML/RAG 📋) |