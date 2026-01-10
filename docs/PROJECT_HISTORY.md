# BioInsight AI - Project History

> 프로젝트 시작일: 2025-12-26
> 마지막 업데이트: 2026-01-11

---

## Phase 1: Foundation (2025-12-26 ~ 2025-12-29)

### 2025-12-26 (Day 1) - 프로젝트 초기화
- [x] **feat**: VectorDB BioInsight - Biomedical Paper RAG System 초기 구축
  - FastAPI 백엔드 설정
  - ChromaDB 벡터 데이터베이스 설정
  - PubMedBERT 임베딩 모델 통합
  - 기본 PDF 파싱 구현

### 2025-12-29 (Day 4) - React 프론트엔드 & 실시간 검색
- [x] **feat**: React 프론트엔드 + Chat with PDF Agent 추가
  - React + Vite + Tailwind CSS 설정
  - SciSpace 스타일 UI 구현
  - Paper Agent로 지능형 PDF Q&A 구현
- [x] **feat**: Web Crawler Agent + 실시간 PubMed 통합
  - PubMed, CrossRef, Semantic Scholar 실시간 검색
  - 트렌딩 논문 API (8개 카테고리)
  - DOI/URL 기반 논문 가져오기
- [x] **feat**: PubMed 검색 관련성 개선 & UI 수정
- [x] **fix**: PubMed 결과 모달 스크롤 문제 해결 (10개 논문 표시)

---

## Phase 2: Knowledge Discovery (2025-12-30 ~ 2025-12-31)

### 2025-12-30 (Day 5) - 지식 시각화 & 아키텍처 정리
- [x] **feat**: PubMed 유사 논문 API & Galaxy 키워드 시각화
- [x] **feat**: Paper Discovery Flow + Major Journals 필터
- [x] **feat**: Chat history 추가 (PubMed paper AI Q&A)
- [x] **feat**: Research analysis modules & Paper Summarizer Agent
- [x] **feat**: Research Trend Analysis APIs (다차원 스코어링)
- [x] **refactor**: 아키텍처 재구조화
  - `src/` → `backend/app/core/` 통합
  - Clean architecture 적용
- [x] **docs**: CLAUDE_1.md 업데이트

### 2025-12-31 (Day 6) - 트렌드 분석 & PDF 개선
- [x] **feat**: 자동화된 트렌드 검증 파이프라인 (다중 신호 스코어링)
- [x] **feat**: 전문 자동 검색 & AI 요약
- [x] **feat**: 서브섹션 감지 (소스 표시 개선)
- [x] **feat**: 통합 ResearchTrends 컴포넌트 (3-tab 인터페이스)
- [x] **feat**: Chat with PDF 응답 간결화
- [x] **feat**: Literature Review 재설계 (논문 관리 도구)
- [x] **feat**: Research Trends 버튼 통합 & 논문 링크 처리 개선
- [x] **fix**: 섹션 표시 이름 정리 (인용 혼동 방지)
- [x] **fix**: Chat with PDF 오류 처리 개선
- [x] **fix**: PDF 파싱 개선

---

## Phase 3: Daily Briefing & News (2026-01-02 ~ 2026-01-05)

### 2026-01-02 (Day 8) - BIO 연구 데일리
- [x] **feat**: BIO 연구 데일리 + AI 뉴스 생성
  - Gemini API 기반 뉴스 요약
  - 다국어 지원
- [x] **feat**: 글로벌 멀티소스 뉴스 지원
- [x] **refactor**: BIO 연구 데일리 UI 통합 (메인 플랫폼 디자인)

### 2026-01-03 (Day 9) - 질병 도메인 확장
- [x] **feat**: 새로운 암 도메인 추가 & RNA-seq 분석 에이전트
  - 췌장암, 혈액암, 교모세포종, 알츠하이머, 다낭성난소증후군
- [x] **feat**: 갈색세포종(Pheochromocytoma) 도메인 추가
- [x] **feat**: BIO Research Daily 전체 i18n 지원

### 2026-01-04 (Day 10) - RNA-seq 파이프라인 시작
- [x] **feat**: RNA-seq 전체 파이프라인 + DESeq2 통합 (rpy2)
  - R DESeq2 연동
  - 차등발현유전자 분석
- [x] **docs**: CLAUDE.md 업데이트

### 2026-01-05 (Day 11) - Daily Briefing 시스템
- [x] **feat**: BIO Daily Briefing 뉴스레터 자동화 시스템
  - FDA Fetcher (규제/승인 뉴스)
  - ClinicalTrials Fetcher (임상시험)
  - bioRxiv/medRxiv Fetcher (프리프린트)
  - PubMed Fetcher (peer-reviewed 논문)
- [x] **feat**: Daily Briefing UI 컴포넌트 & API 라우트
- [x] **feat**: 하이브리드 핫토픽 시스템 (키워드 매칭)
- [x] **feat**: 뉴스레터 개선 (중요도 배너, 임상 메타데이터, 순위 추적)
- [x] **feat**: 뉴스레터 뷰어 (탭 토글)
- [x] **feat**: 신문 스타일 뉴스레터 v3 + PDF 다운로드
- [x] **feat**: RNA-seq RAG 파이프라인 구현 (PubMedBERT + ChromaDB)
- [x] **docs**: CLAUDE.md 업데이트 (Daily Briefing)

---

## Phase 4: RNA-seq 6-Agent Pipeline (2026-01-06 ~ 2026-01-09)

### 2026-01-06 (Day 12) - 파이프라인 모듈화
- [x] **feat**: 향상된 RNA-seq 파이프라인 모듈 (시각화 & 검증)
- [x] **feat**: RNA-seq Weekly Report PPT + RAG 아키텍처 슬라이드
- [x] **docs**: rnaseq-cancer-analyst.md 업데이트 (v2.0 모듈)

### 2026-01-07 (Day 13) - 6-Agent 파이프라인 완성
- [x] **feat**: RNA-seq 6-Agent Pipeline v2.0 보수적 해석
  - Agent 1: DESeq2 DEG 분석
  - Agent 2: Co-expression Network & Hub Gene
  - Agent 3: GO/KEGG Pathway Enrichment
  - Agent 4: DB Validation (DisGeNET, OMIM, COSMIC)
  - Agent 5: Visualization (Volcano, Heatmap, Network)
  - Agent 6: HTML Report Generation

### 2026-01-08 (Day 14) - 테스트 & CI/CD
- [x] **feat**: CI/CD, 테스트, 코드 품질 개선
- [x] **test**: 핵심 모듈 단위 테스트 추가
- [x] **docs**: CLAUDE.md & PRD.md 동기화

### 2026-01-09 (Day 15) - ML & RAG 완성
- [x] **feat**: TCGA-BRCA ML 분류기 (전체 데이터셋 1,231 샘플)
  - CatBoost 모델
  - SHAP 설명력
  - AUC 0.998 달성
- [x] **feat**: RAG 기반 문헌 해석 (RNA-seq 파이프라인)
  - Claude API + PubMedBERT Vector Search
  - PMID 인용 포함
- [x] **feat**: HTML 리포트 Visual Dashboard에 Heatmap 추가
- [x] **feat**: RNA-seq 시각화 개선 & 아키텍처 문서
- [x] **fix**: requirements.txt 의존성 추가 (beautifulsoup4, lxml, rank-bm25)
- [x] **docs**: CLAUDE.md - RAG and ML now complete

---

## Phase 5: i18n & UI Polish (2026-01-10)

### 2026-01-10 (Day 16) - 국제화 & UI 개선
- [x] **feat**: 전체 i18n 지원 & Research Tools 카드 레이아웃 수정
- [x] **feat**: 트렌딩 논문 한국어 번역
- [x] **feat**: Daily Briefing 멀티소스 데이터 포맷 변환 수정
- [x] **fix**: 번역 서비스 변경 (Gemini → Claude API)

---

## Phase 6: Pipeline Testing & 3D Visualization (2026-01-11)

### 2026-01-11 (Day 17) - 파이프라인 검증 & 3D 네트워크
- [x] **test**: TCGA-BRCA 데이터로 전체 파이프라인 테스트
  - 100 샘플 (91 Tumor, 9 Normal) 다운로드
  - ML 모델 검증: 99% 정확도, 100% 민감도, 88.9% 특이도
  - 6-Agent 파이프라인 실행 성공
- [x] **fix**: DESeq2 결과 컬럼 처리 (apeglm shrinkage 5컬럼 대응)
  - `agent1_deg.py` 동적 컬럼 매핑 구현
- [x] **fix**: RAG 유전자 선택 로직 수정
  - 기존: interpretation_score 기준 (DB 매칭 우선) → Hub 65%
  - 수정: **Hub gene 우선** → Hub 100%
  - `agent4_validation.py` 수정
- [x] **feat**: Network 시각화 개선
  - ENSG ID → **유전자 이름** 표시
  - 모든 유전자 라벨 표시
- [x] **feat**: Galaxy 스타일 2D 네트워크
  - 어두운 우주 배경 (#0d1117)
  - Hub gene glow 효과
  - 색상 코딩 (Up: 빨강, Down: 시안)
- [x] **feat**: Obsidian 스타일 3D 인터랙티브 네트워크
  - 3d-force-graph + Three.js 사용
  - 파티클 애니메이션
  - 드래그 회전, 스크롤 줌
  - 노드 호버 툴팁
  - 클릭 시 카메라 포커스
  - 컨트롤 버튼 (Labels, Particles, Reset)
- [x] **feat**: RAG 결과 한국어 뷰어 생성

---

## 주요 마일스톤 요약

| 날짜 | 마일스톤 | 상태 |
|------|----------|------|
| 2025-12-26 | 프로젝트 초기화 | ✅ 완료 |
| 2025-12-29 | React 프론트엔드 + 실시간 검색 | ✅ 완료 |
| 2025-12-30 | Knowledge Graph & 아키텍처 정리 | ✅ 완료 |
| 2026-01-02 | BIO 연구 데일리 | ✅ 완료 |
| 2026-01-05 | Daily Briefing 자동화 | ✅ 완료 |
| 2026-01-07 | RNA-seq 6-Agent Pipeline | ✅ 완료 |
| 2026-01-09 | ML 분류기 + RAG 해석 | ✅ 완료 |
| 2026-01-11 | 파이프라인 검증 + 3D 시각화 | ✅ 완료 |

---

## 기술 스택 변화

```
초기 (12/26)                    현재 (01/11)
─────────────────────────────────────────────────
Backend:
  - FastAPI                     - FastAPI
  - ChromaDB                    - ChromaDB + BM25 Hybrid
  - PubMedBERT                  - PubMedBERT + Cross-encoder

Frontend:
  - (없음)                      - React + Vite + Tailwind
                                - 3d-force-graph (Three.js)
                                - Plotly

Analysis:
  - (없음)                      - DESeq2 (R via rpy2)
                                - CatBoost + SHAP
                                - NetworkX
                                - Enrichr API

AI/LLM:
  - (없음)                      - Claude API
                                - Gemini API
```

---

## 파일 구조 변화

```
초기 구조                       현재 구조
─────────────────────────────────────────────────
/                               /
├── src/                        ├── backend/
│   └── (분산된 코드)           │   └── app/
│                               │       ├── api/routes/
│                               │       └── core/
│                               │
│                               ├── frontend/
│                               │   └── react_app/
│                               │
│                               ├── bio-daily-briefing/
│                               │
│                               ├── rnaseq_pipeline/
│                               │   ├── agents/ (6 agents)
│                               │   ├── ml/
│                               │   ├── rag/
│                               │   └── utils/
│                               │
│                               ├── models/
│                               │   └── rnaseq/breast/
│                               │
│                               └── docs/
```

---

## 다음 계획 (TODO)

### 단기 (1-2주)
- [ ] Proteomics 분석 모듈
- [ ] Genomics 변이 분석
- [ ] RNA-seq API 엔드포인트

### 중기 (1개월)
- [ ] Drug Discovery 모듈
- [ ] GRNFormer 통합 (GPU)
- [ ] Multi-cancer 모델

### 장기
- [ ] 사용자 인증 시스템
- [ ] 클라우드 배포 (AWS/GCP)
- [ ] 실시간 협업 기능
