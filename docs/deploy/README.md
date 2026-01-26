# BioInsight AI - 배포 문서

## 📄 문서 목록

| 문서 | 설명 |
|------|------|
| [DEPLOY.md](./DEPLOY.md) | 배포 가이드 (로컬/GCP/Docker) |
| [DATABASE.md](./DATABASE.md) | 데이터베이스 (ChromaDB) 구성 |

---

## 🚀 빠른 시작

```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env에 API 키 입력 (OpenAI, Anthropic 필수)

# 2. 패키지 설치
pip install -r requirements.txt
cd frontend/react_app && npm install

# 3. 서버 실행
uvicorn backend.app.main:app --port 8000
```

---

## 📊 핵심 정보 요약

### 시스템 요구사항

| 항목 | 값 |
|------|-----|
| Python | 3.11+ |
| Node.js | 20+ |
| R | 4.3+ |
| RAM | 4GB (최소) / 8GB (권장) |

### 필수 API 키

| API | 용도 |
|-----|------|
| OpenAI | 기본 LLM ✅ |
| Anthropic | RAG 해석 ✅ |

### 데이터베이스

| 항목 | 값 |
|------|-----|
| DB 종류 | ChromaDB (벡터 DB) |
| 저장 방식 | **로컬 파일** (별도 서버 불필요) |
| 저장 경로 | `./data/chroma_db/` |
| 용량 | ~2.2GB |

> **중요**: ChromaDB는 백엔드 실행 시 자동으로 로드됩니다. 별도 설치/실행이 필요 없습니다.

### 배포 시 필요 데이터

| 폴더 | 용량 | 필수 |
|------|------|------|
| `data/chroma_db/` | 2.2GB | ✅ |
| `models/rnaseq/` | ~200MB | ✅ |

---

## 🔗 관련 문서

- [GCP 배포 상세 가이드](../../deploy/gcp/README.md)
- [프로젝트 구조](../PROJECT_STRUCTURE.md)
- [프로젝트 분석](../PROJECT_ANALYSIS.md)
